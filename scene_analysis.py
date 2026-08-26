"""Conservative acoustic-scene fusion for household audio evidence."""

from __future__ import annotations

import math
from pathlib import Path
import re

SCENE_LABELS = (
    "live_conversation",
    "live_monologue_or_reading",
    "xiaoai_dialogue",
    "xiaoai_playback",
    "phone_language_learning_playback",
    "phone_or_computer_media_playback",
    "television_or_remote_media",
    "mixed_live_and_playback",
    "non_speech_household_sound",
    "unknown",
)


def _max_tag(tags: list[dict], needles: tuple[str, ...]) -> float:
    return max((float(item.get("probability") or 0) for item in tags if any(needle in str(item.get("label") or "").lower() for needle in needles)), default=0.0)


def _distribution(values: dict[str, float]) -> list[dict]:
    bounded = {label: max(0.0, float(values.get(label, 0.0))) for label in SCENE_LABELS}
    total = sum(bounded.values()) or 1.0
    return [
        {"label": label, "probability": round(value / total, 4)}
        for label, value in sorted(bounded.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ]


def english_text_ratio(transcript: str) -> float:
    latin = len(re.findall(r"[A-Za-z]", transcript or ""))
    chinese = len(re.findall(r"[\u4e00-\u9fff]", transcript or ""))
    return latin / max(1, latin + chinese)


def lexical_repetition_score(transcript: str) -> float:
    """A bounded clue for drill-like playback, never an application identity."""
    tokens = [
        item.lower() for item in re.findall(r"[A-Za-z']+", transcript or "")
        if len(item) > 2 and item.lower() not in {"the", "and", "you", "are", "for", "this", "that", "with"}
    ]
    if len(tokens) < 6:
        return 0.0
    return max(0.0, min(1.0, 1 - len(set(tokens)) / len(tokens)))


def infer_scene(base_scene: dict, audio_tagging: dict | None, transcript: str, array_features: dict | None = None) -> dict:
    tags = list(audio_tagging.get("tags") or []) if audio_tagging else []
    legacy = str(base_scene.get("scene_type") or "unknown")
    playback_available = bool(base_scene.get("playback_reference", {}).get("available"))
    coverage = float(base_scene.get("playback_reference", {}).get("coverage") or 0)

    if legacy == "xiaoai_dialogue":
        scores = {"xiaoai_dialogue": 0.82, "mixed_live_and_playback": 0.12, "unknown": 0.06}
    elif legacy == "device_playback":
        scores = {"xiaoai_playback": 0.9, "unknown": 0.1}
    elif legacy == "mixed":
        scores = {"mixed_live_and_playback": 0.7, "xiaoai_playback": 0.2, "unknown": 0.1}
    else:
        speech = _max_tag(tags, ("speech", "conversation", "narration", "monologue", "child speech"))
        television = _max_tag(tags, ("television", "radio"))
        music = _max_tag(tags, ("music", "podcast", "video game"))
        media = max(television, music)
        narration = _max_tag(tags, ("narration", "monologue"))
        household = _max_tag(tags, ("domestic sounds", "inside, small room", "dishes", "cutlery", "vacuum", "water", "door"))
        english_ratio = english_text_ratio(transcript)
        repetition = lexical_repetition_score(transcript)
        language_drill_clue = english_ratio * (0.55 * music + 0.25 * narration + 0.2 * repetition)
        if television >= 0.3:
            scores = {
                "television_or_remote_media": 0.36 + television * 0.25,
                "phone_or_computer_media_playback": 0.24 + television * 0.15,
                "mixed_live_and_playback": 0.08 + min(0.12, speech * 0.12),
                "unknown": 0.2,
            }
        elif language_drill_clue >= 0.25:
            scores = {
                "phone_language_learning_playback": 0.34 + min(0.18, language_drill_clue * 0.35),
                "live_monologue_or_reading": 0.14,
                "phone_or_computer_media_playback": 0.14,
                "unknown": 0.24,
            }
        elif media >= max(0.3, speech * 0.9):
            scores = {
                "phone_or_computer_media_playback": 0.36 + media * 0.25,
                "television_or_remote_media": 0.18 + media * 0.1,
                "mixed_live_and_playback": 0.08 + min(0.12, speech * 0.12),
                "unknown": 0.2,
            }
        elif speech >= 0.25:
            live_label = "live_monologue_or_reading" if english_ratio >= 0.35 else "live_conversation"
            scores = {
                live_label: 0.42 + min(0.18, speech * 0.18),
                "phone_or_computer_media_playback": 0.14,
                "phone_language_learning_playback": 0.08 if english_ratio >= 0.35 else 0.02,
                "unknown": 0.3,
            }
        elif household >= 0.25:
            scores = {"non_speech_household_sound": 0.55, "unknown": 0.45}
        else:
            scores = {"unknown": 1.0}

    candidates = _distribution(scores)
    primary = candidates[0]["label"]
    primary_probability = candidates[0]["probability"]
    signals = {
        "xiaoai_playback_reference": playback_available,
        "xiaoai_playback_coverage": round(coverage, 4) if playback_available else None,
        "audio_tags": tags[:12],
        "audio_tagging_status": "available" if audio_tagging else "models_unavailable",
        "replay_score": None,
        "array_spatial_features": array_features or {},
        "authorized_device_context": [],
        "custom_household_classifier": {"status": "awaiting_labeled_household_samples"},
        "transcript_pattern_clues": {
            "english_ratio": round(english_text_ratio(transcript), 4),
            "lexical_repetition": round(lexical_repetition_score(transcript), 4),
        },
    }
    versions = ["rule-fusion-v1"]
    if audio_tagging:
        versions.append(str(audio_tagging.get("model") or "sherpa-onnx/CED-mini-int8"))
    return {
        "primary": primary,
        "confidence": primary_probability,
        "candidates": candidates,
        "signals": signals,
        "model_versions": versions,
        "mode": "candidate",
        "needs_review": True,
        "limitations": [
            "phone_app_identity_requires_household_domain_training_or_authorized_device_context",
            "named_speaker_identity_requires_replay_gate",
        ],
    }


def array_spatial_features(audio_paths: list[Path], *, maximum_seconds: float = 8.0, maximum_lag_ms: float = 2.0) -> dict:
    """Compute geometry-independent 3-mic cues without claiming a direction."""
    import numpy as np
    import soundfile as sf

    if len(audio_paths) != 3:
        return {"status": "requires_three_channels"}
    channels = []
    sample_rate = None
    for path in audio_paths:
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate is not None and rate != sample_rate:
            return {"status": "sample_rate_mismatch"}
        sample_rate = rate
        channels.append(np.ascontiguousarray(values[:round(rate * maximum_seconds), 0]))
    length = min((len(item) for item in channels), default=0)
    if not sample_rate or length < sample_rate // 2:
        return {"status": "insufficient_audio"}
    channels = [item[:length].astype(np.float64) for item in channels]
    rms = [math.sqrt(float(np.mean(item * item))) for item in channels]
    rms_dbfs = [20 * math.log10(max(value, 1e-12)) for value in rms]
    maximum_lag = max(1, round(sample_rate * maximum_lag_ms / 1000))
    pairs = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = channels[left_index] - np.mean(channels[left_index])
        right = channels[right_index] - np.mean(channels[right_index])
        best = (-1.0, 0)
        for lag in range(-maximum_lag, maximum_lag + 1):
            if lag < 0:
                a, b = left[-lag:], right[:length + lag]
            elif lag > 0:
                a, b = left[:length - lag], right[lag:]
            else:
                a, b = left, right
            denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
            correlation = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
            if abs(correlation) > abs(best[0]) or best[0] == -1.0:
                best = (correlation, lag)
        pairs.append({
            "channels": [left_index, right_index],
            "delay_ms": round(best[1] * 1000 / sample_rate, 4),
            "peak_correlation": round(best[0], 4),
        })
    return {
        "status": "geometry_unconfigured",
        "sample_seconds": round(length / sample_rate, 3),
        "channel_rms_dbfs": [round(value, 2) for value in rms_dbfs],
        "level_spread_db": round(max(rms_dbfs) - min(rms_dbfs), 2),
        "dominant_channel": max(range(3), key=lambda index: rms[index]),
        "pairwise": pairs,
        "direction": None,
    }
