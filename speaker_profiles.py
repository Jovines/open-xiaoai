#!/usr/bin/env python3
"""Local, privacy-preserving speaker profile storage for Open-XiaoAI.

Embeddings never leave the recorder host.  Events only contain an anonymous
profile id and a bounded similarity score.  A human label is exposed only when
the current sample also passes the live-source gate; a historical voice match
alone is deliberately insufficient because a phone or television can replay a
known person's voice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable


SCHEMA_VERSION = 1
MODEL_NAME = "3D-Speaker-zh"
MAX_PROFILES = 256
MAX_REPRESENTATIVE_SAMPLES = 5


def _unit(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else []


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = _unit(left), _unit(right)
    if not a or len(a) != len(b):
        return -1.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def profile_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:32] or "anonymous"


def speaker_overlap_ratio(segments: list[dict], playback_intervals: list[tuple[float, float]]) -> float:
    total = sum(max(0.0, float(item["end_seconds"]) - float(item["start_seconds"])) for item in segments)
    if total <= 0:
        return 0.0
    overlap = 0.0
    for item in segments:
        start, end = float(item["start_seconds"]), float(item["end_seconds"])
        overlap += sum(max(0.0, min(end, right) - max(start, left)) for left, right in playback_intervals)
    return min(1.0, overlap / total)


class SpeakerProfileStore:
    def __init__(self, path: Path, *, threshold: float = 0.82) -> None:
        self.path = Path(path)
        self.threshold = max(0.5, min(0.99, float(threshold)))

    def load(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("schema_version") == SCHEMA_VERSION and isinstance(value.get("profiles"), list):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return {"schema_version": SCHEMA_VERSION, "model": MODEL_NAME, "profiles": []}

    def save(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.part")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)
        self.path.chmod(0o600)

    @staticmethod
    def _profile_id(event_id: str, local_speaker_id: str, embedding: list[float]) -> str:
        digest = hashlib.sha256(json.dumps({
            "event_id": event_id,
            "speaker": local_speaker_id,
            "embedding_prefix": [round(value, 6) for value in embedding[:16]],
        }, sort_keys=True).encode("utf-8")).hexdigest()
        return f"household-speaker-anonymous-{digest[:12]}"

    @staticmethod
    def _quality_gate(sample: dict, *, playback_overlap: float, scene: dict) -> tuple[bool, list[str]]:
        quality = sample.get("quality") or {}
        reasons = []
        if float(quality.get("speech_seconds") or 0) < 1.5:
            reasons.append("speech_shorter_than_1.5s")
        if float(quality.get("rms_dbfs") or -120) < -45:
            reasons.append("speech_below_-45dbfs")
        if float(quality.get("clipping_fraction") or 0) > 0.01:
            reasons.append("clipping_above_1_percent")
        if playback_overlap >= 0.05:
            reasons.append("xiaoai_playback_overlap")
        primary = str(scene.get("primary") or "unknown")
        if primary not in {"live_conversation", "live_monologue_or_reading"}:
            reasons.append("scene_not_live_speech_primary")
        return not reasons, reasons

    def assign(
        self,
        *,
        event_id: str,
        samples: list[dict],
        speaker_segments: list[dict],
        playback_intervals: list[tuple[float, float]],
        scene: dict,
        audio_refs: list[dict],
    ) -> dict[str, dict]:
        data = self.load()
        profiles = data["profiles"]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        assignments: dict[str, dict] = {}
        changed = False

        for sample in samples:
            local_speaker_id = f"recording-speaker-{int(sample.get('speaker', -1)):02d}"
            embedding = _unit(sample.get("embedding") or [])
            segments = [item for item in speaker_segments if item.get("speaker_id") == local_speaker_id]
            playback_overlap = speaker_overlap_ratio(segments, playback_intervals)
            eligible, gate_reasons = self._quality_gate(sample, playback_overlap=playback_overlap, scene=scene)
            if not embedding or not segments or not eligible:
                assignments[local_speaker_id] = {
                    "profile_id": None,
                    "similarity": None,
                    "identity_label": None,
                    "identity_status": "profile_update_rejected",
                    "profile_update_reasons": gate_reasons or ["embedding_unavailable"],
                }
                continue

            compatible = [item for item in profiles if item.get("model") == MODEL_NAME and len(item.get("centroid") or []) == len(embedding)]
            best = max(compatible, key=lambda item: cosine_similarity(item["centroid"], embedding), default=None)
            similarity = cosine_similarity(best["centroid"], embedding) if best else -1.0
            if best is None or similarity < self.threshold:
                if len(profiles) >= MAX_PROFILES:
                    assignments[local_speaker_id] = {
                        "profile_id": None,
                        "similarity": round(max(-1.0, similarity), 4) if best else None,
                        "identity_label": None,
                        "identity_status": "profile_capacity_reached",
                        "profile_update_reasons": ["manual_profile_maintenance_required"],
                    }
                    continue
                best = {
                    "profile_id": self._profile_id(event_id, local_speaker_id, embedding),
                    "model": MODEL_NAME,
                    "centroid": embedding,
                    "sample_count": 0,
                    "total_speech_seconds": 0.0,
                    "identity_label": None,
                    "label_status": "anonymous",
                    "created_at": now,
                    "updated_at": now,
                    "representative_samples": [],
                }
                profiles.append(best)
                similarity = 1.0
            else:
                old_count = max(1, int(best.get("sample_count") or 1))
                # A slow, bounded centroid update limits damage from a single bad match.
                weight = min(0.2, 1.0 / (old_count + 1))
                best["centroid"] = _unit([(1 - weight) * old + weight * new for old, new in zip(best["centroid"], embedding)])

            quality = sample.get("quality") or {}
            representative = {
                "event_id": event_id,
                "local_speaker_id": local_speaker_id,
                "audio_refs": audio_refs[:4],
                "segments": [{"start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"]} for item in segments[:12]],
                "quality": quality,
                "embedding": embedding,
                "recorded_at": now,
            }
            best["sample_count"] = int(best.get("sample_count") or 0) + 1
            best["total_speech_seconds"] = round(float(best.get("total_speech_seconds") or 0) + float(quality.get("speech_seconds") or 0), 3)
            best["updated_at"] = now
            best["representative_samples"] = [*(best.get("representative_samples") or []), representative][-MAX_REPRESENTATIVE_SAMPLES:]
            changed = True

            replay_score = scene.get("signals", {}).get("replay_score")
            live_confidence = next((float(item.get("probability") or 0) for item in scene.get("candidates", []) if item.get("label") in {"live_conversation", "live_monologue_or_reading"}), 0.0)
            named_safe = best.get("label_status") == "human_confirmed" and replay_score is not None and float(replay_score) <= 0.2 and live_confidence >= 0.75
            assignments[local_speaker_id] = {
                "profile_id": best["profile_id"],
                "similarity": round(max(0.0, similarity), 4),
                "identity_label": best.get("identity_label") if named_safe else None,
                "identity_status": "human_confirmed" if named_safe else ("named_candidate_needs_source_review" if best.get("identity_label") else "anonymous_candidate"),
                "profile_update_reasons": [],
            }

        if changed:
            data["updated_at"] = now
            self.save(data)
        return assignments

    def label(self, profile_id: str, label: str, confirmed_by: str) -> dict:
        if confirmed_by != "household-owner":
            raise ValueError("speaker labels require --confirmed-by household-owner")
        clean_label = " ".join(str(label).split())[:80]
        if not clean_label:
            raise ValueError("label cannot be empty")
        data = self.load()
        profile = next((item for item in data["profiles"] if item.get("profile_id") == profile_id), None)
        if not profile:
            raise KeyError(profile_id)
        profile["identity_label"] = clean_label
        profile["label_status"] = "human_confirmed"
        profile["label_confirmed_by"] = confirmed_by
        profile["label_confirmed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.save(data)
        return profile

    def unlabel(self, profile_id: str) -> dict:
        data = self.load()
        profile = next((item for item in data["profiles"] if item.get("profile_id") == profile_id), None)
        if not profile:
            raise KeyError(profile_id)
        profile["identity_label"] = None
        profile["label_status"] = "anonymous"
        profile.pop("label_confirmed_by", None)
        profile.pop("label_confirmed_at", None)
        self.save(data)
        return profile


def public_profile(profile: dict) -> dict:
    return {
        "profile_id": profile.get("profile_id"),
        "identity_label": profile.get("identity_label"),
        "label_status": profile.get("label_status"),
        "sample_count": profile.get("sample_count"),
        "total_speech_seconds": profile.get("total_speech_seconds"),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "representative_samples": [{key: item.get(key) for key in ("event_id", "audio_refs", "segments", "quality", "recorded_at")} for item in profile.get("representative_samples", [])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local anonymous household speaker profiles")
    parser.add_argument("--store", type=Path, default=Path.home() / ".local/share/open-xiaoai/speaker-profiles.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    label = sub.add_parser("label")
    label.add_argument("profile_id")
    label.add_argument("label")
    label.add_argument("--confirmed-by", required=True)
    unlabel = sub.add_parser("unlabel")
    unlabel.add_argument("profile_id")
    args = parser.parse_args()
    store = SpeakerProfileStore(args.store)
    if args.command == "list":
        output = [public_profile(item) for item in store.load()["profiles"]]
    elif args.command == "label":
        output = public_profile(store.label(args.profile_id, args.label, args.confirmed_by))
    else:
        output = public_profile(store.unlabel(args.profile_id))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
