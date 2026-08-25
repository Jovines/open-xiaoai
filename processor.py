#!/usr/bin/env python3
"""Turn raw XiaoAI array recordings into auditable household speech evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


FILENAME_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[+-]\d{4})")
SUPPORTED_AUDIO = {".flac", ".wav", ".ogg"}


class EventPostError(RuntimeError):
    """An ingest failure with enough information to decide whether to retry."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class TranscriptionTimeout(RuntimeError):
    """The primary ASR exceeded the per-recording CPU time budget."""

    def __init__(self, message: str, *, timeout_seconds: float) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛掉无语音录音，并生成可回听、可审计的中文转写证据")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("PROCESSOR_INPUT_DIR", "/mnt/dx4600/家庭管家/录音/inbox")))
    parser.add_argument("--evidence-dir", type=Path, default=Path(os.environ.get("PROCESSOR_EVIDENCE_DIR", "/mnt/dx4600/家庭管家/录音/evidence")))
    parser.add_argument("--discard-dir", type=Path, default=Path(os.environ.get("PROCESSOR_DISCARD_DIR", "/mnt/dx4600/家庭管家/录音/discarded")))
    parser.add_argument("--required-mount", type=Path, default=Path(os.environ.get("PROCESSOR_REQUIRED_MOUNT", "/mnt/dx4600")))
    parser.add_argument("--nas-uri-root", default=os.environ.get("PROCESSOR_NAS_URI_ROOT", "nas://dx4600/家庭管家/录音/evidence"))
    parser.add_argument("--playback-dir", type=Path, default=Path(os.environ.get("PROCESSOR_PLAYBACK_DIR", "/mnt/dx4600/家庭管家/录音/playback")))
    parser.add_argument("--playback-nas-uri-root", default=os.environ.get("PROCESSOR_PLAYBACK_NAS_URI_ROOT", "nas://dx4600/家庭管家/录音/playback"))
    parser.add_argument("--sense-binary", type=Path, default=Path(os.environ.get("SENSEVOICE_BINARY", str(Path.home() / ".local/opt/sensevoice/llama-funasr-sensevoice"))))
    parser.add_argument("--vad-binary", type=Path, default=Path(os.environ.get("FSMN_VAD_BINARY", str(Path.home() / ".local/opt/sensevoice/llama-funasr-vad"))))
    parser.add_argument("--sense-model", type=Path, default=Path(os.environ.get("SENSEVOICE_MODEL", str(Path.home() / "models/sensevoice-small/sensevoice-small-q8.gguf"))))
    parser.add_argument("--vad-model", type=Path, default=Path(os.environ.get("SENSEVOICE_VAD_MODEL", str(Path.home() / "models/sensevoice-small/fsmn-vad.gguf"))))
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--qwen-timeout-seconds", type=float, default=float(os.environ.get("QWEN_ASR_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--qwen-max-new-tokens", type=int, default=int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "384")))
    parser.add_argument("--firered-enabled", action=argparse.BooleanOptionalAction, default=os.environ.get("FIRERED_ASR_ENABLED", "0").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--firered-python", type=Path, default=Path(os.environ.get("FIRERED_ASR_PYTHON", sys.executable)))
    parser.add_argument("--firered-script", type=Path, default=Path(os.environ.get("FIRERED_ASR_SCRIPT", str(Path(__file__).parent / "scripts/firered_transcribe.py"))))
    parser.add_argument("--firered-source-dir", type=Path, default=Path(os.environ.get("FIRERED_ASR_SOURCE_DIR", str(Path.home() / ".local/opt/fireredasr2s"))))
    parser.add_argument("--firered-deps-dir", type=Path, default=Path(os.environ.get("FIRERED_ASR_DEPS_DIR", str(Path.home() / ".local/opt/fireredasr2-deps-py312"))))
    parser.add_argument("--firered-model-dir", type=Path, default=Path(os.environ.get("FIRERED_ASR_MODEL_DIR", str(Path.home() / "models/fireredasr2-aed"))))
    parser.add_argument("--firered-timeout-seconds", type=float, default=float(os.environ.get("FIRERED_ASR_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--asr-max-timeout-seconds", type=float, default=float(os.environ.get("ASR_MAX_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--diarization-python", type=Path, default=Path(os.environ.get("DIARIZATION_PYTHON", str(Path.home() / ".venvs/sherpa-onnx/bin/python"))))
    parser.add_argument("--diarization-script", type=Path, default=Path(os.environ.get("DIARIZATION_SCRIPT", str(Path(__file__).with_name("diarize.py")))))
    parser.add_argument("--diarization-segmentation", type=Path, default=Path(os.environ.get("DIARIZATION_SEGMENTATION_MODEL", str(Path.home() / "models/speaker-diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"))))
    parser.add_argument("--diarization-embedding", type=Path, default=Path(os.environ.get("DIARIZATION_EMBEDDING_MODEL", str(Path.home() / "models/speaker-diarization/3dspeaker-zh.onnx"))))
    parser.add_argument("--zeris-url", default=os.environ.get("ZERIS_AUDIO_INGEST_URL", ""))
    parser.add_argument("--zeris-token", default=os.environ.get("ZERIS_AUDIO_INGEST_TOKEN", ""))
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("PROCESSOR_POLL_SECONDS", "3")))
    parser.add_argument("--settle-seconds", type=float, default=float(os.environ.get("PROCESSOR_SETTLE_SECONDS", "5")))
    parser.add_argument("--lookahead-wait-seconds", type=float, default=float(os.environ.get("PROCESSOR_LOOKAHEAD_WAIT_SECONDS", "75")))
    parser.add_argument("--discard-grace-hours", type=float, default=float(os.environ.get("PROCESSOR_DISCARD_GRACE_HOURS", "72")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=os.environ.get("PROCESSOR_LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def normalize_asr_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def normalized_agreement(left: str, right: str) -> float:
    a, b = normalize_asr_text(left), normalize_asr_text(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def asr_reliability(primary: str, alternatives: list[str]) -> dict:
    """Do not let a high fuzzy score hide a potentially critical word conflict."""
    active = [text for text in alternatives if normalize_asr_text(text)]
    comparisons = [normalized_agreement(primary, text) for text in active]
    score = sum(comparisons) / len(comparisons) if comparisons else 0.0
    exact_consensus = len(active) == 3 and all(normalize_asr_text(primary) == normalize_asr_text(text) for text in active)
    if exact_consensus:
        agreement = "high"
    elif score >= 0.65:
        agreement = "medium"
    else:
        agreement = "low"
    return {
        "agreement": agreement,
        "score": round(score, 4),
        "needs_review": not exact_consensus,
        "notes": (
            "两种 ASR 与三路麦克风文本完全一致；机器转写仍可能听错，高影响事项必须复核。"
            if exact_consensus
            else "ASR 候选存在文本冲突；不得按整体相似度忽略单字差异，高影响事项必须回听或询问。"
        ),
    }


def sensevoice_consensus(alternatives: list[str]) -> str | None:
    """Return the representative text only when all three microphones agree exactly."""
    active = [text for text in alternatives if normalize_asr_text(text)]
    if len(active) != 3:
        return None
    normalized = normalize_asr_text(active[0])
    return active[0] if all(normalize_asr_text(text) == normalized for text in active[1:]) else None


def apply_asr_adjudication(primary: str, alternatives: list[str], adjudicator_text: str | None) -> tuple[str, str]:
    """Choose only when an independent adjudicator breaks a strong model conflict."""
    consensus = sensevoice_consensus(alternatives)
    if consensus is None or normalize_asr_text(primary) == normalize_asr_text(consensus):
        return primary, "not_applicable"
    if not adjudicator_text:
        return primary, "unavailable"
    normalized_adjudicator = normalize_asr_text(adjudicator_text)
    if normalized_adjudicator == normalize_asr_text(consensus):
        return consensus, "sensevoice_consensus_confirmed"
    if normalized_adjudicator == normalize_asr_text(primary):
        return primary, "primary_confirmed"
    return primary, "three_way_conflict"


def asr_timeout_budget(audio: Path, minimum_seconds: float, maximum_seconds: float = 300) -> float:
    """Allow long speech proportionally more CPU time while retaining a hang watchdog."""
    minimum = max(0.01, minimum_seconds)
    maximum = max(minimum, maximum_seconds)
    try:
        duration = float(audio_properties(audio)["duration_seconds"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, json.JSONDecodeError):
        return minimum
    return min(maximum, max(minimum, 30 + duration * 3))


def acoustic_scene_without_reference() -> dict:
    """State explicitly that speaker-origin attribution has not been performed."""
    return {
        "scene_type": "unknown",
        "interaction_id": None,
        "confidence": None,
        "needs_review": True,
        "playback_reference": {
            "available": False,
            "coverage": None,
            "source": None,
        },
        "turns": [],
    }


def playback_matches_for_window(
    occurred: dt.datetime,
    ended: dt.datetime,
    playback_dir: Path,
    nas_uri_root: str,
) -> list[dict]:
    """Find archived playback PCM that overlaps an absolute microphone event."""
    if not playback_dir.is_dir():
        return []
    local_start = occurred.astimezone()
    local_end = ended.astimezone()
    dates = {local_start.date(), local_end.date()}
    matches = []
    for date in dates:
        directory = playback_dir / date.strftime("%Y/%m/%d")
        for manifest_path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("kind") != "oh2p_playback_reference":
                    continue
                reference_start = dt.datetime.fromisoformat(manifest["occurred_at"])
                reference_end = dt.datetime.fromisoformat(manifest["ended_at"])
                audio = manifest_path.with_name(manifest["audio_file"])
                sha256 = str(manifest["audio_sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
            if not audio.is_file() or not re.fullmatch(r"[a-f0-9]{64}", sha256):
                continue
            overlap_start = max(occurred, reference_start)
            overlap_end = min(ended, reference_end)
            if overlap_end <= overlap_start:
                continue
            relative = audio.relative_to(playback_dir)
            matches.append({
                "uri": f"{nas_uri_root.rstrip('/')}/{relative.as_posix()}",
                "sha256": sha256,
                "offset_start_seconds": round((overlap_start - reference_start).total_seconds(), 3),
                "offset_end_seconds": round((overlap_end - reference_start).total_seconds(), 3),
                "event_start_seconds": round((overlap_start - occurred).total_seconds(), 3),
                "event_end_seconds": round((overlap_end - occurred).total_seconds(), 3),
            })
    return sorted(matches, key=lambda item: (item["event_start_seconds"], item["uri"]))[:4]


def acoustic_scene_with_playback(
    identity: str,
    event_duration_seconds: float,
    speech_segments_seconds: list[tuple[float, float]],
    playback_matches: list[dict],
) -> dict:
    if not playback_matches:
        return acoustic_scene_without_reference()
    playback_intervals = [(item["event_start_seconds"], item["event_end_seconds"]) for item in playback_matches]
    playback_seconds = sum(max(0.0, end - start) for start, end in playback_intervals)
    turns = [{
        "origin": "xiaoai_output",
        "speaker_id": None,
        "start_seconds": start,
        "end_seconds": end,
        "confidence": 1.0,
        "transcript": None,
        "evidence": ["archived_playback_reference"],
    } for start, end in playback_intervals]
    human_turns = []
    unknown_turns = []
    for start, end in speech_segments_seconds:
        duration = max(0.001, end - start)
        overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in playback_intervals)
        turn = {
            "origin": "human" if overlap / duration < 0.1 else "unknown",
            "speaker_id": None,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "confidence": 0.8 if overlap / duration < 0.1 else 0.5,
            "transcript": None,
            "evidence": ["fsmn_vad", "no_device_playback_overlap"] if overlap / duration < 0.1 else ["fsmn_vad", "device_playback_temporal_overlap", "aec_not_yet_applied"],
        }
        turns.append(turn)
        (human_turns if turn["origin"] == "human" else unknown_turns).append(turn)
    first_playback = min(start for start, _ in playback_intervals)
    dialogue_lead = any(0 <= first_playback - item["end_seconds"] <= 5 for item in human_turns)
    if dialogue_lead:
        scene_type = "xiaoai_dialogue"
    elif human_turns or unknown_turns:
        scene_type = "mixed"
    else:
        scene_type = "device_playback"
    return {
        "scene_type": scene_type,
        "interaction_id": f"xiaoai-dialogue:{identity[:20]}" if dialogue_lead else f"playback-scene:{identity[:20]}",
        "confidence": 0.75 if dialogue_lead else 0.7,
        "needs_review": bool(dialogue_lead or unknown_turns),
        "playback_reference": {
            "available": True,
            "coverage": round(min(1.0, playback_seconds / max(0.001, event_duration_seconds)), 4),
            "source": "oh2p-alsa-default-playback-pcm",
        },
        "turns": sorted(turns, key=lambda item: (item["start_seconds"], item["origin"])),
    }


def choose_microphone(transcripts: list[str]) -> int:
    """Choose the transcript most similar to the other active microphones."""
    if len(transcripts) != 3:
        raise ValueError("expected exactly three microphone transcripts")
    scores = [
        sum(normalized_agreement(text, other) for j, other in enumerate(transcripts) if i != j)
        for i, text in enumerate(transcripts)
    ]
    return max(range(3), key=lambda index: scores[index])


def choose_microphone_result(results: list[tuple[int, str]]) -> int:
    """Prefer cross-channel agreement, falling back to the strongest VAD result."""
    texts = [text for _, text in results]
    scores = [
        sum(normalized_agreement(text, other) for j, other in enumerate(texts) if i != j)
        for i, text in enumerate(texts)
    ]
    if max(scores, default=0.0) > 0:
        return max(range(3), key=lambda index: scores[index])
    return max(range(3), key=lambda index: (results[index][0], len(results[index][1])))


def parse_capture_time(path: Path) -> dt.datetime:
    match = FILENAME_TIME.match(path.stem)
    if match:
        return dt.datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S_%z")
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).astimezone()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_properties(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    return {
        "codec": path.suffix.lower().lstrip("."),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bits_per_sample": int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0),
        "duration_seconds": float(value["format"]["duration"]),
    }


def extract_microphones(source: Path, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for channel in range(3):
        output = directory / f"mic-{channel}.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map_channel", f"0.0.{channel}", "-ar", "16000", "-ac", "1", str(output)],
            check=True,
        )
        outputs.append(output)
    return outputs


def concatenate_audio(sources: list[Path], destination: Path) -> None:
    if not sources:
        raise ValueError("cannot concatenate an empty audio list")
    if len(sources) == 1:
        shutil.copyfile(sources[0], destination)
        return
    command = ["ffmpeg", "-v", "error", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    inputs = "".join(f"[{index}:a]" for index in range(len(sources)))
    command.extend([
        "-filter_complex", f"{inputs}concat=n={len(sources)}:v=0:a=1[out]",
        "-map", "[out]", "-ar", "16000", "-ac", "1", str(destination),
    ])
    subprocess.run(command, check=True)


def sensevoice(binary: Path, model: Path, vad: Path, audio: Path) -> tuple[int, str]:
    result = subprocess.run(
        [str(binary), "-m", str(model), "--vad", str(vad), "-a", str(audio)],
        check=True, capture_output=True, text=True,
    )
    combined = "\n".join((result.stdout, result.stderr))
    match = re.search(r"\[sensevoice\]\s+(\d+)\s+vad segments", combined)
    segments = int(match.group(1)) if match else 0
    text_lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("[sensevoice]")]
    return segments, " ".join(text_lines)


def vad_segments(binary: Path, model: Path, audio: Path) -> list[tuple[int, int]]:
    """Return FSMN-VAD intervals as millisecond pairs."""
    result = subprocess.run(
        [str(binary), "-m", str(model), "-a", str(audio)],
        check=True, capture_output=True, text=True,
    )
    combined = "\n".join((result.stdout, result.stderr))
    segments = []
    for line in combined.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            if end > start:
                segments.append((start, end))
    return segments


def merge_vad_segments(segments: list[tuple[int, int]], padding_ms: int = 200) -> list[tuple[int, int]]:
    """Pad and merge speech intervals so consonants at VAD edges are not clipped."""
    merged: list[list[int]] = []
    for raw_start, raw_end in sorted(segments):
        start = max(0, raw_start - padding_ms)
        end = raw_end + padding_ms
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def extract_speech_audio(source: Path, destination: Path, segments: list[tuple[int, int]]) -> None:
    """Concatenate only VAD-positive regions into a compact ASR input."""
    merged = merge_vad_segments(segments)
    if not merged:
        raise ValueError("cannot extract speech without VAD segments")
    filters = []
    inputs = []
    for index, (start, end) in enumerate(merged):
        filters.append(
            f"[0:a]atrim=start={start / 1000:.3f}:end={end / 1000:.3f},"
            f"asetpts=PTS-STARTPTS[s{index}]"
        )
        inputs.append(f"[s{index}]")
    filters.append(f"{''.join(inputs)}concat=n={len(merged)}:v=0:a=1[out]")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-filter_complex", ";".join(filters), "-map", "[out]",
        "-ar", "16000", "-ac", "1", str(destination),
    ], check=True)


def owned_vad_segments(
    segments: list[tuple[int, int]],
    boundary_ms: int,
    carried_until_ms: int = 0,
) -> list[tuple[int, int]]:
    """Assign utterances by start time and suppress a continuation owned upstream."""
    owned = []
    for start, end in segments:
        if start >= boundary_ms:
            continue
        if carried_until_ms > 0 and start <= 500 and end <= carried_until_ms + 1000:
            continue
        owned.append((start, end))
    return owned


def audio_refs_for_segments(
    sources: list[Path],
    durations_ms: list[int],
    segments: list[tuple[int, int]],
    nas_uri_root: str,
) -> list[dict]:
    """Describe every immutable source block touched by the owned utterances."""
    if len(sources) != len(durations_ms):
        raise ValueError("source and duration counts differ")
    first = min(start for start, _ in segments)
    last = max(end for _, end in segments)
    refs = []
    cursor = 0
    for source, duration in zip(sources, durations_ms):
        overlap_start = max(first, cursor)
        overlap_end = min(last, cursor + duration)
        if overlap_end > overlap_start:
            captured = parse_capture_time(source)
            relative = Path(captured.strftime("%Y/%m/%d")) / source.name
            refs.append({
                "uri": f"{nas_uri_root.rstrip('/')}/{relative.as_posix()}",
                "sha256": sha256_file(source),
                "offset_start_seconds": round((overlap_start - cursor) / 1000, 3),
                "offset_end_seconds": round((overlap_end - cursor) / 1000, 3),
            })
        cursor += duration
    return refs


class EvidenceProcessor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.qwen = None
        self.pending_retry_attempts: dict[Path, int] = {}
        self.pending_retry_after: dict[Path, float] = {}

    def require_runtime(self) -> None:
        if not self.args.required_mount.is_mount():
            raise RuntimeError(f"所需存储未挂载：{self.args.required_mount}")
        for program in ("ffmpeg", "ffprobe"):
            if not shutil.which(program):
                raise RuntimeError(f"缺少依赖程序：{program}")
        for path in (self.args.sense_binary, self.args.vad_binary, self.args.sense_model, self.args.vad_model):
            if not path.is_file():
                raise RuntimeError(f"缺少模型或程序：{path}")
        if self.args.firered_enabled:
            required = (
                self.args.firered_python,
                self.args.firered_script,
                self.args.firered_source_dir / "fireredasr2s/fireredasr2/asr.py",
                self.args.firered_model_dir / "model.pth.tar",
            )
            for path in required:
                if not path.is_file():
                    raise RuntimeError(f"缺少 FireRed 仲裁程序或模型：{path}")
            if not self.args.firered_deps_dir.is_dir():
                raise RuntimeError(f"缺少 FireRed Python 依赖目录：{self.args.firered_deps_dir}")
        self.args.input_dir.mkdir(parents=True, exist_ok=True)
        self.args.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.args.discard_dir.mkdir(parents=True, exist_ok=True)

    def load_qwen(self) -> None:
        if self.qwen is not None:
            return
        logging.info("正在 CPU 加载 Qwen3-ASR-1.7B（不会占用 GPU）")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
        self.qwen = Qwen3ASRModel.from_pretrained(
            self.args.qwen_model,
            device_map="cpu",
            dtype="float32",
            max_inference_batch_size=1,
            max_new_tokens=self.args.qwen_max_new_tokens,
        )

    def qwen_transcribe(self, audio: Path) -> str:
        self.load_qwen()
        timeout = asr_timeout_budget(audio, self.args.qwen_timeout_seconds, self.args.asr_max_timeout_seconds)

        def timed_out(signum, frame):
            raise TranscriptionTimeout(f"Qwen 主转写超过动态预算 {timeout:g} 秒", timeout_seconds=timeout)

        previous_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            result = self.qwen.transcribe(str(audio), language="Chinese")
            return result[0].text.strip()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def firered_transcribe(self, audio: Path) -> dict | None:
        """Run the memory-heavy model out of process so its RAM is reclaimed."""
        if not self.args.firered_enabled:
            return None
        timeout = asr_timeout_budget(audio, self.args.firered_timeout_seconds, self.args.asr_max_timeout_seconds)
        try:
            result = subprocess.run([
                str(self.args.firered_python), str(self.args.firered_script),
                "--audio", str(audio),
                "--source-dir", str(self.args.firered_source_dir),
                "--deps-dir", str(self.args.firered_deps_dir),
                "--model-dir", str(self.args.firered_model_dir),
            ], check=True, capture_output=True, text=True, timeout=timeout)
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            payload = json.loads(lines[-1])
            return payload if isinstance(payload, dict) else None
        except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as error:
            logging.warning("FireRed 按需仲裁失败，保留 Qwen 主结果和 SenseVoice 候选：%s", error)
            return None

    def diarize(self, audio: Path) -> tuple[list[dict], str]:
        required = (
            self.args.diarization_python,
            self.args.diarization_script,
            self.args.diarization_segmentation,
            self.args.diarization_embedding,
        )
        if not all(path.is_file() for path in required):
            return [], "models_unavailable"
        try:
            result = subprocess.run([
                str(self.args.diarization_python), str(self.args.diarization_script), str(audio),
                "--segmentation", str(self.args.diarization_segmentation),
                "--embedding", str(self.args.diarization_embedding),
            ], check=True, capture_output=True, text=True, timeout=180)
            raw = json.loads(result.stdout)
            speakers = [{
                "speaker_id": f"recording-speaker-{int(item['speaker']):02d}",
                "label": None,
                "start_seconds": float(item["start_seconds"]),
                "end_seconds": float(item["end_seconds"]),
                "confidence": None,
            } for item in raw]
            return speakers, "anonymous_recording_clusters"
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            logging.warning("说话人分离失败但不阻断原音与转写：%s", error)
            return [], "failed"

    def pending_files(self) -> list[Path]:
        cutoff = time.time() - self.args.settle_seconds
        return sorted(
            path for path in self.args.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO and not path.name.startswith(".") and path.stat().st_mtime <= cutoff
        )

    @staticmethod
    def is_contiguous(left: Path, right: Path, tolerance_seconds: float = 3.0) -> bool:
        delta = (parse_capture_time(right) - parse_capture_time(left)).total_seconds()
        return 0 < delta <= 60 + tolerance_seconds

    def relative_destination(self, source: Path) -> Path:
        captured = parse_capture_time(source)
        return Path(captured.strftime("%Y/%m/%d")) / source.name

    def quarantine_audio(
        self,
        source: Path,
        *,
        classification: str,
        classifier: dict,
        pending_event: Path | None = None,
    ) -> Path:
        """Move an unhelpful recording to recoverable quarantine with an audit record."""
        digest = sha256_file(source)
        relative = self.relative_destination(source)
        quarantine = self.args.discard_dir / relative.with_suffix(relative.suffix + ".quarantine")
        manifest = self.args.discard_dir / relative.with_suffix(".json")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists() or manifest.exists():
            raise RuntimeError(f"隔离目标已存在，拒绝覆盖：{quarantine}")
        source.replace(quarantine)

        rejected_event = None
        if pending_event is not None and pending_event.is_file():
            rejected_event = quarantine.with_suffix(quarantine.suffix + ".event.rejected")
            pending_event.replace(rejected_event)

        classified_at = dt.datetime.now(dt.timezone.utc)
        record = {
            "schema_version": 1,
            "classification": classification,
            "source_sha256": digest,
            "source_bytes": quarantine.stat().st_size,
            "classified_at": classified_at.isoformat(),
            "delete_after": (classified_at + dt.timedelta(hours=self.args.discard_grace_hours)).isoformat(),
            "quarantine_audio": quarantine.name,
            "classifier": classifier,
        }
        if rejected_event is not None:
            record["rejected_event"] = rejected_event.name
        atomic_json(manifest, record)
        logging.info("%s，已隔离 %s 小时后再删除：%s", classification, self.args.discard_grace_hours, quarantine)
        return quarantine

    def preserve_processing_failure(self, source: Path, *, reason: str, details: dict) -> Path:
        """Keep possible speech permanently without publishing an unreliable event."""
        digest = sha256_file(source)
        relative = self.relative_destination(source)
        evidence_audio = self.args.evidence_dir / relative
        failure = evidence_audio.with_suffix(evidence_audio.suffix + ".processing_failed.json")
        evidence_audio.parent.mkdir(parents=True, exist_ok=True)
        if evidence_audio.exists() or failure.exists():
            raise RuntimeError(f"失败证据目标已存在，拒绝覆盖：{evidence_audio}")
        source.replace(evidence_audio)
        atomic_json(failure, {
            "schema_version": 1,
            "classification": "processing_failed",
            "reason": reason,
            "source_sha256": digest,
            "source_bytes": evidence_audio.stat().st_size,
            "preserved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "audio_file": evidence_audio.name,
            "details": details,
        })
        logging.error("主转写失败，原音已永久保留且不向 Agent 发布：%s（%s）", evidence_audio, reason)
        return evidence_audio

    @staticmethod
    def is_single_channel_false_positive(event: dict) -> bool:
        """Recognize the known fan-noise failure: empty primary plus <=1 active mic."""
        if str(event.get("transcript", "")).strip():
            return False
        transcripts = (
            event.get("provenance", {})
            .get("cross_check", {})
            .get("all_microphone_transcripts", [])
        )
        return (
            isinstance(transcripts, list)
            and len(transcripts) == 3
            and sum(bool(str(text).strip()) for text in transcripts) <= 1
        )

    @staticmethod
    def pending_audio_path(pending: Path) -> Path:
        suffix = ".event.pending.json"
        if not pending.name.endswith(suffix):
            raise ValueError(f"不是 pending 事件文件：{pending}")
        return pending.with_name(pending.name.removesuffix(suffix))

    def defer_pending(self, pending: Path, error: Exception) -> None:
        attempts = self.pending_retry_attempts.get(pending, 0) + 1
        delay = min(900.0, 15.0 * (2 ** min(attempts - 1, 6)))
        self.pending_retry_attempts[pending] = attempts
        self.pending_retry_after[pending] = time.monotonic() + delay
        logging.warning("事件暂未上报，%.0f 秒后重试（第 %d 次）：%s：%s", delay, attempts, pending, error)

    def finish_pending(self, pending: Path) -> None:
        self.pending_retry_attempts.pop(pending, None)
        self.pending_retry_after.pop(pending, None)

    def reject_pending(self, pending: Path, error: Exception) -> None:
        rejected = Path(str(pending).replace(".event.pending.json", ".event.rejected.json"))
        pending.replace(rejected)
        self.finish_pending(pending)
        logging.error("事件被永久拒绝，原音保留供人工复核：%s：%s", rejected, error)

    def post_event(self, event: dict) -> None:
        if not str(event.get("transcript", "")).strip():
            raise EventPostError("事件缺少有效 transcript", retryable=False)
        if not self.args.zeris_url:
            logging.warning("未配置 Zeris URL；证据保留为 pending，不会丢失")
            return
        request = urllib.request.Request(
            self.args.zeris_url,
            data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.args.zeris_token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status not in (200, 201):
                    raise EventPostError(f"Zeris 返回 HTTP {response.status}", retryable=response.status >= 500)
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                detail = ""
            retryable = error.code >= 500 or error.code in (408, 425, 429)
            message = f"Zeris 返回 HTTP {error.code}"
            if detail:
                message += f"：{detail}"
            raise EventPostError(message, retryable=retryable) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise EventPostError(f"Zeris 上报失败：{error}", retryable=True) from error

    def carry_path(self, source: Path) -> Path:
        return source.with_suffix(source.suffix + ".carry.json")

    def read_carry(self, source: Path) -> dict:
        path = self.carry_path(source)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            logging.exception("无法读取跨分段延续状态，将保守地重新处理边界：%s", path)
            return {}

    def consume_carry(self, source: Path) -> None:
        path = self.carry_path(source)
        if path.is_file():
            path.unlink()

    def process_file(self, source: Path, lookahead: Path | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="open-xiaoai-process-") as temporary:
            temporary_dir = Path(temporary)
            sources = [source] + ([lookahead] if lookahead is not None else [])
            properties = [audio_properties(item) for item in sources]
            durations_ms = [round(item["duration_seconds"] * 1000) for item in properties]
            boundary_ms = durations_ms[0]
            source_microphones = [extract_microphones(item, temporary_dir / f"source-{index}") for index, item in enumerate(sources)]
            microphones = []
            for channel in range(3):
                output = temporary_dir / f"window-mic-{channel}.wav"
                concatenate_audio([items[channel] for items in source_microphones], output)
                microphones.append(output)

            window_results = [sensevoice(self.args.sense_binary, self.args.sense_model, self.args.vad_model, audio) for audio in microphones]
            channel = choose_microphone_result(window_results)
            speech_segments = vad_segments(self.args.vad_binary, self.args.vad_model, microphones[channel])
            overlapping = [(start, end) for start, end in speech_segments if start < boundary_ms and end > 0]
            carry = self.read_carry(source)
            if not overlapping:
                if carry:
                    relative = self.relative_destination(source)
                    evidence_audio = self.args.evidence_dir / relative
                    evidence_audio.parent.mkdir(parents=True, exist_ok=True)
                    if evidence_audio.exists():
                        raise RuntimeError(f"证据文件已存在，拒绝覆盖：{evidence_audio}")
                    source.replace(evidence_audio)
                    self.consume_carry(source)
                    logging.warning("边界延续状态存在但当前 VAD 未复现，已保守保留原音：%s", evidence_audio)
                    return "continuation"
                self.quarantine_audio(
                    source,
                    classification="no_speech",
                    classifier={"model": "SenseVoiceSmall-Q8", "vad": "FSMN-VAD", "channels_checked": 3},
                )
                self.consume_carry(source)
                return "discarded"

            owned_segments = owned_vad_segments(speech_segments, boundary_ms, int(carry.get("until_ms", 0) or 0))
            if not owned_segments:
                relative = self.relative_destination(source)
                evidence_audio = self.args.evidence_dir / relative
                evidence_audio.parent.mkdir(parents=True, exist_ok=True)
                if evidence_audio.exists():
                    raise RuntimeError(f"证据文件已存在，拒绝覆盖：{evidence_audio}")
                source.replace(evidence_audio)
                self.consume_carry(source)
                logging.info("本段只有上一话语的跨界尾部，原音已保留且不重复转写：%s", evidence_audio)
                return "continuation"

            speech_microphones = []
            for mic_channel, microphone in enumerate(microphones):
                speech_output = temporary_dir / f"speech-mic-{mic_channel}.wav"
                extract_speech_audio(microphone, speech_output, owned_segments)
                speech_microphones.append(speech_output)
            sense_results = [sensevoice(self.args.sense_binary, self.args.sense_model, self.args.vad_model, audio) for audio in speech_microphones]
            texts = [text for _, text in sense_results]
            channel = choose_microphone_result(sense_results)
            try:
                primary = self.qwen_transcribe(speech_microphones[channel])
            except TranscriptionTimeout as error:
                self.preserve_processing_failure(
                    source,
                    reason="primary_asr_timeout",
                    details={
                        "model": "Qwen3-ASR-1.7B",
                        "device": "cpu",
                        "timeout_seconds": error.timeout_seconds,
                        "selected_microphone": channel,
                        "speech_segments_ms": owned_segments,
                        "sensevoice_transcripts": texts,
                        "error": str(error),
                    },
                )
                self.consume_carry(source)
                return "processing_failed"
            active_channels = sum(bool(text.strip()) for text in texts)
            if not primary and active_channels <= 1:
                self.quarantine_audio(
                    source,
                    classification="no_reliable_speech",
                    classifier={
                        "primary_model": "Qwen3-ASR-1.7B",
                        "primary_transcript_empty": True,
                        "cross_check_model": "SenseVoiceSmall-Q8",
                        "vad": "FSMN-VAD",
                        "active_channels": active_channels,
                        "channels_checked": 3,
                    },
                )
                self.consume_carry(source)
                return "discarded"

            consensus = sensevoice_consensus(texts)
            firered = None
            adjudication_triggered = consensus is not None and normalize_asr_text(primary) != normalize_asr_text(consensus)
            adjudication_attempted = adjudication_triggered and self.args.firered_enabled
            if adjudication_triggered:
                firered = self.firered_transcribe(speech_microphones[channel])
            selected_transcript, adjudication_result = apply_asr_adjudication(
                primary,
                texts,
                firered.get("text") if firered else None,
            )

            speakers, diarization_status = self.diarize(microphones[channel])
            event_start_ms = min(start for start, _ in owned_segments)
            event_end_ms = max(end for _, end in owned_segments)
            filtered_speakers = []
            for speaker in speakers:
                start_ms = round(speaker["start_seconds"] * 1000)
                end_ms = round(speaker["end_seconds"] * 1000)
                if any(start_ms < segment_end and end_ms > segment_start for segment_start, segment_end in owned_segments):
                    filtered_speakers.append({
                        **speaker,
                        "start_seconds": round(max(0, start_ms - event_start_ms) / 1000, 3),
                        "end_seconds": round(max(0, end_ms - event_start_ms) / 1000, 3),
                    })
            started = parse_capture_time(source)
            occurred = started + dt.timedelta(milliseconds=event_start_ms)
            ended = started + dt.timedelta(milliseconds=event_end_ms)
            digest = sha256_file(source)
            relative = self.relative_destination(source)
            evidence_audio = self.args.evidence_dir / relative
            evidence_audio.parent.mkdir(parents=True, exist_ok=True)
            if evidence_audio.exists():
                raise RuntimeError(f"证据文件已存在，拒绝覆盖：{evidence_audio}")
            source.replace(evidence_audio)

            referenced_sources = sources if event_end_ms > boundary_ms and lookahead is not None else [source]
            referenced_durations = durations_ms[:len(referenced_sources)]
            refs = audio_refs_for_segments(
                [evidence_audio] + ([lookahead] if len(referenced_sources) > 1 else []),
                referenced_durations,
                owned_segments,
                self.args.nas_uri_root,
            )
            identity = hashlib.sha256(json.dumps({
                "sources": [item["sha256"] for item in refs],
                "segments": owned_segments,
            }, sort_keys=True).encode("utf-8")).hexdigest()
            playback_matches = playback_matches_for_window(
                occurred,
                ended,
                self.args.playback_dir,
                self.args.playback_nas_uri_root,
            )
            playback_refs = [{
                key: item[key]
                for key in ("uri", "sha256", "offset_start_seconds", "offset_end_seconds")
            } for item in playback_matches]
            speech_segments_seconds = [
                (max(0, start - event_start_ms) / 1000, max(0, end - event_start_ms) / 1000)
                for start, end in owned_segments
            ]
            acoustic_scene = acoustic_scene_with_playback(
                identity,
                (event_end_ms - event_start_ms) / 1000,
                speech_segments_seconds,
                playback_matches,
            )

            reliability = asr_reliability(primary, texts)
            if adjudication_result == "sensevoice_consensus_confirmed":
                reliability["notes"] += " FireRed 独立复核支持三麦 SenseVoice 共识，已将其作为展示文本；原始模型冲突仍需保留并可回听。"
            alternatives = [
                {"model": "SenseVoiceSmall", "version": "Q8-GGUF", "text": text}
                for text in texts if text
            ]
            if normalize_asr_text(selected_transcript) != normalize_asr_text(primary):
                alternatives.insert(0, {"model": "Qwen3-ASR", "version": "1.7B-CPU-FP32", "text": primary})
            elif firered and firered.get("text"):
                alternatives.append({
                    "model": "FireRedASR2-AED",
                    "version": "FP32-CPU",
                    "text": firered["text"],
                })
            event = {
                "event_id": f"audio-{identity[:24]}",
                "occurred_at": occurred.isoformat(),
                "ended_at": ended.isoformat(),
                "source": "xiaoai-oh2p-3mic",
                "audio_ref": refs[0]["uri"],
                "audio_sha256": digest,
                "audio_refs": refs,
                "playback_refs": playback_refs,
                "transcript": selected_transcript,
                "language": "zh",
                "alternatives": alternatives,
                "speakers": filtered_speakers,
                "acoustic_scene": acoustic_scene,
                "reliability": reliability,
                "provenance": {
                    "schema_version": 1,
                    "evidence_kind": "fallible_asr",
                    "primary_asr": {
                        "model": "Qwen3-ASR-1.7B",
                        "runtime": "qwen-asr",
                        "device": "cpu",
                        "selected_microphone": channel,
                        "input": "fsmn_vad_speech_regions_concatenated",
                        "speech_segments_ms": owned_segments,
                    },
                    "cross_check": {"model": "SenseVoiceSmall-Q8", "vad": "FSMN-VAD", "all_microphone_transcripts": texts},
                    "adjudication": {
                        "enabled": self.args.firered_enabled,
                        "trigger": "three_microphone_sensevoice_consensus_conflicts_with_qwen",
                        "triggered": adjudication_triggered,
                        "attempted": adjudication_attempted,
                        "completed": firered is not None,
                        "model": "FireRedASR2-AED",
                        "device": "cpu",
                        "result": adjudication_result,
                        "text": firered.get("text") if firered else None,
                        "confidence": firered.get("confidence") if firered else None,
                        "timestamps": firered.get("timestamps", []) if firered else [],
                        "load_seconds": firered.get("load_seconds") if firered else None,
                        "inference_seconds": firered.get("inference_seconds") if firered else None,
                    },
                    "audio": {
                        "storage_blocks": properties,
                        "fixed_block_seconds": properties[0]["duration_seconds"],
                        "lookahead_used": lookahead is not None,
                        "boundary_ownership": "utterance_start",
                        "carried_from_previous_until_ms": int(carry.get("until_ms", 0) or 0),
                        "playback_reference_archived": bool(playback_refs),
                        "playback_reference_count": len(playback_refs),
                    },
                    "speaker_diarization": {
                        "status": diarization_status,
                        "engine": "sherpa-onnx/pyannote-segmentation-3.0/3D-Speaker-zh",
                        "cluster_scope": "recording_only",
                        "identity_claims_allowed": False,
                    },
                },
            }
            pending = evidence_audio.with_suffix(evidence_audio.suffix + ".event.pending.json")
            atomic_json(pending, event)
            continuation_until = max((end - boundary_ms for _, end in owned_segments if end > boundary_ms), default=0)
            if lookahead is not None and continuation_until > 0:
                atomic_json(self.carry_path(lookahead), {
                    "schema_version": 1,
                    "owner_event_id": event["event_id"],
                    "until_ms": continuation_until,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                })
            self.consume_carry(source)
            try:
                self.post_event(event)
            except EventPostError as error:
                if error.retryable:
                    self.defer_pending(pending, error)
                    return "pending"
                self.reject_pending(pending, error)
                return "rejected"
            if self.args.zeris_url:
                pending.replace(evidence_audio.with_suffix(evidence_audio.suffix + ".event.json"))
                self.finish_pending(pending)
            logging.info("已保留跨边界语音证据：%s（%d 个 VAD 区间，模型一致度 %.3f）", evidence_audio, len(owned_segments), score)
            return "evidence"

    def retry_pending_events(self) -> int:
        if not self.args.zeris_url:
            return 0
        sent = 0
        for pending in sorted(self.args.evidence_dir.rglob("*.event.pending.json")):
            if self.pending_retry_after.get(pending, 0.0) > time.monotonic():
                continue
            try:
                event = json.loads(pending.read_text(encoding="utf-8"))
                if self.is_single_channel_false_positive(event):
                    audio = self.pending_audio_path(pending)
                    if audio.is_file():
                        self.quarantine_audio(
                            audio,
                            classification="no_reliable_speech",
                            classifier={
                                "primary_model": "Qwen3-ASR-1.7B",
                                "primary_transcript_empty": True,
                                "cross_check_model": "SenseVoiceSmall-Q8",
                                "reason": "recovered_single_channel_false_positive",
                            },
                            pending_event=pending,
                        )
                        self.finish_pending(pending)
                        continue
                self.post_event(event)
                pending.replace(Path(str(pending).replace(".event.pending.json", ".event.json")))
                self.finish_pending(pending)
                sent += 1
            except EventPostError as error:
                if error.retryable:
                    self.defer_pending(pending, error)
                else:
                    self.reject_pending(pending, error)
            except Exception as error:
                self.defer_pending(pending, error)
        return sent

    def purge_expired_quarantine(self) -> int:
        removed = 0
        now = dt.datetime.now(dt.timezone.utc)
        for manifest in self.args.discard_dir.rglob("*.json"):
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                deadline = dt.datetime.fromisoformat(record["delete_after"])
                name = record["quarantine_audio"]
                if deadline > now or Path(name).name != name:
                    continue
                audio = manifest.parent / name
                if audio.is_file() and sha256_file(audio) == record.get("source_sha256"):
                    audio.unlink()
                    record["deleted_at"] = now.isoformat()
                    atomic_json(manifest, record)
                    removed += 1
            except (KeyError, ValueError, json.JSONDecodeError, OSError):
                logging.exception("无法安全清理隔离文件：%s", manifest)
        return removed

    def run(self) -> int:
        self.require_runtime()
        processed = 0
        while True:
            try:
                removed = self.purge_expired_quarantine()
                if removed:
                    logging.info("已删除 %d 个过期的无语音隔离段", removed)
                self.retry_pending_events()
                sources = self.pending_files()
                for index, source in enumerate(sources):
                    lookahead = sources[index + 1] if index + 1 < len(sources) and self.is_contiguous(source, sources[index + 1]) else None
                    if lookahead is None and time.time() - source.stat().st_mtime < self.args.lookahead_wait_seconds:
                        continue
                    try:
                        self.process_file(source, lookahead)
                        processed += 1
                    except Exception:
                        logging.exception("单个录音处理失败，已保留并继续后续文件：%s", source)
                    if self.args.max_files and processed >= self.args.max_files:
                        return 0
            except Exception:
                logging.exception("处理循环失败；原音或 pending 事件已保留，稍后重试")
            if self.args.once:
                return 0
            time.sleep(max(0.5, self.args.poll_seconds))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        return EvidenceProcessor(args).run()
    except RuntimeError as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
