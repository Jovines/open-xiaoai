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
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


FILENAME_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[+-]\d{4})")
SUPPORTED_AUDIO = {".flac", ".wav", ".ogg"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛掉无语音录音，并生成可回听、可审计的中文转写证据")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("PROCESSOR_INPUT_DIR", "/mnt/dx4600/家庭管家/录音/inbox")))
    parser.add_argument("--evidence-dir", type=Path, default=Path(os.environ.get("PROCESSOR_EVIDENCE_DIR", "/mnt/dx4600/家庭管家/录音/evidence")))
    parser.add_argument("--discard-dir", type=Path, default=Path(os.environ.get("PROCESSOR_DISCARD_DIR", "/mnt/dx4600/家庭管家/录音/discarded")))
    parser.add_argument("--required-mount", type=Path, default=Path(os.environ.get("PROCESSOR_REQUIRED_MOUNT", "/mnt/dx4600")))
    parser.add_argument("--nas-uri-root", default=os.environ.get("PROCESSOR_NAS_URI_ROOT", "nas://dx4600/家庭管家/录音/evidence"))
    parser.add_argument("--sense-binary", type=Path, default=Path(os.environ.get("SENSEVOICE_BINARY", str(Path.home() / ".local/opt/sensevoice/llama-funasr-sensevoice"))))
    parser.add_argument("--sense-model", type=Path, default=Path(os.environ.get("SENSEVOICE_MODEL", str(Path.home() / "models/sensevoice-small/sensevoice-small-q8.gguf"))))
    parser.add_argument("--vad-model", type=Path, default=Path(os.environ.get("SENSEVOICE_VAD_MODEL", str(Path.home() / "models/sensevoice-small/fsmn-vad.gguf"))))
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--diarization-python", type=Path, default=Path(os.environ.get("DIARIZATION_PYTHON", str(Path.home() / ".venvs/sherpa-onnx/bin/python"))))
    parser.add_argument("--diarization-script", type=Path, default=Path(os.environ.get("DIARIZATION_SCRIPT", str(Path(__file__).with_name("diarize.py")))))
    parser.add_argument("--diarization-segmentation", type=Path, default=Path(os.environ.get("DIARIZATION_SEGMENTATION_MODEL", str(Path.home() / "models/speaker-diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"))))
    parser.add_argument("--diarization-embedding", type=Path, default=Path(os.environ.get("DIARIZATION_EMBEDDING_MODEL", str(Path.home() / "models/speaker-diarization/3dspeaker-zh.onnx"))))
    parser.add_argument("--zeris-url", default=os.environ.get("ZERIS_AUDIO_INGEST_URL", ""))
    parser.add_argument("--zeris-token", default=os.environ.get("ZERIS_AUDIO_INGEST_TOKEN", ""))
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("PROCESSOR_POLL_SECONDS", "3")))
    parser.add_argument("--settle-seconds", type=float, default=float(os.environ.get("PROCESSOR_SETTLE_SECONDS", "5")))
    parser.add_argument("--discard-grace-hours", type=float, default=float(os.environ.get("PROCESSOR_DISCARD_GRACE_HOURS", "72")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=os.environ.get("PROCESSOR_LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def normalized_agreement(left: str, right: str) -> float:
    clean = lambda value: re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()
    a, b = clean(left), clean(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def choose_microphone(transcripts: list[str]) -> int:
    """Choose the transcript most similar to the other active microphones."""
    if len(transcripts) != 3:
        raise ValueError("expected exactly three microphone transcripts")
    scores = [
        sum(normalized_agreement(text, other) for j, other in enumerate(transcripts) if i != j)
        for i, text in enumerate(transcripts)
    ]
    return max(range(3), key=lambda index: scores[index])


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
    outputs = []
    for channel in range(3):
        output = directory / f"mic-{channel}.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map_channel", f"0.0.{channel}", "-ar", "16000", "-ac", "1", str(output)],
            check=True,
        )
        outputs.append(output)
    return outputs


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


class EvidenceProcessor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.qwen = None

    def require_runtime(self) -> None:
        if not self.args.required_mount.is_mount():
            raise RuntimeError(f"所需存储未挂载：{self.args.required_mount}")
        for program in ("ffmpeg", "ffprobe"):
            if not shutil.which(program):
                raise RuntimeError(f"缺少依赖程序：{program}")
        for path in (self.args.sense_binary, self.args.sense_model, self.args.vad_model):
            if not path.is_file():
                raise RuntimeError(f"缺少模型或程序：{path}")
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
            max_new_tokens=512,
        )

    def qwen_transcribe(self, audio: Path) -> str:
        self.load_qwen()
        result = self.qwen.transcribe(str(audio), language="Chinese")
        return result[0].text.strip()

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

    def relative_destination(self, source: Path) -> Path:
        captured = parse_capture_time(source)
        return Path(captured.strftime("%Y/%m/%d")) / source.name

    def post_event(self, event: dict) -> None:
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
                    raise RuntimeError(f"Zeris 返回 HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Zeris 上报失败：{error}") from error

    def process_file(self, source: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="open-xiaoai-process-") as temporary:
            microphones = extract_microphones(source, Path(temporary))
            sense_results = [sensevoice(self.args.sense_binary, self.args.sense_model, self.args.vad_model, audio) for audio in microphones]
            if not any(segments > 0 and text for segments, text in sense_results):
                digest = sha256_file(source)
                relative = self.relative_destination(source)
                quarantine = self.args.discard_dir / relative.with_suffix(relative.suffix + ".quarantine")
                discarded = self.args.discard_dir / relative.with_suffix(".json")
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                source.replace(quarantine)
                classified_at = dt.datetime.now(dt.timezone.utc)
                atomic_json(discarded, {
                    "schema_version": 1,
                    "classification": "no_speech",
                    "source_sha256": digest,
                    "source_bytes": quarantine.stat().st_size,
                    "classified_at": classified_at.isoformat(),
                    "delete_after": (classified_at + dt.timedelta(hours=self.args.discard_grace_hours)).isoformat(),
                    "quarantine_audio": quarantine.name,
                    "classifier": {"model": "SenseVoiceSmall-Q8", "vad": "FSMN-VAD", "channels_checked": 3},
                })
                logging.info("无语音，已隔离 %s 小时后再删除：%s", self.args.discard_grace_hours, quarantine)
                return "discarded"

            texts = [text for _, text in sense_results]
            channel = choose_microphone(texts)
            primary = self.qwen_transcribe(microphones[channel])
            speakers, diarization_status = self.diarize(microphones[channel])
            audio = audio_properties(source)
            duration = audio["duration_seconds"]
            started = parse_capture_time(source)
            ended = started + dt.timedelta(seconds=duration)
            digest = sha256_file(source)
            relative = self.relative_destination(source)
            evidence_audio = self.args.evidence_dir / relative
            evidence_audio.parent.mkdir(parents=True, exist_ok=True)
            if evidence_audio.exists():
                raise RuntimeError(f"证据文件已存在，拒绝覆盖：{evidence_audio}")
            source.replace(evidence_audio)

            comparisons = [normalized_agreement(primary, text) for text in texts if text]
            score = sum(comparisons) / len(comparisons) if comparisons else 0.0
            agreement = "high" if score >= 0.88 else "medium" if score >= 0.65 else "low"
            event = {
                "event_id": f"audio-{digest[:24]}",
                "occurred_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "source": "xiaoai-oh2p-3mic",
                "audio_ref": f"{self.args.nas_uri_root.rstrip('/')}/{relative.as_posix()}",
                "audio_sha256": digest,
                "transcript": primary,
                "language": "zh",
                "alternatives": [
                    {"model": "SenseVoiceSmall", "version": "Q8-GGUF", "text": text}
                    for text in texts if text
                ],
                "speakers": speakers,
                "reliability": {
                    "agreement": agreement,
                    "score": round(score, 4),
                    "needs_review": agreement == "low",
                    "notes": "机器转写可能听错；高影响事项必须回听原音或向家庭成员确认。",
                },
                "provenance": {
                    "schema_version": 1,
                    "evidence_kind": "fallible_asr",
                    "primary_asr": {"model": "Qwen3-ASR-1.7B", "runtime": "qwen-asr", "device": "cpu", "selected_microphone": channel},
                    "cross_check": {"model": "SenseVoiceSmall-Q8", "vad": "FSMN-VAD", "all_microphone_transcripts": texts},
                    "audio": audio,
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
            self.post_event(event)
            if self.args.zeris_url:
                pending.replace(evidence_audio.with_suffix(evidence_audio.suffix + ".event.json"))
            logging.info("已保留语音证据：%s（模型一致度 %.3f）", evidence_audio, score)
            return "evidence"

    def retry_pending_events(self) -> int:
        if not self.args.zeris_url:
            return 0
        sent = 0
        for pending in sorted(self.args.evidence_dir.rglob("*.event.pending.json")):
            event = json.loads(pending.read_text(encoding="utf-8"))
            self.post_event(event)
            pending.replace(Path(str(pending).replace(".event.pending.json", ".event.json")))
            sent += 1
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
                for source in self.pending_files():
                    self.process_file(source)
                    processed += 1
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
