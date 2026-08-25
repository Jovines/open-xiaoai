#!/usr/bin/env python3
"""Benchmark Chinese ASR engines on auditable, labeled audio cases."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import tempfile
import time
import unicodedata
import wave


TAG = re.compile(r"<\|[^|]*\|>")


def normalize_zh(text: str) -> str:
    """Canonical CER text: strip tags/punctuation and normalize case/width."""
    value = unicodedata.normalize("NFKC", TAG.sub("", text)).upper()
    return "".join(character for character in value if character.isalnum())


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Return Levenshtein substitutions, deletions and insertions."""
    left, right = normalize_zh(reference), normalize_zh(hypothesis)
    rows = [[(0, 0, 0, 0)] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(1, len(left) + 1):
        rows[i][0] = (i, 0, i, 0)
    for j in range(1, len(right) + 1):
        rows[0][j] = (j, 0, 0, j)
    for i, expected in enumerate(left, 1):
        for j, actual in enumerate(right, 1):
            if expected == actual:
                rows[i][j] = rows[i - 1][j - 1]
                continue
            substitution = rows[i - 1][j - 1]
            deletion = rows[i - 1][j]
            insertion = rows[i][j - 1]
            choices = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            rows[i][j] = min(choices, key=lambda item: item[0])
    edits, substitutions, deletions, insertions = rows[-1][-1]
    return {
        "reference_characters": len(left),
        "edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def load_manifest(path: Path) -> list[dict]:
    cases, identities = [], set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        case = json.loads(raw)
        for field in ("id", "audio", "reference"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"{path}:{line_number} 缺少 {field}")
        if case["id"] in identities:
            raise ValueError(f"{path}:{line_number} id 重复：{case['id']}")
        identities.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"评测清单为空：{path}")
    return cases


def resolve_audio(case: dict, audio_root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(case["audio"])))
    path = expanded if expanded.is_absolute() else audio_root / expanded
    if not path.is_file():
        raise FileNotFoundError(f"评测音频不存在：{path}")
    expected = case.get("sha256")
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"评测音频哈希不符：{path}")
    return path


def prepare_audio(case: dict, source: Path, destination: Path) -> None:
    command = ["ffmpeg", "-v", "error", "-y"]
    if "start_seconds" in case:
        command.extend(["-ss", str(float(case["start_seconds"]))])
    command.extend(["-i", str(source)])
    if "duration_seconds" in case:
        command.extend(["-t", str(float(case["duration_seconds"]))])
    if "channel" in case:
        command.extend(["-map_channel", f"0.0.{int(case['channel'])}"])
    command.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(destination)])
    subprocess.run(command, check=True)


def wav_quality(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        width = source.getsampwidth()
        frames = source.getnframes()
        if channels != 1 or width != 2:
            raise ValueError(f"评测中间文件必须是 mono PCM16：{path}")
        raw = source.readframes(frames)
    import array
    samples = array.array("h", raw)
    if samples.itemsize != 2:
        raise RuntimeError("当前平台的 signed short 不是 16 bit")
    peak = max((abs(value) for value in samples), default=0) / 32768
    clipped = sum(abs(value) >= 32767 for value in samples)
    near_clipping = sum(abs(value) >= round(0.95 * 32767) for value in samples)
    square_sum = sum(float(value) * value for value in samples)
    count = max(1, len(samples))
    return {
        "duration_seconds": round(frames / rate, 6),
        "peak": round(peak, 6),
        "rms": round((square_sum / count) ** 0.5 / 32768, 6),
        "clipped_fraction": round(clipped / count, 8),
        "near_clipping_fraction": round(near_clipping / count, 8),
    }


class SenseVoiceEngine:
    name = "sensevoice-small-q8"

    def __init__(self, binary: Path, model: Path, vad: Path) -> None:
        self.binary, self.model, self.vad = binary, model, vad
        for path in (binary, model, vad):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.load_seconds = 0.0

    def transcribe(self, audio: Path) -> str:
        result = subprocess.run(
            [str(self.binary), "-m", str(self.model), "--vad", str(self.vad), "-a", str(audio)],
            check=True, capture_output=True, text=True,
        )
        lines = [
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("[sensevoice]")
        ]
        return " ".join(lines)


class QwenEngine:
    name = "qwen3-asr-1.7b-cpu-fp32"

    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
        started = time.perf_counter()
        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            device_map="cpu",
            dtype="float32",
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
        )
        self.load_seconds = time.perf_counter() - started

    def transcribe(self, audio: Path) -> str:
        return self.model.transcribe(str(audio), language="Chinese")[0].text.strip()


def score_result(case: dict, hypothesis: str, seconds: float, runtime: float, repeat: int) -> dict:
    counts = edit_counts(case["reference"], hypothesis)
    critical = [normalize_zh(value) for value in case.get("critical_terms", [])]
    normalized_hypothesis = normalize_zh(hypothesis)
    return {
        "case_id": case["id"],
        "repeat": repeat,
        "reference": case["reference"],
        "hypothesis": hypothesis,
        "normalized_reference": normalize_zh(case["reference"]),
        "normalized_hypothesis": normalized_hypothesis,
        **counts,
        "cer": round(counts["edits"] / max(1, counts["reference_characters"]), 6),
        "critical_terms": case.get("critical_terms", []),
        "critical_terms_matched": sum(term in normalized_hypothesis for term in critical),
        "critical_terms_total": len(critical),
        "runtime_seconds": round(runtime, 6),
        "audio_seconds": seconds,
        "rtf": round(runtime / max(0.001, seconds), 6),
        "tags": case.get("tags", []),
    }


def aggregate(results: list[dict]) -> dict:
    references = sum(item["reference_characters"] for item in results)
    runtime = sum(item["runtime_seconds"] for item in results)
    audio = sum(item["audio_seconds"] for item in results)
    by_case: dict[str, set[str]] = {}
    for item in results:
        by_case.setdefault(item["case_id"], set()).add(item["normalized_hypothesis"])
    return {
        "runs": len(results),
        "cases": len(by_case),
        "micro_cer": round(sum(item["edits"] for item in results) / max(1, references), 6),
        "substitutions": sum(item["substitutions"] for item in results),
        "deletions": sum(item["deletions"] for item in results),
        "insertions": sum(item["insertions"] for item in results),
        "critical_term_accuracy": round(
            sum(item["critical_terms_matched"] for item in results)
            / max(1, sum(item["critical_terms_total"] for item in results)), 6
        ),
        "unstable_cases": sum(len(values) > 1 for values in by_case.values()),
        "rtf": round(runtime / max(0.001, audio), 6),
    }


def aggregate_by_tag(results: list[dict]) -> dict[str, dict]:
    tags = sorted({tag for item in results for tag in item.get("tags", [])})
    return {tag: aggregate([item for item in results if tag in item.get("tags", [])]) for tag in tags}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可重复的中文 ASR micro-CER/关键词/性能评测")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--engine", action="append", choices=("qwen", "sensevoice"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--qwen-max-new-tokens", type=int, default=384)
    parser.add_argument("--sense-binary", type=Path, default=Path(os.environ.get("SENSEVOICE_BINARY", str(Path.home() / ".local/opt/sensevoice/llama-funasr-sensevoice"))))
    parser.add_argument("--sense-model", type=Path, default=Path(os.environ.get("SENSEVOICE_MODEL", str(Path.home() / "models/sensevoice-small/sensevoice-small-q8.gguf"))))
    parser.add_argument("--sense-vad", type=Path, default=Path(os.environ.get("SENSEVOICE_VAD_MODEL", str(Path.home() / "models/sensevoice-small/fsmn-vad.gguf"))))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    cases = load_manifest(args.manifest)
    engines = []
    for name in dict.fromkeys(args.engine):
        if name == "sensevoice":
            engines.append(SenseVoiceEngine(args.sense_binary, args.sense_model, args.sense_vad))
        else:
            engines.append(QwenEngine(args.qwen_model, args.qwen_max_new_tokens))
    all_results, summaries, qualities = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="open-xiaoai-asr-eval-") as directory:
        prepared = {}
        for case in cases:
            destination = Path(directory) / f"{case['id']}.wav"
            prepare_audio(case, resolve_audio(case, args.audio_root), destination)
            prepared[case["id"]] = destination
            qualities[case["id"]] = wav_quality(destination)
        for engine in engines:
            results = []
            for repeat in range(1, args.repeat + 1):
                for case in cases:
                    audio = prepared[case["id"]]
                    started = time.perf_counter()
                    hypothesis = engine.transcribe(audio)
                    runtime = time.perf_counter() - started
                    results.append(score_result(case, hypothesis, qualities[case["id"]]["duration_seconds"], runtime, repeat))
            all_results[engine.name] = results
            summaries[engine.name] = {
                **aggregate(results),
                "model_load_seconds": round(engine.load_seconds, 6),
                "by_tag": aggregate_by_tag(results),
            }
    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "platform": {"node": platform.node(), "machine": platform.machine(), "python": platform.python_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "repeat": args.repeat,
        "audio_quality": qualities,
        "summary": summaries,
        "results": all_results,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("engine\tmicro_CER\tcritical_accuracy\tunstable_cases\tRTF\tload_s")
    for name, summary in summaries.items():
        print(f"{name}\t{summary['micro_cer']:.4f}\t{summary['critical_term_accuracy']:.4f}\t{summary['unstable_cases']}\t{summary['rtf']:.4f}\t{summary['model_load_seconds']:.2f}")
    print(f"report\t{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
