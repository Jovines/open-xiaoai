#!/usr/bin/env python3
"""Evaluate conservative household acoustic-scene predictions.

The manifest and audio stay outside Git.  Each JSONL row requires id, audio and
label; optional day, speaker and device fields are reported so train/test
leakage can be spotted before a household-specific classifier is trusted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

from scene_analysis import SCENE_LABELS, infer_scene


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(str(row.get(key) or "").strip() for key in ("id", "audio", "label")):
            raise ValueError(f"manifest line {number} requires id, audio and label")
        if row["label"] not in SCENE_LABELS:
            raise ValueError(f"manifest line {number} has unsupported label {row['label']}")
        rows.append(row)
    ids = [str(item["id"]) for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest ids must be unique")
    return rows


def audio_path(row: dict, root: Path) -> Path:
    path = Path(str(row["audio"])).expanduser()
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(row.get("sha256") or "")
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"audio hash mismatch for {row['id']}")
    return path


def tag_audio(path: Path, args: argparse.Namespace) -> dict:
    result = subprocess.run([
        str(args.python), str(args.script), str(path),
        "--model", str(args.model), "--labels", str(args.labels), "--top-k", str(args.top_k),
    ], check=True, capture_output=True, text=True, timeout=args.timeout_seconds)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def metrics(rows: list[dict]) -> dict:
    labels = sorted({item["label"] for item in rows})
    per_class = {}
    f1_values = []
    for label in labels:
        tp = sum(item["label"] == label and item["prediction"] == label for item in rows)
        fp = sum(item["label"] != label and item["prediction"] == label for item in rows)
        fn = sum(item["label"] == label and item["prediction"] != label for item in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {"support": tp + fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    bins = defaultdict(list)
    for item in rows:
        bins[min(9, int(float(item["confidence"]) * 10))].append(item)
    ece = 0.0
    for values in bins.values():
        accuracy = sum(item["prediction"] == item["label"] for item in values) / len(values)
        confidence = sum(float(item["confidence"]) for item in values) / len(values)
        ece += len(values) / max(1, len(rows)) * abs(accuracy - confidence)
    unknown = [item for item in rows if item["label"] == "unknown"]
    known = [item for item in rows if item["label"] != "unknown"]
    return {
        "samples": len(rows),
        "accuracy": round(sum(item["prediction"] == item["label"] for item in rows) / max(1, len(rows)), 4),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "expected_calibration_error_10_bins": round(ece, 4),
        "unknown_recall": round(sum(item["prediction"] == "unknown" for item in unknown) / max(1, len(unknown)), 4),
        "known_rejected_as_unknown": round(sum(item["prediction"] == "unknown" for item in known) / max(1, len(known)), 4),
        "per_class": per_class,
        "confusion": dict(Counter(f"{item['label']} -> {item['prediction']}" for item in rows)),
    }


def sliced_metrics(rows: list[dict], field: str) -> dict:
    grouped = defaultdict(list)
    for item in rows:
        grouped[str(item.get(field) or "unspecified")].append(item)
    return {name: metrics(values) for name, values in sorted(grouped.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path.home() / ".venvs/sherpa-onnx/bin/python")
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().parents[1] / "audio_tag.py")
    parser.add_argument("--model", type=Path, default=Path.home() / "models/audio-tagging/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx")
    parser.add_argument("--labels", type=Path, default=Path.home() / "models/audio-tagging/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    results = []
    started = time.monotonic()
    for row in rows:
        path = audio_path(row, args.audio_root)
        tagging = tag_audio(path, args)
        scene = infer_scene({"scene_type": "unknown", "playback_reference": {"available": False}}, tagging, str(row.get("transcript") or ""))
        results.append({
            "id": row["id"], "label": row["label"], "prediction": scene["primary"], "confidence": scene["confidence"],
            "day": row.get("day"), "speaker": row.get("speaker"), "device": row.get("device"),
            "candidates": scene["candidates"], "audio_tagging": tagging,
        })
    report = {
        "schema_version": 1,
        "model": "sherpa-onnx/CED-mini-int8 + rule-fusion-v1",
        "metrics": metrics(results),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "dataset_slices": {field: sliced_metrics(results, field) for field in ("day", "speaker", "device")},
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.part")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
