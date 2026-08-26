#!/usr/bin/env python3
"""CPU-only anonymous speaker diarization helper for processor.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def subtract_intervals(start: float, end: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return parts of [start, end] that do not overlap another speaker."""
    available = [(start, end)]
    for left, right in sorted(blocked):
        next_available = []
        for current_start, current_end in available:
            if right <= current_start or left >= current_end:
                next_available.append((current_start, current_end))
                continue
            if left > current_start:
                next_available.append((current_start, min(left, current_end)))
            if right < current_end:
                next_available.append((max(right, current_start), current_end))
        available = next_available
    return [(left, right) for left, right in available if right - left >= 0.1]


def create_embedding_extractor(model: Path):
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model), num_threads=2, provider="cpu",
    )
    if not config.validate():
        raise RuntimeError("invalid sherpa-onnx speaker embedding configuration")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def speaker_embedding(samples: np.ndarray, sample_rate: int, extractor) -> list[float]:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return []
    return [round(float(value), 7) for value in extractor.compute(stream)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--embedding-intervals-json", default="[]")
    args = parser.parse_args()
    requested_intervals = json.loads(args.embedding_intervals_json)
    allowed_intervals = [
        (float(item[0]), float(item[1]))
        for item in requested_intervals
        if isinstance(item, list) and len(item) == 2 and float(item[1]) > float(item[0])
    ]

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(args.segmentation), window_shift_ratio=0.1,
            ),
            num_threads=2,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(args.embedding), num_threads=2, provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=args.threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("invalid sherpa-onnx diarization configuration")
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if sample_rate != engine.sample_rate:
        raise RuntimeError(f"expected {engine.sample_rate} Hz, got {sample_rate} Hz")
    mono = samples[:, 0]
    result = engine.process(mono).sort_by_start_time()
    segments = [
        {"speaker": int(item.speaker), "start_seconds": round(float(item.start), 3), "end_seconds": round(float(item.end), 3)}
        for item in result
    ]
    embeddings = []
    embedding_extractor = create_embedding_extractor(args.embedding)
    for speaker in sorted({item["speaker"] for item in segments}):
        own = [item for item in segments if item["speaker"] == speaker]
        other = [(item["start_seconds"], item["end_seconds"]) for item in segments if item["speaker"] != speaker]
        clean_intervals = []
        for item in own:
            clean_intervals.extend(subtract_intervals(item["start_seconds"], item["end_seconds"], other))
        if allowed_intervals:
            clean_intervals = [
                (max(start, allowed_start), min(end, allowed_end))
                for start, end in clean_intervals
                for allowed_start, allowed_end in allowed_intervals
                if min(end, allowed_end) - max(start, allowed_start) >= 0.1
            ]
        chunks = [mono[max(0, round(start * sample_rate)):min(len(mono), round(end * sample_rate))] for start, end in clean_intervals]
        clean = np.concatenate([item for item in chunks if item.size]) if any(item.size for item in chunks) else np.empty((0,), dtype=np.float32)
        if clean.size > sample_rate * 30:
            clean = clean[:sample_rate * 30]
        rms = math.sqrt(float(np.mean(np.square(clean, dtype=np.float64)))) if clean.size else 0.0
        embeddings.append({
            "speaker": speaker,
            "embedding": speaker_embedding(clean, sample_rate, embedding_extractor) if clean.size >= sample_rate else [],
            "quality": {
                "speech_seconds": round(clean.size / sample_rate, 3),
                "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2),
                "clipping_fraction": round(float(np.mean(np.abs(clean) >= 0.999)), 6) if clean.size else 0.0,
                "overlap_excluded": True,
            },
        })
    print(json.dumps({"segments": segments, "speaker_embeddings": embeddings}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
