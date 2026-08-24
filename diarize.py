#!/usr/bin/env python3
"""CPU-only anonymous speaker diarization helper for processor.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

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
    result = engine.process(samples[:, 0]).sort_by_start_time()
    print(json.dumps([
        {"speaker": int(item.speaker), "start_seconds": round(float(item.start), 3), "end_seconds": round(float(item.end), 3)}
        for item in result
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
