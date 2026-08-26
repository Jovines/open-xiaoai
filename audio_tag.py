#!/usr/bin/env python3
"""Run lightweight CED AudioSet tagging in an isolated CPU process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import sherpa_onnx
import soundfile as sf


def merge_window_results(results, top_k: int) -> list[dict]:
    """Keep the strongest observation per AudioSet label across fixed windows."""
    strongest: dict[int, tuple[str, float]] = {}
    for result in results:
        for item in result:
            probability = float(item.prob)
            if item.index not in strongest or probability > strongest[item.index][1]:
                strongest[int(item.index)] = (str(item.name), probability)
    return [
        {"label": label, "index": index, "probability": round(probability, 6)}
        for index, (label, probability) in sorted(
            strongest.items(), key=lambda item: item[1][1], reverse=True,
        )[:top_k]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            ced=str(args.model), num_threads=2, debug=False, provider="cpu",
        ),
        labels=str(args.labels),
        top_k=max(1, min(20, args.top_k)),
    )
    if not config.validate():
        raise RuntimeError("invalid sherpa-onnx CED audio tagging configuration")
    tagger = sherpa_onnx.AudioTagging(config)
    samples, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(samples[:, 0])
    # CED uses AudioSet-style fixed context. Feeding a long household event in
    # one stream can exceed its positional tensor, so evaluate bounded windows.
    window_samples = max(1, round(sample_rate * 10))
    windows = [mono[start:start + window_samples] for start in range(0, len(mono), window_samples)]
    started = time.monotonic()
    results = []
    for window in windows:
        stream = tagger.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=window)
        results.append(tagger.compute(stream))
    print(json.dumps({
        "model": "sherpa-onnx/CED-mini-int8",
        "inference_seconds": round(time.monotonic() - started, 4),
        "audio_seconds": round(len(mono) / sample_rate, 4),
        "windows_analyzed": len(windows),
        "window_seconds": 10,
        "aggregation": "maximum_probability_per_label",
        "tags": merge_window_results(results, max(1, min(20, args.top_k))),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
