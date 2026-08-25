#!/usr/bin/env python3
"""Run FireRedASR2-AED as an isolated, CPU-only adjudicator and emit one JSON result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--deps-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.deps_dir))
    sys.path.insert(0, str(args.source_dir))

    import torch
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    loaded_at = time.perf_counter()
    model = FireRedAsr2.from_pretrained(
        "aed",
        str(args.model_dir),
        FireRedAsr2Config(use_gpu=False, use_half=False, beam_size=3, nbest=1, return_timestamp=True),
    )
    load_seconds = time.perf_counter() - loaded_at
    started = time.perf_counter()
    results = model.transcribe([args.audio.stem], [str(args.audio)])
    inference_seconds = time.perf_counter() - started
    result = results[0] if results else {"text": ""}
    print(json.dumps({
        "text": result.get("text", "").strip(),
        "confidence": result.get("confidence"),
        "timestamps": result.get("timestamp", []),
        "load_seconds": round(load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
