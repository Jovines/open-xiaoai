#!/usr/bin/env python3
"""Geometry-neutral enhancement for the three synchronized OH2P microphones.

The raw 48 kHz/24-bit array recording remains the source of truth.  This module
only builds a compact 16 kHz mono derivative for VAD, ASR and acoustic models.
It deliberately performs no speech recognition.
"""

from __future__ import annotations

from pathlib import Path
import math
import wave

import numpy as np


def _coalesce(segments: list[tuple[int, int]], join_gap_ms: int = 0) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(segments):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + join_gap_ms:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([max(0, start), end])
    return [(start, end) for start, end in merged]


def fuse_vad_segments(
    channels: list[list[tuple[int, int]]],
    *,
    min_channels: int = 2,
    join_gap_ms: int = 120,
) -> list[tuple[int, int]]:
    """Return time spans supported by at least ``min_channels`` microphones.

    Per-channel intervals are first coalesced so overlapping intervals from one
    detector never count as multiple microphones.  A short join closes VAD
    flicker without adding padding; ASR edge padding is applied later.
    """
    if not channels:
        return []
    if min_channels < 1 or min_channels > len(channels):
        raise ValueError("min_channels must be within the microphone count")
    changes: dict[int, int] = {}
    for segments in channels:
        for start, end in _coalesce(segments):
            changes[start] = changes.get(start, 0) + 1
            changes[end] = changes.get(end, 0) - 1

    active = 0
    opened: int | None = None
    result: list[tuple[int, int]] = []
    for timestamp in sorted(changes):
        before = active
        active += changes[timestamp]
        if before < min_channels <= active:
            opened = timestamp
        elif before >= min_channels > active and opened is not None:
            result.append((opened, timestamp))
            opened = None
    return _coalesce(result, join_gap_ms=join_gap_ms)


def _read_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"array enhancer expects mono PCM16 WAV: {path}")
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return sample_rate, samples.astype(np.float32) / 32768.0


def _write_pcm16_mono(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bounded = np.clip(samples, -0.98, 0.98)
    encoded = np.rint(bounded * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(encoded.tobytes())


def _aligned(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Advance a delayed signal by ``delay_samples`` using linear interpolation."""
    positions = np.arange(signal.size, dtype=np.float64)
    return np.interp(positions + delay_samples, positions, signal, left=0.0, right=0.0).astype(np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size < 32:
        return 0.0
    a = left[:size].astype(np.float64) - float(np.mean(left[:size]))
    b = right[:size].astype(np.float64) - float(np.mean(right[:size]))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def gcc_phat_delay(
    reference: np.ndarray,
    signal: np.ndarray,
    *,
    max_delay_samples: int,
) -> tuple[float, float]:
    """Estimate array delay and aligned waveform coherence.

    A positive delay means ``signal`` arrived later than ``reference``.  The
    second return value is ordinary aligned correlation, not an ASR confidence.
    """
    size = min(reference.size, signal.size)
    if size < 64 or max_delay_samples < 0:
        return 0.0, 0.0
    x = reference[:size].astype(np.float64)
    y = signal[:size].astype(np.float64)
    x -= float(np.mean(x))
    y -= float(np.mean(y))
    if float(np.sqrt(np.mean(x * x))) < 1e-5 or float(np.sqrt(np.mean(y * y))) < 1e-5:
        return 0.0, 0.0

    # A Hann window reduces the false circular peak at the finite chunk edges.
    window = np.hanning(size)
    fft_size = 1 << (2 * size - 1).bit_length()
    cross = np.fft.rfft(y * window, fft_size) * np.conj(np.fft.rfft(x * window, fft_size))
    cross /= np.maximum(np.abs(cross), 1e-12)
    circular = np.fft.irfft(cross, fft_size)
    limit = min(int(max_delay_samples), size - 1)
    lags = np.arange(-limit, limit + 1)
    values = np.asarray([circular[lag] if lag >= 0 else circular[fft_size + lag] for lag in lags])
    magnitudes = np.abs(values)
    peak_index = int(np.argmax(magnitudes))
    delay = float(lags[peak_index])
    if 0 < peak_index < magnitudes.size - 1:
        left, center, right = magnitudes[peak_index - 1:peak_index + 2]
        curvature = left - 2 * center + right
        if abs(float(curvature)) > 1e-12:
            delay += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    coherence = max(0.0, _correlation(x.astype(np.float32), _aligned(y.astype(np.float32), delay)))
    return delay, coherence


def _enhance_chunk(
    chunks: list[np.ndarray],
    *,
    sample_rate: int,
    max_delay_ms: float,
    min_coherence: float,
    reference_index: int | None = None,
    reference_weight: float = 0.7,
) -> tuple[np.ndarray, dict]:
    max_delay_samples = max(1, round(sample_rate * max_delay_ms / 1000))
    reference_scores = []
    reference_delays: list[list[float]] = []
    for candidate_index, reference in enumerate(chunks):
        delays = []
        coherences = []
        for channel, signal in enumerate(chunks):
            if channel == candidate_index:
                delays.append(0.0)
                continue
            delay, coherence = gcc_phat_delay(reference, signal, max_delay_samples=max_delay_samples)
            delays.append(delay)
            coherences.append(coherence)
        reference_delays.append(delays)
        reference_scores.append(float(np.mean(coherences)) if coherences else 0.0)

    reference_index = int(np.argmax(reference_scores)) if reference_index is None else reference_index
    delays = reference_delays[reference_index]
    aligned = [_aligned(signal, delay) for signal, delay in zip(chunks, delays)]
    peer_scores = []
    for channel, signal in enumerate(aligned):
        peers = [max(0.0, _correlation(signal, other)) for index, other in enumerate(aligned) if index != channel]
        peer_scores.append(float(np.mean(peers)) if peers else 0.0)

    rms = np.asarray([float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) for signal in aligned])
    nonzero_rms = rms[rms > 1e-6]
    median_rms = float(np.median(nonzero_rms)) if nonzero_rms.size else 0.0
    weights = []
    for channel, (signal, coherence, level) in enumerate(zip(aligned, peer_scores, rms)):
        clipping = float(np.mean(np.abs(signal) >= 0.975))
        level_penalty = min(1.0, (median_rms * 2.5) / level) if level > 0 and median_rms > 0 else 0.0
        clipping_penalty = 0.2 if clipping > 0.001 else 1.0
        usable = channel == reference_index or coherence >= min_coherence
        weights.append((max(0.02, coherence) ** 2) * level_penalty * clipping_penalty if usable else 0.0)

    used = [index for index, weight in enumerate(weights) if weight > 0]
    if len(used) < 2 or max(reference_scores, default=0.0) < min_coherence:
        # With no trustworthy inter-channel relationship, summing can cancel
        # speech.  Preserve the strongest coherent channel as a safe fallback.
        best = int(np.argmax(np.asarray(peer_scores) - np.asarray([float(np.mean(np.abs(x) >= 0.975)) for x in aligned])))
        weights = [1.0 if index == best else 0.0 for index in range(len(chunks))]
        used = [best]
        mode = "single_channel_fallback"
        reference_index = best
    else:
        mode = "reference_preserving_delay_and_sum"

    total = float(sum(weights)) or 1.0
    normalized_weights = [float(weight / total) for weight in weights]
    if len(used) >= 2 and normalized_weights[reference_index] < reference_weight:
        other_total = 1.0 - normalized_weights[reference_index]
        remaining = max(0.0, 1.0 - reference_weight)
        normalized_weights = [
            reference_weight if index == reference_index else weight * remaining / other_total
            for index, weight in enumerate(normalized_weights)
        ]
    enhanced = sum(signal * weight for signal, weight in zip(aligned, normalized_weights))
    fade_samples = min(round(sample_rate * 0.005), enhanced.size // 2)
    if fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        enhanced[:fade_samples] *= fade
        enhanced[-fade_samples:] *= fade[::-1]
    quality = float(np.mean([peer_scores[index] for index in used])) if len(used) >= 2 else 0.0
    return enhanced.astype(np.float32), {
        "mode": mode,
        "reference_channel": reference_index,
        "delays_samples": [round(value, 3) for value in delays],
        "delays_ms": [round(value * 1000 / sample_rate, 4) for value in delays],
        "weights": [round(value, 4) for value in normalized_weights],
        "peer_coherence": [round(value, 4) for value in peer_scores],
        "quality": round(quality, 4),
        "used_channels": used,
    }


def enhance_speech_audio(
    microphones: list[Path],
    destination: Path,
    segments: list[tuple[int, int]],
    *,
    max_delay_ms: float = 2.0,
    min_coherence: float = 0.15,
    reference_weight: float = 0.7,
    padding_ms: int = 200,
) -> dict:
    """Align and fuse all microphones per speech region into one ASR waveform."""
    if len(microphones) < 2:
        raise ValueError("array enhancement requires at least two microphones")
    loaded = [_read_pcm16_mono(path) for path in microphones]
    sample_rates = {sample_rate for sample_rate, _ in loaded}
    if len(sample_rates) != 1:
        raise ValueError("microphone sample rates differ")
    sample_rate = sample_rates.pop()
    arrays = [samples for _, samples in loaded]
    size = min(array.size for array in arrays)
    padded = _coalesce([
        (max(0, start - padding_ms), end + padding_ms)
        for start, end in segments
    ])
    if not 1 / len(microphones) <= reference_weight <= 1:
        raise ValueError("reference_weight must be between equal weighting and 1")
    raw_chunks: list[tuple[int, int, list[np.ndarray]]] = []
    for start_ms, end_ms in padded:
        start = min(size, round(start_ms * sample_rate / 1000))
        end = min(size, round(end_ms * sample_rate / 1000))
        if end - start < 64:
            continue
        raw_chunks.append((start_ms, end_ms, [array[start:end] for array in arrays]))
    if not raw_chunks:
        raise ValueError("cannot enhance audio without non-empty speech regions")

    # Pick one anchor for the whole utterance.  Changing the anchor between
    # clauses measurably altered short Chinese words in the ASR benchmark.
    reference_votes = [0.0] * len(microphones)
    for start_ms, end_ms, chunks in raw_chunks:
        _, probe = _enhance_chunk(
            chunks,
            sample_rate=sample_rate,
            max_delay_ms=max_delay_ms,
            min_coherence=min_coherence,
            reference_weight=1 / len(microphones),
        )
        reference_votes[probe["reference_channel"]] += (end_ms - start_ms) * max(0.01, probe["quality"])
    global_reference = int(np.argmax(reference_votes))

    output_chunks = []
    details = []
    for start_ms, end_ms, chunks in raw_chunks:
        enhanced, detail = _enhance_chunk(
            chunks,
            sample_rate=sample_rate,
            max_delay_ms=max_delay_ms,
            min_coherence=min_coherence,
            reference_index=global_reference,
            reference_weight=reference_weight,
        )
        output_chunks.append(enhanced)
        details.append({"start_ms": start_ms, "end_ms": end_ms, **detail})

    output = np.concatenate(output_chunks)
    peak_before_scale = float(np.max(np.abs(output))) if output.size else 0.0
    scale = min(1.0, 0.98 / peak_before_scale) if peak_before_scale > 0 else 1.0
    output *= scale
    _write_pcm16_mono(destination, sample_rate, output)

    durations = np.asarray([item["end_ms"] - item["start_ms"] for item in details], dtype=np.float64)
    quality = float(np.average([item["quality"] for item in details], weights=durations))
    fallback_fraction = float(np.average(
        [1.0 if item["mode"] == "single_channel_fallback" else 0.0 for item in details],
        weights=durations,
    ))
    reference_duration = [0.0] * len(microphones)
    for item, duration in zip(details, durations):
        reference_duration[item["reference_channel"]] += float(duration)
    return {
        "algorithm": "gcc_phat_weighted_delay_and_sum",
        "sample_rate": sample_rate,
        "input_channels": len(microphones),
        "output_channels": 1,
        "max_delay_ms": max_delay_ms,
        "min_coherence": min_coherence,
        "reference_weight": reference_weight,
        "quality_score": round(quality, 4),
        "quality_kind": "array_coherence_not_transcript_probability",
        "fallback_fraction": round(fallback_fraction, 4),
        "reference_channel": global_reference,
        "peak_before_scale": round(peak_before_scale, 6),
        "output_scale": round(scale, 6),
        "segments": details,
    }
