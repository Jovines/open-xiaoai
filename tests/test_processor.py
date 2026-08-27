import datetime as dt
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import numpy as np

from array_enhancement import enhance_speech_audio, fuse_vad_segments, gcc_phat_delay
from processor import (
    asr_timeout_budget,
    asr_reliability,
    EventPostError,
    EvidenceProcessor,
    TranscriptionTimeout,
    acoustic_scene_without_reference,
    acoustic_scene_with_playback,
    atomic_json,
    audio_properties,
    audio_refs_for_segments,
    concatenate_audio,
    extract_speech_audio,
    merge_vad_segments,
    owned_vad_segments,
    parse_capture_time,
    playback_matches_for_window,
    vad_segments,
)


class ProcessorTests(unittest.TestCase):
    def make_processor(self, directory: str) -> EvidenceProcessor:
        root = Path(directory)
        return EvidenceProcessor(SimpleNamespace(
            evidence_dir=root / "evidence",
            discard_dir=root / "discarded",
            discard_grace_hours=72,
            settle_seconds=0,
            lookahead_wait_seconds=75,
            zeris_url="http://zeris.invalid/events",
            zeris_token="test",
            qwen_timeout_seconds=0.02,
            qwen_max_new_tokens=384,
            array_min_vad_channels=2,
            array_max_delay_ms=2.0,
            array_min_coherence=0.15,
            array_reference_weight=0.70,
            asr_max_timeout_seconds=300,
        ))

    def test_directional_transcripts_are_not_adjudicated_by_capture_layer(self):
        result = asr_reliability(["明天交水费", "明天交水电费", "明天交水费"])
        self.assertEqual(result["agreement"], "not_adjudicated")
        self.assertFalse(result["needs_review"])
        self.assertIsNone(result["score"])
        self.assertIn("不投票、不要求一致", result["notes"])

    def test_empty_direction_is_preserved_without_rejecting_other_views(self):
        result = asr_reliability(["", "明天交水费", ""])
        self.assertEqual(result["agreement"], "not_adjudicated")
        self.assertIn("1 路方向麦克风", result["notes"])

    def test_asr_timeout_scales_for_a_full_minute_of_speech(self):
        with mock.patch("processor.audio_properties", return_value={"duration_seconds": 60}):
            self.assertEqual(asr_timeout_budget(Path("speech.wav"), 180, 300), 210)

    def test_asr_timeout_keeps_a_real_hang_ceiling(self):
        with mock.patch("processor.audio_properties", return_value={"duration_seconds": 600}):
            self.assertEqual(asr_timeout_budget(Path("speech.wav"), 180, 300), 300)

    def test_missing_playback_reference_never_claims_a_speaker_origin(self):
        scene = acoustic_scene_without_reference()
        self.assertEqual(scene["scene_type"], "unknown")
        self.assertTrue(scene["needs_review"])
        self.assertFalse(scene["playback_reference"]["available"])
        self.assertEqual(scene["turns"], [])

    def test_playback_manifest_overlap_becomes_bounded_nas_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root / "2026/08/25"
            day.mkdir(parents=True)
            audio = day / "playback.flac"
            audio.write_bytes(b"reference")
            atomic_json(day / "playback.json", {
                "kind": "oh2p_playback_reference",
                "occurred_at": "2026-08-25T02:00:05+00:00",
                "ended_at": "2026-08-25T02:00:10+00:00",
                "audio_file": audio.name,
                "audio_sha256": "b" * 64,
            })
            matches = playback_matches_for_window(
                dt.datetime.fromisoformat("2026-08-25T10:00:00+08:00"),
                dt.datetime.fromisoformat("2026-08-25T10:00:08+08:00"),
                root,
                "nas://archive/playback",
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["offset_start_seconds"], 0.0)
            self.assertEqual(matches[0]["offset_end_seconds"], 3.0)
            self.assertEqual(matches[0]["event_start_seconds"], 5.0)

    def test_scene_keeps_playback_and_overlapping_near_end_uncertainty_separate(self):
        scene = acoustic_scene_with_playback(
            "a" * 64,
            8.0,
            [(0.0, 2.0), (5.0, 7.0)],
            [{"event_start_seconds": 4.0, "event_end_seconds": 8.0}],
        )
        self.assertEqual(scene["scene_type"], "xiaoai_dialogue")
        self.assertTrue(scene["playback_reference"]["available"])
        origins = [item["origin"] for item in scene["turns"]]
        self.assertIn("human", origins)
        self.assertIn("xiaoai_output", origins)
        self.assertIn("unknown", origins)
        self.assertNotIn("overlap", origins)
        self.assertTrue(scene["needs_review"])

    def test_capture_time_uses_embedded_timezone(self):
        value = parse_capture_time(Path("2026-08-25_08-30-00_+0800.flac"))
        self.assertEqual(value.utcoffset(), dt.timedelta(hours=8))
        self.assertEqual(value.hour, 8)

    def test_atomic_json_never_leaves_part_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "event.json"
            atomic_json(path, {"text": "中文"})
            self.assertIn("中文", path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_name(f".{path.name}.part").exists())

    def test_audio_properties_are_read_from_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", "0.25", "-c:a", "pcm_s16le", str(fixture),
            ], check=True)
            properties = audio_properties(fixture)
            self.assertEqual(properties["sample_rate"], 16000)
            self.assertEqual(properties["channels"], 1)
            self.assertAlmostEqual(properties["duration_seconds"], 0.25, places=2)

    def test_vad_segments_parses_only_valid_intervals(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="[vad] 2 segments (max_seg=30000ms)\n100 500\n900 1300\ninvalid\n",
            stderr="",
        )
        with mock.patch("processor.subprocess.run", return_value=completed):
            self.assertEqual(
                vad_segments(Path("vad"), Path("model"), Path("audio.wav")),
                [(100, 500), (900, 1300)],
            )

    def test_vad_padding_merges_overlapping_intervals(self):
        self.assertEqual(merge_vad_segments([(100, 500), (600, 900), (2000, 2100)]), [(0, 1100), (1800, 2300)])

    def test_vad_fusion_requires_two_distinct_microphones(self):
        channels = [
            [(100, 600), (500, 800)],
            [(200, 700)],
            [(1200, 1500)],
        ]
        self.assertEqual(fuse_vad_segments(channels, min_channels=2), [(200, 700)])

    def test_gcc_phat_reports_positive_delay_for_later_microphone(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(0, 0.1, 4096).astype(np.float32)
        signal = np.concatenate((np.zeros(5, dtype=np.float32), reference[:-5]))
        delay, coherence = gcc_phat_delay(reference, signal, max_delay_samples=16)
        self.assertAlmostEqual(delay, 5, delta=0.6)
        self.assertGreater(coherence, 0.95)

    @staticmethod
    def write_wave(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
        encoded = np.rint(np.clip(samples, -0.98, 0.98) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(sample_rate)
            destination.writeframes(encoded.tobytes())

    def test_array_enhancement_aligns_good_channels_and_rejects_noise_outlier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(11)
            excitation = rng.normal(0, 0.12, 16000).astype(np.float32)
            base = np.convolve(excitation, np.ones(7, dtype=np.float32) / 7, mode="same")
            mic0 = base + rng.normal(0, 0.01, base.size)
            mic1 = np.concatenate((np.zeros(4), base[:-4])) + rng.normal(0, 0.01, base.size)
            mic2 = rng.normal(0, 0.25, base.size)
            microphones = []
            for index, samples in enumerate((mic0, mic1, mic2)):
                path = root / f"mic-{index}.wav"
                self.write_wave(path, samples)
                microphones.append(path)
            output = root / "enhanced.wav"
            metadata = enhance_speech_audio(microphones, output, [(0, 1000)])
            detail = metadata["segments"][0]
            self.assertEqual(detail["mode"], "reference_preserving_delay_and_sum")
            self.assertLess(detail["weights"][2], 0.05)
            self.assertGreaterEqual(detail["weights"][metadata["reference_channel"]], 0.7)
            self.assertGreater(metadata["quality_score"], 0.3)
            with wave.open(str(output), "rb") as enhanced:
                samples = np.frombuffer(enhanced.readframes(enhanced.getnframes()), dtype="<i2")
            self.assertLessEqual(int(np.max(np.abs(samples))), round(32767 * 0.98))

    def test_speech_extraction_removes_long_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            output = Path(directory) / "speech.wav"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                "-t", "2", str(source),
            ], check=True)
            extract_speech_audio(source, output, [(200, 500), (1500, 1800)])
            self.assertAlmostEqual(audio_properties(output)["duration_seconds"], 1.4, places=1)

    def test_audio_concatenation_preserves_both_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.wav"
            second = Path(directory) / "second.wav"
            output = Path(directory) / "joined.wav"
            for path in (first, second):
                subprocess.run([
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "0.3", str(path),
                ], check=True)
            concatenate_audio([first, second], output)
            self.assertAlmostEqual(audio_properties(output)["duration_seconds"], 0.6, places=1)

    def test_boundary_ownership_keeps_crossing_utterance_only_upstream(self):
        segments = [(1000, 2500), (59000, 62000), (70000, 72000)]
        self.assertEqual(owned_vad_segments(segments, 60000), [(1000, 2500), (59000, 62000)])
        continuation_view = [(0, 2000), (10000, 12000)]
        self.assertEqual(owned_vad_segments(continuation_view, 60000, carried_until_ms=2000), [(10000, 12000)])

    def test_cross_boundary_refs_include_offsets_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "2026-08-25_09-00-00_+0800.flac"
            second = root / "2026-08-25_09-01-00_+0800.flac"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            refs = audio_refs_for_segments(
                [first, second], [60000, 60000], [(59000, 62000)], "nas://archive/evidence",
            )
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0]["offset_start_seconds"], 59.0)
            self.assertEqual(refs[1]["offset_end_seconds"], 2.0)
            self.assertRegex(refs[0]["sha256"], r"^[a-f0-9]{64}$")

    def test_retryable_failure_does_not_block_later_pending_event(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            day = processor.args.evidence_dir / "2026/08/25"
            day.mkdir(parents=True)
            first = day / "2026-08-25_02-00-00_+0800.flac.event.pending.json"
            second = day / "2026-08-25_02-01-00_+0800.flac.event.pending.json"
            atomic_json(first, {"transcript": "第一条"})
            atomic_json(second, {"transcript": "第二条"})

            def post(event):
                if event["transcript"] == "第一条":
                    raise EventPostError("temporary", retryable=True)

            with mock.patch.object(processor, "post_event", side_effect=post):
                self.assertEqual(processor.retry_pending_events(), 1)

            self.assertTrue(first.is_file())
            self.assertTrue(Path(str(second).replace(".event.pending.json", ".event.json")).is_file())
            self.assertIn(first, processor.pending_retry_after)

    def test_post_event_rejects_empty_transcript_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            with self.assertRaises(EventPostError) as caught:
                processor.post_event({"transcript": "  "})
            self.assertFalse(caught.exception.retryable)

    def test_qwen_transcription_has_a_hard_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            processor.qwen = mock.Mock()
            processor.qwen.transcribe.side_effect = lambda *args, **kwargs: __import__("time").sleep(1)
            with self.assertRaises(TranscriptionTimeout):
                processor.qwen_transcribe(Path(directory) / "audio.wav")

    def test_processing_failure_is_preserved_without_expiry(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            source = Path(directory) / "2026-08-25_09-10-13_+0800.flac"
            source.write_bytes(b"possible-speech")
            evidence = processor.preserve_processing_failure(
                source,
                reason="primary_asr_timeout",
                details={"timeout_seconds": 45},
            )
            failure = evidence.with_suffix(evidence.suffix + ".processing_failed.json")
            self.assertTrue(evidence.is_file())
            self.assertTrue(failure.is_file())
            self.assertNotIn("delete_after", failure.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
