import datetime as dt
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from processor import (
    EventPostError,
    EvidenceProcessor,
    atomic_json,
    audio_properties,
    choose_microphone,
    normalized_agreement,
    parse_capture_time,
)


class ProcessorTests(unittest.TestCase):
    def make_processor(self, directory: str) -> EvidenceProcessor:
        root = Path(directory)
        return EvidenceProcessor(SimpleNamespace(
            evidence_dir=root / "evidence",
            discard_dir=root / "discarded",
            discard_grace_hours=72,
            settle_seconds=0,
            zeris_url="http://zeris.invalid/events",
            zeris_token="test",
        ))

    def test_agreement_ignores_spacing_and_punctuation(self):
        self.assertEqual(normalized_agreement("明天，交水费。", "明天交水费"), 1.0)

    def test_microphone_medoid_rejects_outlier(self):
        self.assertEqual(choose_microphone(["明天交水费", "明天记得交水费", "今天去浇花"]), 0)

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

    def test_empty_primary_with_one_active_channel_is_noise_false_positive(self):
        event = {
            "transcript": "",
            "provenance": {"cross_check": {"all_microphone_transcripts": ["So.", "", ""]}},
        }
        self.assertTrue(EvidenceProcessor.is_single_channel_false_positive(event))
        event["transcript"] = "明天交水费"
        self.assertFalse(EvidenceProcessor.is_single_channel_false_positive(event))

    def test_retry_quarantines_known_false_positive_then_sends_next_event(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            day = processor.args.evidence_dir / "2026/08/25"
            day.mkdir(parents=True)
            noisy_audio = day / "2026-08-25_01-12-11_+0800.flac"
            noisy_audio.write_bytes(b"fan-noise")
            noisy_pending = noisy_audio.with_suffix(".flac.event.pending.json")
            atomic_json(noisy_pending, {
                "transcript": "",
                "provenance": {"cross_check": {"all_microphone_transcripts": ["So.", "", ""]}},
            })
            speech_audio = day / "2026-08-25_01-13-11_+0800.flac"
            speech_audio.write_bytes(b"speech")
            speech_pending = speech_audio.with_suffix(".flac.event.pending.json")
            atomic_json(speech_pending, {"transcript": "明天交水费"})

            with mock.patch.object(processor, "post_event") as post_event:
                self.assertEqual(processor.retry_pending_events(), 1)

            post_event.assert_called_once_with({"transcript": "明天交水费"})
            self.assertFalse(noisy_audio.exists())
            self.assertFalse(noisy_pending.exists())
            self.assertTrue((processor.args.discard_dir / "2026/08/25/2026-08-25_01-12-11_+0800.flac.quarantine").is_file())
            self.assertTrue(speech_audio.with_suffix(".flac.event.json").is_file())

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


if __name__ == "__main__":
    unittest.main()
