import datetime as dt
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from processor import (
    apply_asr_adjudication,
    asr_reliability,
    EventPostError,
    EvidenceProcessor,
    TranscriptionTimeout,
    acoustic_scene_without_reference,
    acoustic_scene_with_playback,
    atomic_json,
    audio_properties,
    audio_refs_for_segments,
    choose_microphone,
    choose_microphone_result,
    concatenate_audio,
    extract_speech_audio,
    merge_vad_segments,
    normalized_agreement,
    owned_vad_segments,
    parse_capture_time,
    playback_matches_for_window,
    sensevoice_consensus,
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
            firered_enabled=True,
            firered_python=root / "python",
            firered_script=root / "firered_transcribe.py",
            firered_source_dir=root / "source",
            firered_deps_dir=root / "deps",
            firered_model_dir=root / "model",
            firered_timeout_seconds=45,
        ))

    def test_agreement_ignores_spacing_and_punctuation(self):
        self.assertEqual(normalized_agreement("明天，交水费。", "明天交水费"), 1.0)

    def test_one_character_model_conflict_requires_review(self):
        result = asr_reliability("小爱仍有功能", ["小爱原有功能"] * 3)
        self.assertEqual(result["agreement"], "medium")
        self.assertTrue(result["needs_review"])

    def test_exact_three_microphone_consensus_is_high_but_not_infallible(self):
        result = asr_reliability("明天交水费", ["明天，交水费。"] * 3)
        self.assertEqual(result["agreement"], "high")
        self.assertFalse(result["needs_review"])
        self.assertIn("仍可能听错", result["notes"])

    def test_fire_red_can_confirm_three_microphone_consensus(self):
        alternatives = ["小爱原有功能"] * 3
        self.assertEqual(sensevoice_consensus(alternatives), "小爱原有功能")
        selected, result = apply_asr_adjudication("小爱仍有功能", alternatives, "小爱原有功能")
        self.assertEqual(selected, "小爱原有功能")
        self.assertEqual(result, "sensevoice_consensus_confirmed")

    def test_fire_red_disagreement_never_silently_rewrites_primary(self):
        selected, result = apply_asr_adjudication("明天交水费", ["明天交电费"] * 3, "明天交燃气费")
        self.assertEqual(selected, "明天交水费")
        self.assertEqual(result, "three_way_conflict")

    def test_two_sensevoice_results_are_not_a_strong_consensus(self):
        self.assertIsNone(sensevoice_consensus(["明天交水费", "明天交水费", ""]))

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

    def test_microphone_medoid_rejects_outlier(self):
        self.assertEqual(choose_microphone(["明天交水费", "明天记得交水费", "今天去浇花"]), 0)

    def test_microphone_result_falls_back_to_strongest_vad_channel(self):
        self.assertEqual(choose_microphone_result([(1, "So."), (4, ""), (2, "")]), 1)

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

    def test_qwen_transcription_has_a_hard_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            processor.qwen = mock.Mock()
            processor.qwen.transcribe.side_effect = lambda *args, **kwargs: __import__("time").sleep(1)
            with self.assertRaises(TranscriptionTimeout):
                processor.qwen_transcribe(Path(directory) / "audio.wav")

    def test_firered_adjudicator_is_isolated_and_parses_json(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            completed = mock.Mock(stdout='logs\n{"text":"原有","confidence":0.98}\n')
            with mock.patch("processor.subprocess.run", return_value=completed) as run:
                result = processor.firered_transcribe(Path(directory) / "speech.wav")
            self.assertEqual(result["text"], "原有")
            self.assertEqual(run.call_args.kwargs["timeout"], 45)
            command = run.call_args.args[0]
            self.assertIn("--audio", command)
            self.assertIn("--model-dir", command)

    def test_firered_timeout_falls_back_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = self.make_processor(directory)
            with mock.patch("processor.subprocess.run", side_effect=subprocess.TimeoutExpired("firered", 45)):
                self.assertIsNone(processor.firered_transcribe(Path(directory) / "speech.wav"))

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
