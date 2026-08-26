from pathlib import Path
import tempfile
import unittest

from speaker_profiles import SpeakerProfileStore, cosine_similarity


def scene(*, primary="live_conversation", live=0.8, replay=None):
    return {
        "primary": primary,
        "candidates": [{"label": "live_conversation", "probability": live}],
        "signals": {"replay_score": replay},
    }


def sample(speaker=0, embedding=None):
    return {
        "speaker": speaker,
        "embedding": embedding or [1.0, 0.0, 0.0],
        "quality": {"speech_seconds": 3.0, "rms_dbfs": -20.0, "clipping_fraction": 0.0},
    }


def segments(speaker=0):
    return [{"speaker_id": f"recording-speaker-{speaker:02d}", "start_seconds": 0.0, "end_seconds": 3.0}]


class SpeakerProfileTests(unittest.TestCase):
    def test_cosine_similarity_is_scale_independent(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [8, 0]), 1.0)

    def test_similar_clean_samples_reuse_one_anonymous_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerProfileStore(Path(directory) / "profiles.json", threshold=0.8)
            first = store.assign(event_id="audio-one", samples=[sample()], speaker_segments=segments(), playback_intervals=[], scene=scene(), audio_refs=[])
            second = store.assign(event_id="audio-two", samples=[sample(embedding=[0.99, 0.05, 0])], speaker_segments=segments(), playback_intervals=[], scene=scene(), audio_refs=[])
            self.assertEqual(first["recording-speaker-00"]["profile_id"], second["recording-speaker-00"]["profile_id"])
            self.assertEqual(len(store.load()["profiles"]), 1)
            self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)

    def test_xiaoai_overlap_never_updates_a_voice_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerProfileStore(Path(directory) / "profiles.json")
            result = store.assign(
                event_id="audio-playback", samples=[sample()], speaker_segments=segments(),
                playback_intervals=[(0.0, 2.0)], scene=scene(primary="mixed_live_and_playback"), audio_refs=[],
            )
            self.assertEqual(result["recording-speaker-00"]["identity_status"], "profile_update_rejected")
            self.assertEqual(store.load()["profiles"], [])

    def test_non_live_scene_never_updates_a_voice_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerProfileStore(Path(directory) / "profiles.json")
            result = store.assign(
                event_id="audio-phone", samples=[sample()], speaker_segments=segments(),
                playback_intervals=[], scene=scene(primary="phone_or_computer_media_playback"), audio_refs=[],
            )
            assignment = result["recording-speaker-00"]
            self.assertEqual(assignment["identity_status"], "profile_update_rejected")
            self.assertIn("scene_not_live_speech_primary", assignment["profile_update_reasons"])
            self.assertEqual(store.load()["profiles"], [])

    def test_human_label_is_withheld_until_live_and_replay_gates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerProfileStore(Path(directory) / "profiles.json", threshold=0.8)
            first = store.assign(event_id="audio-one", samples=[sample()], speaker_segments=segments(), playback_intervals=[], scene=scene(), audio_refs=[])
            profile_id = first["recording-speaker-00"]["profile_id"]
            store.label(profile_id, "家庭成员甲", "household-owner")
            uncertain = store.assign(event_id="audio-two", samples=[sample()], speaker_segments=segments(), playback_intervals=[], scene=scene(replay=None), audio_refs=[])
            verified = store.assign(event_id="audio-three", samples=[sample()], speaker_segments=segments(), playback_intervals=[], scene=scene(replay=0.1), audio_refs=[])
            self.assertIsNone(uncertain["recording-speaker-00"]["identity_label"])
            self.assertEqual(verified["recording-speaker-00"]["identity_label"], "家庭成员甲")


if __name__ == "__main__":
    unittest.main()
