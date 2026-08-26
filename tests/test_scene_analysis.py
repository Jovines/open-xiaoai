from pathlib import Path
import subprocess
import tempfile
import unittest

from scene_analysis import array_spatial_features, infer_scene, lexical_repetition_score


class SceneAnalysisTests(unittest.TestCase):
    def test_missing_models_remains_unknown_instead_of_guessing_a_device(self):
        result = infer_scene({"scene_type": "unknown", "playback_reference": {"available": False}}, None, "你好")
        self.assertEqual(result["primary"], "unknown")
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["signals"]["audio_tagging_status"], "models_unavailable")

    def test_digital_playback_reference_has_priority_over_coarse_tags(self):
        result = infer_scene(
            {"scene_type": "xiaoai_dialogue", "playback_reference": {"available": True, "coverage": 0.8}},
            {"model": "CED", "tags": [{"label": "Speech", "probability": 0.99}]},
            "明天天气怎么样",
        )
        self.assertEqual(result["primary"], "xiaoai_dialogue")
        self.assertTrue(result["signals"]["xiaoai_playback_reference"])

    def test_media_tags_never_claim_a_specific_phone_application(self):
        result = infer_scene(
            {"scene_type": "unknown", "playback_reference": {"available": False}},
            {"model": "CED", "tags": [{"label": "Television", "probability": 0.88}, {"label": "Speech", "probability": 0.5}]},
            "learn the word apple",
        )
        self.assertIn(result["primary"], {"television_or_remote_media", "phone_or_computer_media_playback"})
        duolingo = next((item["probability"] for item in result["candidates"] if item["label"] == "phone_language_learning_playback"), 0)
        self.assertEqual(duolingo, 0)
        self.assertEqual(result["signals"]["custom_household_classifier"]["status"], "awaiting_labeled_household_samples")

    def test_english_drill_with_music_is_a_generic_language_learning_candidate(self):
        result = infer_scene(
            {"scene_type": "unknown", "playback_reference": {"available": False}},
            {"model": "CED", "tags": [
                {"label": "Speech", "probability": 0.88},
                {"label": "Music", "probability": 0.51},
                {"label": "Narration, monologue", "probability": 0.35},
            ]},
            "species species endangered endangered save save countryside grocery",
        )
        self.assertEqual(result["primary"], "phone_language_learning_playback")
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["signals"]["custom_household_classifier"]["status"], "awaiting_labeled_household_samples")

    def test_short_normal_english_does_not_look_like_a_repetition_drill(self):
        self.assertEqual(lexical_repetition_score("How are you today?"), 0.0)

    def test_three_channel_features_are_geometry_neutral(self):
        try:
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("optional runtime audio dependencies are not installed in system Python")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for channel in range(3):
                path = root / f"mic-{channel}.wav"
                subprocess.run([
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={440 + channel}:sample_rate=16000",
                    "-t", "0.6", str(path),
                ], check=True)
                paths.append(path)
            result = array_spatial_features(paths)
            self.assertEqual(result["status"], "geometry_unconfigured")
            self.assertEqual(len(result["pairwise"]), 3)
            self.assertIsNone(result["direction"])


if __name__ == "__main__":
    unittest.main()
