import json
from pathlib import Path
import tempfile
import unittest

from evaluation.scene_benchmark import load_manifest, metrics, sliced_metrics


class SceneBenchmarkTests(unittest.TestCase):
    def test_metrics_include_unknown_rejection_and_calibration(self):
        result = metrics([
            {"label": "live_conversation", "prediction": "live_conversation", "confidence": 0.8},
            {"label": "unknown", "prediction": "unknown", "confidence": 0.7},
            {"label": "unknown", "prediction": "live_conversation", "confidence": 0.6},
        ])
        self.assertAlmostEqual(result["unknown_recall"], 0.5)
        self.assertEqual(result["known_rejected_as_unknown"], 0.0)
        self.assertIn("expected_calibration_error_10_bins", result)

    def test_manifest_rejects_duplicate_ids_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("\n".join([
                json.dumps({"id": "same", "audio": "a.wav", "label": "live_conversation"}),
                json.dumps({"id": "same", "audio": "b.wav", "label": "live_conversation"}),
            ]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)
            path.write_text(json.dumps({"id": "one", "audio": "a.wav", "label": "duolingo"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_slice_metrics_exposes_device_specific_failures(self):
        result = sliced_metrics([
            {"label": "live_conversation", "prediction": "live_conversation", "confidence": 0.8, "device": "none"},
            {"label": "phone_or_computer_media_playback", "prediction": "unknown", "confidence": 0.6, "device": "phone-a"},
        ], "device")
        self.assertEqual(result["none"]["accuracy"], 1.0)
        self.assertEqual(result["phone-a"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
