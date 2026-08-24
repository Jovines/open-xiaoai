import datetime as dt
from pathlib import Path
import subprocess
import tempfile
import unittest

from processor import atomic_json, audio_properties, choose_microphone, normalized_agreement, parse_capture_time


class ProcessorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
