import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from evaluation import asr_benchmark


class MetricTests(unittest.TestCase):
    def test_normalize_zh_removes_tags_width_and_punctuation(self):
        self.assertEqual(asr_benchmark.normalize_zh("<|zh|>Ａ，小 爱！"), "A小爱")

    def test_edit_counts_distinguish_substitution(self):
        result = asr_benchmark.edit_counts("小爱原有功能", "小爱仍有功能")
        self.assertEqual(result["edits"], 1)
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 0)
        self.assertEqual(result["insertions"], 0)

    def test_aggregate_uses_micro_cer_and_flags_instability(self):
        case = {"id": "one", "reference": "原有", "critical_terms": ["原有"]}
        first = asr_benchmark.score_result(case, "仍有", 1.0, 0.1, 1)
        second = asr_benchmark.score_result(case, "原有", 1.0, 0.1, 2)
        result = asr_benchmark.aggregate([first, second])
        self.assertEqual(result["micro_cer"], 0.25)
        self.assertEqual(result["critical_term_accuracy"], 0.5)
        self.assertEqual(result["unstable_cases"], 1)

    def test_aggregate_by_tag_keeps_room_and_clean_separate(self):
        clean = asr_benchmark.score_result(
            {"id": "clean", "reference": "原有", "tags": ["clean"]}, "仍有", 1, 0.1, 1
        )
        room = asr_benchmark.score_result(
            {"id": "room", "reference": "原有", "tags": ["room"]}, "原有", 1, 0.1, 1
        )
        result = asr_benchmark.aggregate_by_tag([clean, room])
        self.assertEqual(result["clean"]["micro_cer"], 0.5)
        self.assertEqual(result["room"]["micro_cer"], 0.0)


class ManifestTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            row = json.dumps({"id": "same", "audio": "a.wav", "reference": "文本"}, ensure_ascii=False)
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "id 重复"):
                asr_benchmark.load_manifest(path)

    def test_audio_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")
            digest = hashlib.sha256(b"audio").hexdigest()
            self.assertEqual(
                asr_benchmark.resolve_audio({"audio": "sample.wav", "sha256": digest}, root),
                audio,
            )
            with self.assertRaisesRegex(ValueError, "哈希不符"):
                asr_benchmark.resolve_audio({"audio": "sample.wav", "sha256": "0" * 64}, root)


if __name__ == "__main__":
    unittest.main()
