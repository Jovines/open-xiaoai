import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_historical_replay.py"
SPEC = importlib.util.spec_from_file_location("publish_historical_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HistoricalReplayTests(unittest.TestCase):
    def test_replay_event_is_deterministic_and_preserves_original(self):
        original = {"event_id": "audio-original-0001", "transcript": "测试", "provenance": {"schema_version": 1}}
        first = MODULE.replay_event(original, "return-home-v1", "2026-08-27T00:00:00+00:00")
        second = MODULE.replay_event(original, "return-home-v1", "2026-08-27T00:00:00+00:00")
        self.assertEqual(first, second)
        self.assertRegex(first["event_id"], r"^audio-replay-[a-f0-9]{24}$")
        self.assertEqual(first["provenance"]["historical_replay"]["original_event_id"], original["event_id"])
        self.assertTrue(first["provenance"]["historical_replay"]["read_only_replay"])
        self.assertNotIn("historical_replay", original["provenance"])


if __name__ == "__main__":
    unittest.main()
