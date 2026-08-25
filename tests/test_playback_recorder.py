import datetime as dt
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace

from playback_recorder import FrameParser, HEADER, MAGIC, PCM_FRAME_BYTES, PlaybackRecorder


def packet(stream_id: int, sequence: int, timestamp_ns: int, payload: bytes, flags: int = 0) -> bytes:
    return HEADER.pack(MAGIC, 1, HEADER.size, stream_id, sequence, timestamp_ns, len(payload), flags) + payload


class PlaybackRecorderTests(unittest.TestCase):
    def test_parser_resynchronizes_and_accepts_fragmented_multi_stream_frames(self):
        parser = FrameParser()
        data = b"noise" + packet(11, 0, 100, b"a" * PCM_FRAME_BYTES) + packet(22, 7, 200, b"tail", 1)
        frames = []
        for index in range(0, len(data), 137):
            frames.extend(parser.feed(data[index:index + 137]))
        self.assertEqual([(item["stream_id"], item["sequence"], len(item["payload"])) for item in frames], [(11, 0, PCM_FRAME_BYTES), (22, 7, 4)])

    def test_session_archive_fills_sequence_gap_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            recorder = PlaybackRecorder(SimpleNamespace(output_dir=root, inactivity_seconds=0, required_mount=None))
            timestamp = int(dt.datetime(2026, 8, 25, tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)
            recorder.consume(packet(99, 0, timestamp, b"a" * PCM_FRAME_BYTES))
            recorder.consume(packet(99, 2, timestamp + 20_000_000, b"b" * PCM_FRAME_BYTES))
            self.assertEqual(recorder.finalize_inactive(force=True), 1)
            manifests = list(root.rglob("*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["stream_id"], "0000000000000063")
            self.assertEqual(manifest["dropped_frames_filled_with_silence"], 1)
            self.assertAlmostEqual(manifest["duration_seconds"], 0.03, places=3)
            self.assertTrue(manifests[0].with_suffix(".flac").is_file())
            self.assertFalse(list(root.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()
