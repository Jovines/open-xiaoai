import argparse
import datetime as dt
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import recorder


def make_args(**overrides):
    values = {
        "host": "192.168.8.242",
        "user": "root",
        "output_dir": Path("recordings"),
        "required_mount": None,
        "sample_rate": 48000,
        "archive_channels": 3,
        "pcm_gain": 96.0,
        "segment_seconds": 3600,
        "codec": "opus",
        "bitrate": "96k",
        "retry_seconds": 5.0,
        "retention_days": 0,
        "min_free_gb": 5.0,
        "max_segments": 0,
        "log_level": "INFO",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CommandTests(unittest.TestCase):
    def test_ssh_uses_password_helper_when_password_is_present(self):
        with mock.patch.dict(os.environ, {"SSHPASS": "test"}, clear=False):
            command = recorder.build_ssh_command(make_args(segment_seconds=10))
        self.assertEqual(command[:3], ["sshpass", "-e", "ssh"])
        self.assertIn("root@192.168.8.242", command)
        self.assertIn("-r 48000 -c 4", command[-1])
        self.assertNotIn("-d 10", command[-1])
        self.assertTrue(command[-1].endswith("-c 4"))

    def test_ssh_requires_key_in_non_interactive_mode_without_password(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            command = recorder.build_ssh_command(make_args())
        self.assertEqual(command[0], "ssh")
        self.assertIn("BatchMode=yes", command)

    def test_ffmpeg_applies_a113_sample_correction_with_headroom(self):
        command = recorder.build_ffmpeg_command(make_args(), Path("capture.part"))
        self.assertIn("volume=96,pan=3.0|FL=c0|FR=c1|FC=c2", command)
        self.assertIn("libopus", command)
        self.assertIn("96k", command)

    def test_mono_mode_keeps_the_first_microphone(self):
        command = recorder.build_ffmpeg_command(
            make_args(archive_channels=1), Path("capture.part")
        )
        self.assertIn("volume=96,pan=mono|c0=c0", command)

    def test_gain_can_be_raised_for_controlled_diagnostics(self):
        command = recorder.build_ffmpeg_command(
            make_args(pcm_gain=256.0), Path("capture.part")
        )
        self.assertIn("volume=256,pan=3.0|FL=c0|FR=c1|FC=c2", command)

    def test_default_gain_leaves_headroom(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = recorder.parse_args([])
        self.assertEqual(args.pcm_gain, 96.0)

    def test_gain_above_sample_mapping_limit_is_rejected(self):
        with self.assertRaises(SystemExit):
            recorder.parse_args(["--pcm-gain", "257"])

    def test_lossless_mode_preserves_24_bit_samples(self):
        command = recorder.build_ffmpeg_command(
            make_args(codec="flac"), Path("capture.part")
        )
        self.assertIn("s32", command)


class FileTests(unittest.TestCase):
    def test_required_mount_prevents_local_fallback(self):
        args = make_args(required_mount=Path("/definitely/not/a/mount"))
        instance = recorder.ContinuousRecorder(args)
        self.assertFalse(instance.has_disk_space())

    def test_output_path_is_grouped_by_local_date(self):
        with tempfile.TemporaryDirectory() as directory:
            now = dt.datetime(2026, 8, 24, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=8)))
            temporary, final = recorder.output_paths(Path(directory), "opus", now)
            self.assertEqual(final.parent, Path(directory) / "2026" / "08" / "24")
            self.assertEqual(final.suffix, ".ogg")
            self.assertEqual(temporary.name, f".{final.name}.part")

    def test_retention_zero_never_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "old.ogg"
            recording.write_bytes(b"audio")
            self.assertEqual(recorder.cleanup_expired(Path(directory), 0), 0)
            self.assertTrue(recording.exists())

    def test_retention_deletes_only_expired_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.ogg"
            recent = root / "recent.ogg"
            ignored = root / "note.txt"
            for path in (old, recent, ignored):
                path.write_bytes(b"data")
            now = time.time()
            os.utime(old, (now - 3 * 86_400, now - 3 * 86_400))
            removed = recorder.cleanup_expired(root, 2, now=now)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(ignored.exists())


if __name__ == "__main__":
    unittest.main()
