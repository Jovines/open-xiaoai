import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "device" / "audio_fanout.c"
HEADER = struct.Struct("<4sHHQQQII")
FRAME_BYTES = 1920


class AudioFanoutTests(unittest.TestCase):
    def compile(self, directory: Path) -> Path:
        binary = directory / "audio_fanout"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            str(SOURCE), "-o", str(binary),
        ], check=True)
        return binary

    @staticmethod
    def read_exact_fifo(path: Path, size: int, output: list[bytes]) -> None:
        with path.open("rb", buffering=0) as stream:
            data = bytearray()
            while len(data) < size:
                chunk = stream.read(size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            output.append(bytes(data))

    def test_absent_tap_reader_never_blocks_main_stream(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            binary = self.compile(directory)
            compat, main, tap = (directory / name for name in ("compat.fifo", "main.fifo", "tap.fifo"))
            os.mkfifo(main)
            expected = bytes((index % 251 for index in range(FRAME_BYTES * 8)))
            received = []
            reader = threading.Thread(target=self.read_exact_fifo, args=(main, len(expected), received), daemon=True)
            reader.start()
            completed = subprocess.run(
                [str(binary), str(compat), str(main), str(tap)],
                input=expected, timeout=3, check=True,
            )
            self.assertEqual(completed.returncode, 0)
            reader.join(timeout=2)
            self.assertEqual(received, [expected])

    def test_tap_frames_are_atomic_timestamped_and_sequenced(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            binary = self.compile(directory)
            compat, main, tap = (directory / name for name in ("compat.fifo", "main.fifo", "tap.fifo"))
            os.mkfifo(main)
            os.mkfifo(tap)
            expected = b"a" * FRAME_BYTES + b"b" * 317
            main_received, tap_received = [], []
            main_reader = threading.Thread(target=self.read_exact_fifo, args=(main, len(expected), main_received), daemon=True)
            tap_reader = threading.Thread(
                target=self.read_exact_fifo,
                args=(tap, HEADER.size + FRAME_BYTES + HEADER.size + 317, tap_received),
                daemon=True,
            )
            main_reader.start()
            tap_reader.start()
            subprocess.run([str(binary), str(compat), str(main), str(tap)], input=expected, timeout=3, check=True)
            main_reader.join(timeout=2)
            tap_reader.join(timeout=2)
            self.assertEqual(main_received, [expected])
            packet = tap_received[0]
            first = HEADER.unpack_from(packet, 0)
            second_offset = HEADER.size + FRAME_BYTES
            second = HEADER.unpack_from(packet, second_offset)
            self.assertEqual(first[:3], (b"OXR1", 1, HEADER.size))
            self.assertGreater(first[3], 0)
            self.assertEqual(first[4], 0)
            self.assertGreater(first[5], 0)
            self.assertEqual(first[6:], (FRAME_BYTES, 0))
            self.assertEqual(packet[HEADER.size:second_offset], b"a" * FRAME_BYTES)
            self.assertEqual(second[:3], (b"OXR1", 1, HEADER.size))
            self.assertEqual(second[3], first[3])
            self.assertEqual(second[4], 1)
            self.assertEqual(second[6:], (317, 1))
            self.assertEqual(packet[second_offset + HEADER.size:], b"b" * 317)


if __name__ == "__main__":
    unittest.main()
