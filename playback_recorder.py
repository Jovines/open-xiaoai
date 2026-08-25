#!/usr/bin/env python3
"""Archive framed OH2P playback-reference PCM without back-pressuring playback."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import time


MAGIC = b"OXR1"
HEADER = struct.Struct("<4sHHQQQII")
VERSION = 1
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_BYTES = 2
PCM_FRAME_BYTES = 1_920
MAX_GAP_FRAMES = 500


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="持续归档 OH2P 自身播放 reference")
    parser.add_argument("--host", default=os.environ.get("RECORDER_HOST", "192.168.8.242"))
    parser.add_argument("--user", default=os.environ.get("RECORDER_USER", "root"))
    parser.add_argument("--fifo", default=os.environ.get("PLAYBACK_FIFO", "/tmp/open_xiaoai_playback.fifo"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("PLAYBACK_OUTPUT_DIR", str(Path.home() / "recordings" / "open-xiaoai-playback"))))
    parser.add_argument("--required-mount", type=Path, default=Path(value) if (value := os.environ.get("PLAYBACK_REQUIRED_MOUNT")) else None)
    parser.add_argument("--inactivity-seconds", type=float, default=float(os.environ.get("PLAYBACK_INACTIVITY_SECONDS", "2")))
    parser.add_argument("--retry-seconds", type=float, default=float(os.environ.get("PLAYBACK_RETRY_SECONDS", "5")))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=os.environ.get("PLAYBACK_LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    ssh = ["sshpass", "-e", "ssh"] if os.environ.get("SSHPASS") else ["ssh", "-o", "BatchMode=yes"]
    ssh.extend([
        "-T", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "HostKeyAlgorithms=+ssh-rsa",
        f"{args.user}@{args.host}", f"cat {args.fifo}",
    ])
    return ssh


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mounted(path: Path | None) -> bool:
    return path is None or path.is_mount()


class FrameParser:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[dict]:
        self.buffer.extend(data)
        frames = []
        while True:
            position = self.buffer.find(MAGIC)
            if position < 0:
                if len(self.buffer) > len(MAGIC) - 1:
                    del self.buffer[:-(len(MAGIC) - 1)]
                break
            if position:
                del self.buffer[:position]
            if len(self.buffer) < HEADER.size:
                break
            magic, version, header_bytes, stream_id, sequence, timestamp_ns, payload_bytes, flags = HEADER.unpack_from(self.buffer)
            if magic != MAGIC or version != VERSION or header_bytes != HEADER.size or payload_bytes > PCM_FRAME_BYTES:
                del self.buffer[0]
                continue
            packet_bytes = header_bytes + payload_bytes
            if len(self.buffer) < packet_bytes:
                break
            payload = bytes(self.buffer[header_bytes:packet_bytes])
            del self.buffer[:packet_bytes]
            frames.append({
                "stream_id": stream_id, "sequence": sequence, "timestamp_ns": timestamp_ns,
                "payload": payload, "flags": flags,
            })
        return frames


class PlaybackSession:
    def __init__(self, output_dir: Path, frame: dict) -> None:
        self.output_dir = output_dir
        self.stream_id = frame["stream_id"]
        self.first_timestamp_ns = frame["timestamp_ns"]
        self.last_timestamp_ns = frame["timestamp_ns"]
        self.last_arrival = time.monotonic()
        self.expected_sequence = frame["sequence"]
        self.packet_count = 0
        self.dropped_frames = 0
        self.duplicate_frames = 0
        self.payload_bytes = 0
        moment = dt.datetime.fromtimestamp(self.first_timestamp_ns / 1_000_000_000, tz=dt.timezone.utc).astimezone()
        directory = output_dir / moment.strftime("%Y/%m/%d")
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{moment.strftime('%Y-%m-%d_%H-%M-%S_%f_%z')}-{self.stream_id:016x}"
        self.raw_path = directory / f".{stem}.s16le.part"
        self.flac_path = directory / f"{stem}.flac"
        self.stream = self.raw_path.open("xb")

    def append(self, frame: dict) -> None:
        sequence = frame["sequence"]
        if sequence < self.expected_sequence:
            self.duplicate_frames += 1
            return
        gap = sequence - self.expected_sequence
        if gap:
            if gap > MAX_GAP_FRAMES:
                raise ValueError("playback_sequence_discontinuity")
            self.stream.write(bytes(PCM_FRAME_BYTES * gap))
            self.payload_bytes += PCM_FRAME_BYTES * gap
            self.dropped_frames += gap
        self.stream.write(frame["payload"])
        self.payload_bytes += len(frame["payload"])
        self.packet_count += 1
        self.expected_sequence = sequence + 1
        self.last_timestamp_ns = max(self.last_timestamp_ns, frame["timestamp_ns"])
        self.last_arrival = time.monotonic()

    def finalize(self) -> dict | None:
        if self.stream.closed:
            return None
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.payload_bytes == 0:
            self.raw_path.unlink(missing_ok=True)
            return None
        temporary_flac = self.flac_path.with_name(f".{self.flac_path.name}.part")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-i", str(self.raw_path),
            "-c:a", "flac", "-compression_level", "8", "-f", "flac", str(temporary_flac),
        ], check=True)
        temporary_flac.replace(self.flac_path)
        self.raw_path.unlink()
        duration = self.payload_bytes / (SAMPLE_RATE * CHANNELS * SAMPLE_BYTES)
        occurred = dt.datetime.fromtimestamp(self.first_timestamp_ns / 1_000_000_000, tz=dt.timezone.utc)
        manifest = {
            "schema_version": 1,
            "kind": "oh2p_playback_reference",
            "stream_id": f"{self.stream_id:016x}",
            "occurred_at": occurred.isoformat(),
            "ended_at": (occurred + dt.timedelta(seconds=duration)).isoformat(),
            "device_last_timestamp_ns": self.last_timestamp_ns,
            "duration_seconds": round(duration, 6),
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_format": "s16le",
            "packets": self.packet_count,
            "dropped_frames_filled_with_silence": self.dropped_frames,
            "duplicate_frames_ignored": self.duplicate_frames,
            "audio_file": self.flac_path.name,
            "audio_sha256": hashlib.sha256(self.flac_path.read_bytes()).hexdigest(),
            "clock": "oh2p_clock_realtime",
        }
        atomic_json(self.flac_path.with_suffix(".json"), manifest)
        return manifest


class PlaybackRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.parser = FrameParser()
        self.sessions: dict[int, PlaybackSession] = {}
        self.running = True

    def consume(self, data: bytes) -> int:
        count = 0
        for frame in self.parser.feed(data):
            session = self.sessions.get(frame["stream_id"])
            if session is None:
                session = PlaybackSession(self.args.output_dir, frame)
                self.sessions[frame["stream_id"]] = session
            try:
                session.append(frame)
            except ValueError:
                session.finalize()
                session = PlaybackSession(self.args.output_dir, frame)
                self.sessions[frame["stream_id"]] = session
                session.append(frame)
            count += 1
        return count

    def finalize_inactive(self, force: bool = False) -> int:
        now = time.monotonic()
        completed = 0
        for stream_id, session in list(self.sessions.items()):
            if force or now - session.last_arrival >= self.args.inactivity_seconds:
                manifest = session.finalize()
                del self.sessions[stream_id]
                if manifest:
                    completed += 1
                    logging.info("已归档播放 reference：%s（%.3f 秒，丢帧 %d）", manifest["audio_file"], manifest["duration_seconds"], manifest["dropped_frames_filled_with_silence"])
        return completed

    def capture_once(self) -> None:
        if not mounted(self.args.required_mount):
            logging.error("播放 reference 存储未挂载：%s", self.args.required_mount)
            time.sleep(self.args.retry_seconds)
            return
        process = subprocess.Popen(build_ssh_command(self.args), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert process.stdout is not None
        try:
            while self.running and process.poll() is None:
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if ready:
                    data = os.read(process.stdout.fileno(), 64 * 1024)
                    if not data:
                        break
                    self.consume(data)
                self.finalize_inactive()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            self.finalize_inactive(force=True)

    def run(self) -> None:
        while self.running:
            self.capture_once()
            if self.running:
                time.sleep(self.args.retry_seconds)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    recorder = PlaybackRecorder(args)
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: setattr(recorder, "running", False))
    recorder.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
