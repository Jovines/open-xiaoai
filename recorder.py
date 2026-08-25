#!/usr/bin/env python3
"""Continuously record an Open-XiaoAI speaker microphone over SSH."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CAPTURE_CHANNELS = 4
DEFAULT_ARCHIVE_CHANNELS = 3
DEFAULT_PCM_GAIN = 96.0
CAPTURE_FORMAT = "S32_LE"
CAPTURE_PCM = "noop"
MIN_VALID_FILE_SIZE = 1_024
CODEC_EXTENSIONS = {"opus": ".ogg", "flac": ".flac", "wav": ".wav"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="持续保存 Xiaomi 智能音箱 Pro（OH2P）的麦克风音频",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=env_int("RECORDER_SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
        help="采样率，默认 48000 Hz",
    )
    parser.add_argument(
        "--archive-channels",
        type=int,
        choices=[1, 3],
        default=env_int("RECORDER_ARCHIVE_CHANNELS", DEFAULT_ARCHIVE_CHANNELS),
        help="保存一路麦克风或三路有效阵列通道，默认 3",
    )
    parser.add_argument(
        "--pcm-gain",
        type=float,
        default=env_float("RECORDER_PCM_GAIN", DEFAULT_PCM_GAIN),
        help="A113 低位样本映射增益；默认 96，给突发声音保留约 8.5 dB 余量",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("RECORDER_HOST", "192.168.8.242"),
        help="音箱 IP（默认读取 RECORDER_HOST）",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("RECORDER_USER", "root"),
        help="SSH 用户名",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "RECORDER_OUTPUT_DIR",
                str(Path.home() / "recordings" / "open-xiaoai"),
            )
        ),
        help="录音根目录",
    )
    parser.add_argument(
        "--required-mount",
        type=Path,
        default=(
            Path(value)
            if (value := os.environ.get("RECORDER_REQUIRED_MOUNT"))
            else None
        ),
        help="必须存在的挂载点；NAS 掉线时暂停录音",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=env_int("RECORDER_SEGMENT_SECONDS", 60),
        help="临时分段时长，默认 60 秒",
    )
    parser.add_argument(
        "--codec",
        choices=sorted(CODEC_EXTENSIONS),
        default=os.environ.get("RECORDER_CODEC", "flac"),
        help="保存格式，默认 flac 无损",
    )
    parser.add_argument(
        "--bitrate",
        default=os.environ.get("RECORDER_BITRATE", "96k"),
        help="Opus 总比特率，默认三通道 96k",
    )
    parser.add_argument(
        "--retry-seconds",
        type=float,
        default=env_float("RECORDER_RETRY_SECONDS", 5.0),
        help="断线重试间隔",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=env_int("RECORDER_RETENTION_DAYS", 0),
        help="保留天数；0 表示永久保留",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=env_float("RECORDER_MIN_FREE_GB", 5.0),
        help="低于该磁盘余量时暂停录音",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="成功录制指定段数后退出；0 表示持续运行",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("RECORDER_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    if args.segment_seconds < 1:
        parser.error("--segment-seconds 必须大于 0")
    if args.sample_rate not in (16_000, 48_000):
        parser.error("--sample-rate 当前仅支持 16000 或 48000")
    if not 0 < args.pcm_gain <= 256:
        parser.error("--pcm-gain 必须大于 0 且不超过 256")
    if args.retry_seconds < 0:
        parser.error("--retry-seconds 不能小于 0")
    if args.retention_days < 0:
        parser.error("--retention-days 不能小于 0")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb 不能小于 0")
    if not re.fullmatch(r"\d+(?:\.\d+)?[kKmM]?", args.bitrate):
        parser.error("--bitrate 格式无效，例如 24k")
    return args


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    ssh = ["ssh"]
    if os.environ.get("SSHPASS"):
        ssh = ["sshpass", "-e", "ssh"]
    else:
        ssh.extend(["-o", "BatchMode=yes"])

    ssh.extend(
        [
            "-T",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "HostKeyAlgorithms=+ssh-rsa",
            f"{args.user}@{args.host}",
            (
                f"arecord --quiet -t raw -D {CAPTURE_PCM} "
                f"-f {CAPTURE_FORMAT} -r {args.sample_rate} "
                f"-c {DEFAULT_CAPTURE_CHANNELS}"
            ),
        ]
    )
    return ssh


def build_ffmpeg_command(
    args: argparse.Namespace,
    temporary_path: Path,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",
        "-f",
        "s32le",
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(DEFAULT_CAPTURE_CHANNELS),
        "-i",
        "pipe:0",
        # A113 PDM samples occupy the lower 24 bits of each S32 sample. A gain of
        # 256 maps them to full scale but clips nearby loudspeaker playback in
        # practice, so production defaults to 96 (about 8.5 dB headroom).
        "-af",
        (
            f"volume={args.pcm_gain:g},pan=mono|c0=c0"
            if args.archive_channels == 1
            else f"volume={args.pcm_gain:g},pan=3.0|FL=c0|FR=c1|FC=c2"
        ),
    ]
    if args.codec == "opus":
        command.extend(
            [
                "-c:a",
                "libopus",
                "-b:a",
                args.bitrate,
                "-vbr",
                "on",
                "-application",
                "audio",
                "-f",
                "ogg",
            ]
        )
    elif args.codec == "flac":
        command.extend(
            [
                "-c:a",
                "flac",
                "-sample_fmt",
                "s32",
                "-compression_level",
                "8",
                "-f",
                "flac",
            ]
        )
    else:
        command.extend(["-c:a", "pcm_s24le", "-f", "wav"])
    command.append(str(temporary_path))
    return command


def output_paths(
    output_dir: Path,
    codec: str,
    now: dt.datetime | None = None,
) -> tuple[Path, Path]:
    now = now or dt.datetime.now().astimezone()
    directory = output_dir / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
    directory.mkdir(parents=True, exist_ok=True)
    extension = CODEC_EXTENSIONS[codec]
    stem = now.strftime("%Y-%m-%d_%H-%M-%S_%z")
    final_path = directory / f"{stem}{extension}"
    index = 1
    while final_path.exists() or final_path.with_name(f".{final_path.name}.part").exists():
        final_path = directory / f"{stem}-{index}{extension}"
        index += 1
    temporary_path = final_path.with_name(f".{final_path.name}.part")
    return temporary_path, final_path


def cleanup_expired(output_dir: Path, retention_days: int, now: float | None = None) -> int:
    if retention_days <= 0 or not output_dir.exists():
        return 0
    cutoff = (now or time.time()) - retention_days * 86_400
    removed = 0
    for extension in CODEC_EXTENSIONS.values():
        for path in output_dir.rglob(f"*{extension}"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
    return removed


def require_programs(use_password: bool) -> None:
    required = ["ssh", "ffmpeg"]
    if use_password:
        required.append("sshpass")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"缺少依赖程序: {', '.join(missing)}")


class BufferedCapture:
    """Keep draining SSH audio while ffmpeg rotates output files."""

    _END = object()

    def __init__(self, stream: object, max_blocks: int = 128) -> None:
        self.stream = stream
        self.blocks: queue.Queue[bytes | object] = queue.Queue(maxsize=max_blocks)
        self.pending = b""
        self.thread = threading.Thread(target=self._pump, name="xiaoai-capture", daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        try:
            while True:
                block = self.stream.read(65_536)
                if not block:
                    break
                self.blocks.put(block)
        finally:
            self.blocks.put(self._END)

    def read(self, maximum: int) -> bytes:
        if not self.pending:
            block = self.blocks.get()
            if block is self._END:
                return b""
            assert isinstance(block, bytes)
            self.pending = block
        result = self.pending[:maximum]
        self.pending = self.pending[maximum:]
        return result


class ContinuousRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        self.active_ssh: subprocess.Popen[bytes] | None = None
        self.active_ffmpeg: subprocess.Popen[bytes] | None = None

    def request_stop(self, _signum: int | None = None, _frame: object | None = None) -> None:
        if not self.stop_requested:
            logging.info("收到停止信号，正在保存当前有效残段")
        self.stop_requested = True
        if self.active_ssh and self.active_ssh.poll() is None:
            self.active_ssh.terminate()

    def has_disk_space(self) -> bool:
        if self.args.required_mount and not self.args.required_mount.is_mount():
            logging.error("所需存储未挂载：%s", self.args.required_mount)
            return False
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.args.output_dir).free
        required = int(self.args.min_free_gb * 1024**3)
        if free >= required:
            return True
        logging.error(
            "磁盘余量不足：剩余 %.2f GiB，要求至少 %.2f GiB",
            free / 1024**3,
            self.args.min_free_gb,
        )
        return False

    @staticmethod
    def _read_error(stream: object) -> str:
        stream.seek(0)
        return stream.read().decode("utf-8", errors="replace").strip()

    def record_segment(self, source: BufferedCapture) -> tuple[bool, bool]:
        temporary_path, final_path = output_paths(self.args.output_dir, self.args.codec)
        ffmpeg_command = build_ffmpeg_command(self.args, temporary_path)
        logging.info("开始录音：%s", final_path)
        bytes_per_second = self.args.sample_rate * DEFAULT_CAPTURE_CHANNELS * 4
        remaining = self.args.segment_seconds * bytes_per_second
        captured = 0
        stream_alive = True

        with tempfile.TemporaryFile() as ffmpeg_error:
            try:
                self.active_ffmpeg = subprocess.Popen(
                    ffmpeg_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=ffmpeg_error,
                )
                assert self.active_ffmpeg.stdin is not None
                while remaining > 0:
                    chunk = source.read(min(65_536, remaining))
                    if not chunk:
                        stream_alive = False
                        break
                    self.active_ffmpeg.stdin.write(chunk)
                    captured += len(chunk)
                    remaining -= len(chunk)
                self.active_ffmpeg.stdin.close()
                ffmpeg_status = self.active_ffmpeg.wait()

                valid = (
                    ffmpeg_status == 0
                    and captured > bytes_per_second // 4
                    and temporary_path.exists()
                    and temporary_path.stat().st_size >= MIN_VALID_FILE_SIZE
                )
                if valid:
                    temporary_path.replace(final_path)
                    logging.info(
                        "录音已保存：%s（%.2f MiB）",
                        final_path,
                        final_path.stat().st_size / 1024**2,
                    )
                    if not stream_alive and not self.stop_requested:
                        logging.warning("音箱连接提前结束，已保存残段并将自动重连")
                    return True, stream_alive

                message = self._read_error(ffmpeg_error)
                logging.error(
                    "录音失败（ffmpeg=%s, captured=%d）%s",
                    ffmpeg_status,
                    captured,
                    f"；ffmpeg: {message}" if message else "",
                )
                temporary_path.unlink(missing_ok=True)
                return False, stream_alive
            except (OSError, BrokenPipeError) as error:
                logging.error("无法完成录音分段：%s", error)
                temporary_path.unlink(missing_ok=True)
                return False, False
            finally:
                self.active_ffmpeg = None

    def capture_connection(self, completed: int) -> tuple[int, bool]:
        connection_complete = False
        with tempfile.TemporaryFile() as ssh_error:
            try:
                self.active_ssh = subprocess.Popen(
                    build_ssh_command(self.args),
                    stdout=subprocess.PIPE,
                    stderr=ssh_error,
                )
                assert self.active_ssh.stdout is not None
                source = BufferedCapture(self.active_ssh.stdout)
                while not self.stop_requested:
                    saved, stream_alive = self.record_segment(source)
                    if saved:
                        completed += 1
                        if self.args.max_segments and completed >= self.args.max_segments:
                            connection_complete = True
                            return completed, True
                    if not stream_alive:
                        break
                return completed, self.stop_requested
            except OSError as error:
                logging.error("无法启动音箱采集：%s", error)
                return completed, False
            finally:
                if self.active_ssh:
                    if self.active_ssh.poll() is None:
                        self.active_ssh.terminate()
                    try:
                        status = self.active_ssh.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.active_ssh.kill()
                        status = self.active_ssh.wait(timeout=5)
                    if status and not self.stop_requested and not connection_complete:
                        message = self._read_error(ssh_error)
                        logging.error(
                            "音箱采集连接已断开（ssh=%s）%s",
                            status,
                            f"；SSH: {message}" if message else "",
                        )
                self.active_ssh = None

    def run(self) -> int:
        require_programs(bool(os.environ.get("SSHPASS")))
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        completed = 0

        while not self.stop_requested:
            removed = cleanup_expired(self.args.output_dir, self.args.retention_days)
            if removed:
                logging.info("已清理 %d 个过期录音文件", removed)
            if not self.has_disk_space():
                time.sleep(max(self.args.retry_seconds, 1.0))
                continue

            completed, finished = self.capture_connection(completed)
            if finished or (self.args.max_segments and completed >= self.args.max_segments):
                break
            if not self.stop_requested:
                time.sleep(self.args.retry_seconds)

        logging.info("录音服务已停止，共完成 %d 个文件", completed)
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return ContinuousRecorder(args).run()
    except RuntimeError as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
