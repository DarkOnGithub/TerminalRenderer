import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np


def probe_video_stream(path: str) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    width = int(out[0])
    height = int(out[1])
    rate_text = out[2]
    if "/" in rate_text:
        num, den = rate_text.split("/", 1)
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(rate_text)
    return width, height, fps


@dataclass
class DecodedFrame:
    frame_idx: int
    frame_np: np.ndarray


class LatestFrameReader:
    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        *,
        output_width: int | None = None,
        output_height: int | None = None,
        vf: str | None = None,
        realtime: bool = False,
        pause_after_first_frame: bool = False,
    ):
        self.width = width
        self.height = height
        self.output_width = int(output_width) if output_width is not None else width
        self.output_height = int(output_height) if output_height is not None else height
        self.frame_bytes = self.output_width * self.output_height * 3
        self.queue: queue.Queue[DecodedFrame | None] = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.pause_after_first_frame = pause_after_first_frame
        self.done = False
        self.error: Exception | None = None
        cmd = [
            "ffmpeg",
            "-v",
            "error",
        ]
        if realtime:
            cmd.append("-re")
        cmd.extend(["-i", path])
        if vf:
            cmd.extend(["-vf", vf])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _put_latest(self, item: DecodedFrame | None) -> None:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def _reader_loop(self) -> None:
        frame_idx = 0
        try:
            if self.proc.stdout is None:
                return
            while not self.stop_event.is_set():
                raw = self.proc.stdout.read(self.frame_bytes)
                if not raw or len(raw) < self.frame_bytes:
                    break
                frame_np = (
                    np.frombuffer(raw, dtype=np.uint8)
                    .reshape(self.output_height, self.output_width, 3)
                    .copy()
                )
                self._put_latest(DecodedFrame(frame_idx=frame_idx, frame_np=frame_np))
                frame_idx += 1
                if self.pause_after_first_frame and frame_idx == 1:
                    while not self.stop_event.is_set():
                        if self.resume_event.wait(timeout=0.05):
                            break
        except Exception as exc:
            self.error = exc
        finally:
            self.done = True
            self._put_latest(None)

    def get(self, timeout: float = 0.1) -> tuple[int, np.ndarray] | None:
        while True:
            if self.error is not None:
                raise self.error
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if self.done:
                    return None
                continue
            if item is None:
                return None
            return item.frame_idx, item.frame_np

    def resume(self) -> None:
        self.resume_event.set()

    def close(self) -> None:
        self.stop_event.set()
        self.resume_event.set()
        try:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            if self.proc.stderr is not None:
                self.proc.stderr.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.thread.join(timeout=1)


def should_drop_frame(
    now: float,
    start_time: float,
    playback_frame_idx: int,
    fps: float,
    audio_delay: float,
    max_frame_lag: float,
    lead_time: float = 0.0,
) -> bool:
    frame_interval = 1.0 / max(fps, 1e-6)
    target_time = start_time + (playback_frame_idx * frame_interval) + audio_delay
    allowed_lateness = max(0.0, max_frame_lag) * frame_interval
    return (now + max(0.0, lead_time)) > target_time + allowed_lateness


def playback_target_time(
    start_time: float,
    playback_frame_idx: int,
    fps: float,
    audio_delay: float,
) -> float:
    frame_interval = 1.0 / max(fps, 1e-6)
    return start_time + (max(0, int(playback_frame_idx)) * frame_interval) + audio_delay
