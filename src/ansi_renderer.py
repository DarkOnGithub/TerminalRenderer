import queue
import os
import subprocess
import threading
import time
from typing import Any, Generator

import torch

from .ansi_generator import ansi_generate
from .config import Config, SYNC_OUTPUT_BEGIN, SYNC_OUTPUT_END
from .frame_processing import pre_process_frame
from .utils import setup_lookup


QUEUE_SIZE = 12
BUFFER_POOL_SIZE = 14
INITIAL_BUFFER_SIZE = 16 * 1024 * 1024
WRITE_CHUNK_SIZE = 2_097_152


class AnsiRenderer:
    def __init__(
        self,
        frame_generator: Generator[torch.Tensor, None, None],
        config: Config,
        autostart: bool = True,
    ):
        self.frame_generator = frame_generator
        self.config = config
        self.ansi_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.cuda_enabled = (
            self.config.device.type == "cuda" and torch.cuda.is_available()
        )
        self.copy_stream = None
        if self.cuda_enabled:
            self.copy_stream = torch.cuda.Stream(device=self.config.device)
        self._pending_copy_done_event = None
        self._has_writev = hasattr(os, "writev")
        self._sync_begin_view = memoryview(SYNC_OUTPUT_BEGIN)
        self._sync_end_view = memoryview(SYNC_OUTPUT_END)

        self.free_buffers = queue.Queue()
        for _ in range(BUFFER_POOL_SIZE):
            self.free_buffers.put(
                torch.empty(
                    INITIAL_BUFFER_SIZE,
                    dtype=torch.uint8,
                    pin_memory=self.cuda_enabled,
                )
            )

        self.lookup_vals, self.lookup_lens = setup_lookup(
            max(self.config.width + 1, self.config.height + 1, 256), self.config.device
        )
        self.generator_thread = threading.Thread(
            target=self._generator_thread, daemon=True
        )
        self.thread_crashed = threading.Event()
        self.thread_exception = None

        self.start_time = None
        self.rendered_frames = 0
        self.audio_process = None
        self._output_initialized = False

        if autostart:
            self.generator_thread.start()

    def build_frame_payload(
        self,
        previous_frame: torch.Tensor | None,
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        return self._build_frame_payload(previous_frame, frame)

    def _quality_settings(self) -> tuple[int, int, int]:
        return (
            int(self.config.quant_mask),
            int(self.config.diff_thresh),
            int(self.config.run_color_diff_thresh),
        )

    def _current_quality_settings(self) -> tuple[int, int, int]:
        return self._quality_settings()

    def _build_frame_payload(
        self,
        previous_frame: torch.Tensor | None,
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        old_shape = previous_frame.shape if previous_frame is not None else None

        quant_mask, diff_thresh, run_color_diff_thresh = self._quality_settings()

        xs, ys, colors_rgb, updated_previous = pre_process_frame(
            previous_frame,
            frame,
            self.config,
            quant_mask=quant_mask,
            diff_thresh_override=diff_thresh,
        )

        if xs.numel() == 0:
            return None, updated_previous

        ansi_gpu = ansi_generate(
            xs,
            ys,
            colors_rgb,
            self.lookup_vals,
            self.lookup_lens,
            self.config,
            run_color_diff_thresh_override=run_color_diff_thresh,
        )

        shape_changed = old_shape is not None and old_shape != updated_previous.shape
        if shape_changed:
            clear_seq = torch.tensor(
                list(b"\033[2J\033[H"),
                dtype=torch.uint8,
                device=self.config.device,
            )
            ansi_gpu = torch.cat([clear_seq, ansi_gpu])

        return ansi_gpu, updated_previous

    def _write_all(self, fd: int, view: memoryview, chunk_size: int) -> None:
        bounded_chunk = max(4096, int(chunk_size)) if chunk_size > 0 else 0
        while view:
            try:
                if bounded_chunk > 0 and view.nbytes > bounded_chunk:
                    written = os.write(fd, view[:bounded_chunk])
                else:
                    written = os.write(fd, view)
            except InterruptedError:
                continue

            if written <= 0:
                raise RuntimeError("Short write on terminal output")
            view = view[written:]

    def _writev_all(self, fd: int, segments: list[memoryview]) -> None:
        filtered = [segment for segment in segments if segment.nbytes > 0]
        if not filtered:
            return

        index = 0
        offset = 0
        while index < len(filtered):
            head = filtered[index]
            if offset:
                head = head[offset:]
            iovecs = [head, *filtered[index + 1 :]]

            try:
                written = os.writev(fd, iovecs)
            except InterruptedError:
                continue

            if written <= 0:
                raise RuntimeError("Short writev on terminal output")

            remaining = written
            head_len = head.nbytes
            if remaining < head_len:
                offset += remaining
                continue

            remaining -= head_len
            index += 1
            offset = 0

            while remaining > 0 and index < len(filtered):
                current_len = filtered[index].nbytes
                if remaining < current_len:
                    offset = remaining
                    remaining = 0
                else:
                    remaining -= current_len
                    index += 1

    def render_frame(self, frame: Any, frame_idx: int) -> None:
        if frame is None:
            return

        first_render = self.start_time is None
        if self.start_time is None:
            os.write(self.config.output_fd, b"\033[?1049h\033[2J\033[?25l\033[H")
            self._output_initialized = True

        pending_event = self._pending_copy_done_event
        if pending_event is not None:
            pending_event.synchronize()
            self._pending_copy_done_event = None

        if isinstance(frame, torch.Tensor):
            if frame.device.type != "cpu":
                data = bytes(frame.view(-1).tolist())
            else:
                data = memoryview(frame.numpy())
        else:
            if isinstance(frame, (bytes, bytearray, memoryview)):
                data = frame
            else:
                try:
                    data = memoryview(frame)
                except TypeError:
                    data = bytes(frame)

        view = memoryview(data)
        fd = self.config.output_fd
        sync_output = bool(self.config.sync_output)
        use_writev = sync_output and self._has_writev

        if use_writev:
            self._writev_all(fd, [self._sync_begin_view, view, self._sync_end_view])
        elif sync_output:
            self._write_all(fd, self._sync_begin_view, 0)
            try:
                self._write_all(fd, view, WRITE_CHUNK_SIZE)
            finally:
                self._write_all(fd, self._sync_end_view, 0)
        else:
            self._write_all(fd, view, WRITE_CHUNK_SIZE)

        if first_render:
            if self.config.audio_path:
                self.audio_process = subprocess.Popen(
                    [
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        "-vn",
                        "-sn",
                        self.config.audio_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self.start_time = time.perf_counter()

        self.rendered_frames += 1

    def get_next_ansi_sequence(self) -> Generator[tuple[Any, int], None, None]:
        while True:
            if self.thread_crashed.is_set():
                if self.thread_exception:
                    raise self.thread_exception
                else:
                    raise RuntimeError(
                        "Generator thread crashed without exception details"
                    )
            while True:
                try:
                    item = self.ansi_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    if self.thread_crashed.is_set():
                        if self.thread_exception:
                            raise self.thread_exception
                        raise RuntimeError(
                            "Generator thread crashed without exception details"
                        )
            if item is None:
                break

            ansi, frame_idx, copy_done_event, buffer_to_release = item
            self._pending_copy_done_event = copy_done_event
            yield ansi, frame_idx
            if (
                self._pending_copy_done_event is copy_done_event
                and copy_done_event is not None
            ):
                copy_done_event.synchronize()
            self._pending_copy_done_event = None
            self.free_buffers.put(buffer_to_release)

    def _generator_thread(self) -> None:
        try:
            previous_frame = None
            current_frame_idx = 0
            while True:
                # Catch up logic: if we are behind the wall clock, skip frames in the generator
                if self.start_time is not None:
                    elapsed = time.perf_counter() - self.start_time
                    # Aim to stay ~2 frames ahead for buffering, but skip if we fall behind
                    target_idx = int(
                        (elapsed - self.config.audio_delay) * self.config.fps
                    )

                    while current_frame_idx < target_idx:
                        frame = next(self.frame_generator, None)
                        if frame is None:
                            break
                        current_frame_idx += 1

                frame = next(self.frame_generator, None)

                if frame is None:
                    break

                ansi_gpu, previous_frame = self._build_frame_payload(
                    previous_frame, frame
                )

                if ansi_gpu is None:
                    current_frame_idx += 1
                    continue

                cpu_buf = self.free_buffers.get()

                if cpu_buf.size(0) < ansi_gpu.size(0):
                    cpu_buf = torch.empty(
                        int(ansi_gpu.size(0) * 1.2),
                        dtype=torch.uint8,
                        pin_memory=self.cuda_enabled,
                    )

                cpu_view = cpu_buf[: ansi_gpu.size(0)]

                copy_done_event = None
                if self.cuda_enabled:
                    if self.copy_stream is not None:
                        current_stream = torch.cuda.current_stream(
                            device=self.config.device
                        )
                        with torch.cuda.stream(self.copy_stream):
                            self.copy_stream.wait_stream(current_stream)
                            cpu_view.copy_(ansi_gpu, non_blocking=True)
                            copy_done_event = torch.cuda.Event()
                            copy_done_event.record(self.copy_stream)
                    else:
                        cpu_view.copy_(ansi_gpu, non_blocking=True)
                        copy_done_event = torch.cuda.Event()
                        copy_done_event.record(
                            torch.cuda.current_stream(device=self.config.device)
                        )
                else:
                    cpu_view.copy_(ansi_gpu, non_blocking=False)

                cpu_payload = cpu_view.numpy()

                self.ansi_queue.put(
                    (cpu_payload, current_frame_idx, copy_done_event, cpu_buf)
                )

                current_frame_idx += 1
            self.ansi_queue.put(None)
        except Exception as e:
            print(f"Error in generator thread: {e}")
            self.thread_exception = e
            self.thread_crashed.set()
            self.ansi_queue.put(None)
            raise

    def __del__(self):
        try:
            if hasattr(self, "audio_process") and self.audio_process:
                self.audio_process.terminate()
                self.audio_process.wait()

            if hasattr(self, "config") and getattr(self, "_output_initialized", False):
                os.write(self.config.output_fd, b"\033[?25h\033[?1049l")
        except Exception:
            pass
