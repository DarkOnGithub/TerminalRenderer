import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Generator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.ansi_renderer import AnsiRenderer, GpuBuildTiming
from src.ansi_generator import ansi_generate
from src.config import (
    CELL_ASPECT,
    CLEAR_SCREEN,
    DISABLE_ALT_BUFFER,
    ENABLE_ALT_BUFFER,
    HIDE_CURSOR,
    SHOW_CURSOR,
    SYNC_OUTPUT_BEGIN,
    SYNC_OUTPUT_END,
    Config,
)
from src.frame_processing import pre_process_frame
from src.video_playback import (
    LatestFrameReader,
    playback_target_time,
    probe_video_stream,
    should_drop_frame,
)

PANE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
INIT_SEQUENCE = ENABLE_ALT_BUFFER + CLEAR_SCREEN + HIDE_CURSOR + b"\033[H"
FINAL_SEQUENCE = SHOW_CURSOR + DISABLE_ALT_BUFFER
FPS_OVERLAY_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "/": ("001", "001", "010", "100", "100"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


class PaneDisconnectedError(RuntimeError):
    pass


@dataclass
class PaneSpec:
    pane_id: str
    fifo_path: str
    columns: int
    lines: int
    target_width: int
    target_height: int
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass
class PaneRuntime:
    spec: PaneSpec
    fd: int
    renderer: AnsiRenderer
    cell_bounds: tuple[int, int, int, int] | None = None
    previous_frame: torch.Tensor | None = None
    previous_frame_scratch: torch.Tensor | None = None
    build_stream: torch.cuda.Stream | None = None
    copy_stream: torch.cuda.Stream | None = None
    cpu_payload_buffer: torch.Tensor | None = None


@dataclass
class PanePayload:
    pane: PaneRuntime
    payload_ref: object | None
    payload_view: memoryview | None
    copy_done_event: torch.cuda.Event | None
    gpu_build_timing: GpuBuildTiming | None
    next_previous_frame: torch.Tensor


@dataclass
class SharedBuildRuntime:
    renderer: AnsiRenderer
    previous_frame: torch.Tensor | None = None
    previous_frame_scratch: torch.Tensor | None = None


@dataclass
class FlushStats:
    total_time: float
    per_pane_times: dict[str, float]


@dataclass
class RuntimeStats:
    pane_ids: tuple[str, ...]
    window_started_at: float
    presented: int = 0
    dropped: int = 0
    skipped_input_frames: int = 0
    fetch_time_sum: float = 0.0
    upload_time_sum: float = 0.0
    build_time_sum: float = 0.0
    gpu_build_time_sum: float = 0.0
    gpu_preprocess_time_sum: float = 0.0
    gpu_gen_time_sum: float = 0.0
    sleep_time_sum: float = 0.0
    flush_time_sum: float = 0.0
    lateness_sum: float = 0.0
    total_payload_bytes: int = 0
    pane_payload_bytes: dict[str, int] = field(default_factory=dict)
    pane_build_time_sum: dict[str, float] = field(default_factory=dict)
    pane_gpu_build_time_sum: dict[str, float] = field(default_factory=dict)
    pane_flush_time_sum: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pane_payload_bytes:
            self.pane_payload_bytes = {pane_id: 0 for pane_id in self.pane_ids}
        if not self.pane_build_time_sum:
            self.pane_build_time_sum = {pane_id: 0.0 for pane_id in self.pane_ids}
        if not self.pane_gpu_build_time_sum:
            self.pane_gpu_build_time_sum = {pane_id: 0.0 for pane_id in self.pane_ids}
        if not self.pane_flush_time_sum:
            self.pane_flush_time_sum = {pane_id: 0.0 for pane_id in self.pane_ids}

    def record(
        self,
        *,
        dropped: bool,
        skipped_input_frames: int,
        fetch_time: float,
        upload_time: float,
        build_time: float,
        gpu_build_time: float,
        gpu_preprocess_time: float,
        gpu_gen_time: float,
        pane_build_times: dict[str, float],
        pane_gpu_build_times: dict[str, float],
        sleep_time: float,
        flush_stats: FlushStats,
        lateness: float,
        payload_bytes: dict[str, int],
    ) -> None:
        if dropped:
            self.dropped += 1
        else:
            self.presented += 1

        self.skipped_input_frames += max(0, int(skipped_input_frames))
        self.fetch_time_sum += fetch_time
        self.upload_time_sum += upload_time
        self.build_time_sum += build_time
        self.gpu_build_time_sum += gpu_build_time
        self.gpu_preprocess_time_sum += gpu_preprocess_time
        self.gpu_gen_time_sum += gpu_gen_time
        self.sleep_time_sum += sleep_time
        self.flush_time_sum += flush_stats.total_time
        self.lateness_sum += lateness

        frame_total_bytes = 0
        for pane_id in self.pane_ids:
            pane_bytes = int(payload_bytes.get(pane_id, 0))
            frame_total_bytes += pane_bytes
            self.pane_payload_bytes[pane_id] += pane_bytes
            self.pane_build_time_sum[pane_id] += float(
                pane_build_times.get(pane_id, 0.0)
            )
            self.pane_gpu_build_time_sum[pane_id] += float(
                pane_gpu_build_times.get(pane_id, 0.0)
            )
            self.pane_flush_time_sum[pane_id] += float(
                flush_stats.per_pane_times.get(pane_id, 0.0)
            )
        self.total_payload_bytes += frame_total_bytes

    def reset_window(self, now: float) -> None:
        self.window_started_at = now
        self.presented = 0
        self.dropped = 0
        self.skipped_input_frames = 0
        self.fetch_time_sum = 0.0
        self.upload_time_sum = 0.0
        self.build_time_sum = 0.0
        self.gpu_build_time_sum = 0.0
        self.gpu_preprocess_time_sum = 0.0
        self.gpu_gen_time_sum = 0.0
        self.sleep_time_sum = 0.0
        self.flush_time_sum = 0.0
        self.lateness_sum = 0.0
        self.total_payload_bytes = 0
        for pane_id in self.pane_ids:
            self.pane_payload_bytes[pane_id] = 0
            self.pane_build_time_sum[pane_id] = 0.0
            self.pane_gpu_build_time_sum[pane_id] = 0.0
            self.pane_flush_time_sum[pane_id] = 0.0


_FLUSH_EXECUTOR: ThreadPoolExecutor | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one video across four tiled Alacritty terminals."
    )
    parser.add_argument("video_path", help="Path to the video file to play")
    parser.add_argument(
        "--session-dir",
        help="Reuse an existing launcher session directory or create one at this path",
    )
    parser.add_argument(
        "--launcher",
        default="./open_four_alacritty.sh",
        help="Launcher script used when a session needs to be created",
    )
    parser.add_argument(
        "--render-mode",
        choices=("pixel", "quadrant", "octant"),
        default="quadrant",
        help="ANSI render mode for each pane",
    )
    parser.add_argument(
        "--quadrant-cell-divisor",
        type=int,
        default=2,
        help="Cell divisor used for quadrant mode",
    )
    parser.add_argument(
        "--octant-cell-width-divisor",
        type=int,
        default=2,
        help="Cell width divisor used for octant mode",
    )
    parser.add_argument(
        "--octant-cell-height-divisor",
        type=int,
        default=4,
        help="Cell height divisor used for octant mode",
    )
    parser.add_argument(
        "--audio-delay",
        type=float,
        default=0.0,
        help="Additional delay applied to video frames relative to audio",
    )
    parser.add_argument(
        "--cell-aspect",
        type=float,
        default=CELL_ASPECT,
        help="Displayed width/height ratio of one terminal cell",
    )
    parser.add_argument(
        "--diff-thresh",
        type=int,
        default=8,
        help="Minimum per-channel cell color delta before a cell is treated as changed",
    )
    parser.add_argument(
        "--run-color-diff-thresh",
        type=int,
        default=8,
        help="Minimum per-channel run color delta before starting a new ANSI style run",
    )
    parser.add_argument(
        "--cursor-moves",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Cursor addressing mode for ANSI output; absolute is usually faster "
            "for multi-pane playback"
        ),
    )
    parser.add_argument(
        "--timing-file",
        help="Optional CSV path for per-frame timing data",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Print multi-pane runtime stats to stderr every N seconds; 0 disables it",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("pane", "global", "off"),
        default="pane",
        help="How synchronized-update wrappers are applied to pane writes",
    )
    parser.add_argument(
        "--max-frame-lag",
        type=float,
        default=1.0,
        help="Drop frames when video falls behind audio by more than this many frames",
    )
    return parser.parse_args(argv)


def require_cmd(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Missing required command: {name}")


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def compute_fit_geometry(
    orig_height: int,
    orig_width: int,
    target_height: int,
    target_width: int,
    cell_aspect: float = CELL_ASPECT,
) -> tuple[int, int, int, int]:
    effective_aspect = max(1e-6, float(cell_aspect))
    scale = min(
        (target_width * effective_aspect) / max(orig_width, 1),
        target_height / max(orig_height, 1),
    )
    new_width = max(
        1,
        min(target_width, int(round((orig_width * scale) / effective_aspect))),
    )
    new_height = max(1, min(target_height, int(round(orig_height * scale))))
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    return new_height, new_width, top, left


def build_ffmpeg_canvas_filter(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    cell_aspect: float = CELL_ASPECT,
) -> str | None:
    new_height, new_width, top, left = compute_fit_geometry(
        source_height,
        source_width,
        target_height,
        target_width,
        cell_aspect=cell_aspect,
    )
    if (
        new_width == source_width
        and new_height == source_height
        and target_width == source_width
        and target_height == source_height
        and top == 0
        and left == 0
    ):
        return None

    filters = [f"scale={new_width}:{new_height}:flags=fast_bilinear"]
    if new_width != target_width or new_height != target_height:
        filters.append(f"pad={target_width}:{target_height}:{left}:{top}:color=black")
    return ",".join(filters)


def wait_for_session_file(session_dir: str, timeout: float = 10.0) -> str:
    session_file = os.path.join(session_dir, "session.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(session_file) and os.path.getsize(session_file) > 0:
            return session_file
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for session metadata at {session_file}")


def launch_session(session_dir: str, launcher: str) -> str:
    launcher_path = resolve_path(launcher)
    subprocess.run([launcher_path, "--session-dir", session_dir], check=True)
    return wait_for_session_file(session_dir)


def load_session(session_file: str) -> dict:
    with open(session_file, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    panes = session.get("panes", [])
    pane_ids = {pane.get("id") for pane in panes}
    if pane_ids != set(PANE_ORDER):
        raise RuntimeError(
            f"Expected 4 panes {PANE_ORDER}, got {sorted(pane_ids)} from {session_file}"
        )
    return session


def pane_target_size(
    columns: int,
    lines: int,
    render_mode: str,
    quadrant_cell_divisor: int,
    octant_cell_width_divisor: int,
    octant_cell_height_divisor: int,
) -> tuple[int, int]:
    if render_mode == "pixel":
        return columns, lines
    if render_mode == "quadrant":
        factor = max(1, int(quadrant_cell_divisor))
        return columns * factor, lines * factor
    if render_mode == "octant":
        return (
            columns * max(1, int(octant_cell_width_divisor)),
            lines * max(1, int(octant_cell_height_divisor)),
        )
    raise ValueError(f"Unsupported render mode: {render_mode}")


def build_pane_specs(
    session: dict, args: argparse.Namespace
) -> tuple[list[PaneSpec], int, int]:
    panes_by_id = {pane["id"]: pane for pane in session["panes"]}

    top_left = panes_by_id["top_left"]
    top_right = panes_by_id["top_right"]
    bottom_left = panes_by_id["bottom_left"]
    bottom_right = panes_by_id["bottom_right"]

    tl_width, tl_height = pane_target_size(
        int(top_left["columns"]),
        int(top_left["lines"]),
        args.render_mode,
        args.quadrant_cell_divisor,
        args.octant_cell_width_divisor,
        args.octant_cell_height_divisor,
    )
    tr_width, tr_height = pane_target_size(
        int(top_right["columns"]),
        int(top_right["lines"]),
        args.render_mode,
        args.quadrant_cell_divisor,
        args.octant_cell_width_divisor,
        args.octant_cell_height_divisor,
    )
    bl_width, bl_height = pane_target_size(
        int(bottom_left["columns"]),
        int(bottom_left["lines"]),
        args.render_mode,
        args.quadrant_cell_divisor,
        args.octant_cell_width_divisor,
        args.octant_cell_height_divisor,
    )
    br_width, br_height = pane_target_size(
        int(bottom_right["columns"]),
        int(bottom_right["lines"]),
        args.render_mode,
        args.quadrant_cell_divisor,
        args.octant_cell_width_divisor,
        args.octant_cell_height_divisor,
    )

    left_width = tl_width
    right_width = tr_width
    top_height = tl_height
    bottom_height = bl_height

    if bl_width != left_width or br_width != right_width:
        raise RuntimeError("Pane column widths do not align across the 4-pane layout")
    if tr_height != top_height or br_height != bottom_height:
        raise RuntimeError("Pane row heights do not align across the 4-pane layout")

    specs = [
        PaneSpec(
            pane_id="top_left",
            fifo_path=top_left["fifo"],
            columns=int(top_left["columns"]),
            lines=int(top_left["lines"]),
            target_width=tl_width,
            target_height=tl_height,
            x0=0,
            x1=left_width,
            y0=0,
            y1=top_height,
        ),
        PaneSpec(
            pane_id="top_right",
            fifo_path=top_right["fifo"],
            columns=int(top_right["columns"]),
            lines=int(top_right["lines"]),
            target_width=tr_width,
            target_height=tr_height,
            x0=left_width,
            x1=left_width + right_width,
            y0=0,
            y1=top_height,
        ),
        PaneSpec(
            pane_id="bottom_left",
            fifo_path=bottom_left["fifo"],
            columns=int(bottom_left["columns"]),
            lines=int(bottom_left["lines"]),
            target_width=bl_width,
            target_height=bl_height,
            x0=0,
            x1=left_width,
            y0=top_height,
            y1=top_height + bottom_height,
        ),
        PaneSpec(
            pane_id="bottom_right",
            fifo_path=bottom_right["fifo"],
            columns=int(bottom_right["columns"]),
            lines=int(bottom_right["lines"]),
            target_width=br_width,
            target_height=br_height,
            x0=left_width,
            x1=left_width + right_width,
            y0=top_height,
            y1=top_height + bottom_height,
        ),
    ]
    return specs, left_width + right_width, top_height + bottom_height


def fit_frame_to_canvas(
    frame: torch.Tensor,
    target_height: int,
    target_width: int,
    cell_aspect: float = CELL_ASPECT,
) -> torch.Tensor:
    orig_height, orig_width = frame.shape[:2]
    new_height, new_width, top, left = compute_fit_geometry(
        orig_height,
        orig_width,
        target_height,
        target_width,
        cell_aspect=cell_aspect,
    )

    if (
        orig_height == target_height
        and orig_width == target_width
        and new_height == target_height
        and new_width == target_width
        and top == 0
        and left == 0
    ):
        return frame

    if new_width == orig_width and new_height == orig_height:
        resized = frame
    else:
        resized = (
            F.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0).float(),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .to(frame.dtype)
        )

    canvas = torch.zeros(
        (target_height, target_width, 3), dtype=frame.dtype, device=frame.device
    )
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def ensure_canvas_frame(
    frame: torch.Tensor,
    target_height: int,
    target_width: int,
    cell_aspect: float = CELL_ASPECT,
) -> torch.Tensor:
    if frame.shape[0] == target_height and frame.shape[1] == target_width:
        return frame
    return fit_frame_to_canvas(
        frame,
        target_height,
        target_width,
        cell_aspect=cell_aspect,
    )


def build_renderer(
    spec: PaneSpec, args: argparse.Namespace, fps: float, device: torch.device
) -> AnsiRenderer:
    return build_configured_renderer(
        spec.target_width,
        spec.target_height,
        args,
        fps,
        device,
    )


def build_configured_renderer(
    width: int,
    height: int,
    args: argparse.Namespace,
    fps: float,
    device: torch.device,
) -> AnsiRenderer:
    def empty_frame_generator() -> Generator[torch.Tensor, None, None]:
        if False:
            yield torch.empty((0, 0, 3), dtype=torch.uint8, device=device)

    config = Config(
        width=width,
        height=height,
        device=device,
        fps=fps,
        render_mode=str(args.render_mode),
        quadrant_cell_divisor=int(args.quadrant_cell_divisor),
        octant_cell_width_divisor=int(args.octant_cell_width_divisor),
        octant_cell_height_divisor=int(args.octant_cell_height_divisor),
        quant_mask=0xFF,
        diff_thresh=int(args.diff_thresh),
        run_color_diff_thresh=int(args.run_color_diff_thresh),
        adaptive_quality=False,
        adaptive_quant_masks=(0xFF,),
        adaptive_diff_thresh_offsets=(0,),
        adaptive_run_color_diff_offsets=(0,),
        adaptive_ema_alpha=0.12,
        target_frame_bytes=0,
        frame_byte_buffer_frames=8,
        max_frame_bytes=0,
        relative_cursor_moves=str(args.cursor_moves).lower() == "relative",
        use_rep=True,
        rep_min_run=12,
        sync_output=False,
        prefer_writev=True,
        write_chunk_size=2_097_152,
        queue_size=12,
        buffer_pool_size=14,
        initial_buffer_size=16 * 1024 * 1024,
        async_copy_stream=True,
        pacing_render_lead=True,
        pacing_render_alpha=0.18,
        timing_enabled=False,
        timing_file="timing.csv",
    )
    return AnsiRenderer(empty_frame_generator(), config, autostart=False)


def build_shared_runtime(
    total_width: int,
    total_height: int,
    args: argparse.Namespace,
    fps: float,
    device: torch.device,
) -> SharedBuildRuntime:
    return SharedBuildRuntime(
        renderer=build_configured_renderer(total_width, total_height, args, fps, device)
    )


def write_all(fd: int, data: memoryview | bytes, chunk_size: int = 0) -> None:
    view = data if isinstance(data, memoryview) else memoryview(data)
    bounded_chunk = max(4096, int(chunk_size)) if chunk_size > 0 else 0
    while view:
        try:
            if bounded_chunk > 0 and view.nbytes > bounded_chunk:
                written = os.write(fd, view[:bounded_chunk])
            else:
                written = os.write(fd, view)
        except BrokenPipeError as exc:
            raise PaneDisconnectedError("pane fifo closed") from exc
        if written <= 0:
            raise RuntimeError("Short write to pane fifo")
        view = view[written:]


def writev_all(fd: int, buffers: list[memoryview | bytes]) -> None:
    views = [buf if isinstance(buf, memoryview) else memoryview(buf) for buf in buffers]
    index = 0
    offset = 0

    while index < len(views):
        current = views[index:]
        if offset:
            current = [current[0][offset:]] + current[1:]

        try:
            written = os.writev(fd, current)
        except BrokenPipeError as exc:
            raise PaneDisconnectedError("pane fifo closed") from exc
        if written <= 0:
            raise RuntimeError("Short writev to pane fifo")

        remaining = written
        head_len = views[index].nbytes - offset
        if remaining < head_len:
            offset += remaining
            continue

        remaining -= head_len
        index += 1
        offset = 0

        while remaining > 0 and index < len(views):
            current_len = views[index].nbytes
            if remaining < current_len:
                offset = remaining
                remaining = 0
            else:
                remaining -= current_len
                index += 1


def _get_flush_executor() -> ThreadPoolExecutor:
    global _FLUSH_EXECUTOR
    if _FLUSH_EXECUTOR is None:
        _FLUSH_EXECUTOR = ThreadPoolExecutor(max_workers=len(PANE_ORDER))
    return _FLUSH_EXECUTOR


def clone_previous_frame(pane: PaneRuntime) -> torch.Tensor | None:
    previous_frame = pane.previous_frame
    if previous_frame is None:
        return None

    scratch = pane.previous_frame_scratch
    if (
        scratch is None
        or scratch.shape != previous_frame.shape
        or scratch.dtype != previous_frame.dtype
        or scratch.device != previous_frame.device
    ):
        scratch = previous_frame.clone()
        pane.previous_frame_scratch = scratch
    else:
        scratch.copy_(previous_frame)
    return scratch


def pane_cell_bounds(
    spec: PaneSpec,
    render_mode: str,
    quadrant_cell_divisor: int,
    octant_cell_width_divisor: int,
    octant_cell_height_divisor: int,
) -> tuple[int, int, int, int]:
    if render_mode == "pixel":
        return spec.x0, spec.x1, spec.y0, spec.y1
    if render_mode == "quadrant":
        divisor = max(1, int(quadrant_cell_divisor))
        return (
            spec.x0 // divisor,
            spec.x1 // divisor,
            spec.y0 // divisor,
            spec.y1 // divisor,
        )
    if render_mode == "octant":
        x_divisor = max(1, int(octant_cell_width_divisor))
        y_divisor = max(1, int(octant_cell_height_divisor))
        return (
            spec.x0 // x_divisor,
            spec.x1 // x_divisor,
            spec.y0 // y_divisor,
            spec.y1 // y_divisor,
        )
    raise ValueError(f"Unsupported render mode: {render_mode}")


def _new_gpu_segment() -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
    if not torch.cuda.is_available():
        return None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    return start_event, end_event


def tensor_payload_view(
    pane: PaneRuntime,
    payload: torch.Tensor,
) -> tuple[object, memoryview, torch.cuda.Event | None]:
    if payload.device.type == "cpu":
        return payload, memoryview(payload.numpy()), None

    buffer_size = int(payload.numel())
    cpu_payload_buffer = pane.cpu_payload_buffer
    if (
        cpu_payload_buffer is None
        or cpu_payload_buffer.dtype != payload.dtype
        or cpu_payload_buffer.numel() < buffer_size
    ):
        cpu_payload_buffer = torch.empty(
            max(buffer_size, int(buffer_size * 1.2)),
            dtype=payload.dtype,
            pin_memory=True,
        )
        pane.cpu_payload_buffer = cpu_payload_buffer

    cpu_payload_view = cpu_payload_buffer[:buffer_size]
    copy_stream = pane.copy_stream
    copy_done_event = None
    if copy_stream is not None:
        current_stream = torch.cuda.current_stream(device=payload.device)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            cpu_payload_view.copy_(payload, non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(copy_stream)
    else:
        cpu_payload_view.copy_(payload, non_blocking=False)

    return cpu_payload_view, memoryview(cpu_payload_view.numpy()), copy_done_event


def payload_bytes_by_pane(payloads: list[PanePayload]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for payload in payloads:
        totals[payload.pane.spec.pane_id] = (
            0 if payload.payload_view is None else int(payload.payload_view.nbytes)
        )
    return totals


def gpu_build_times_by_pane(payloads: list[PanePayload]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for payload in payloads:
        pane_id = payload.pane.spec.pane_id
        timing = payload.gpu_build_timing
        if timing is None:
            totals[pane_id] = 0.0
            continue
        if payload.copy_done_event is None:
            timing.synchronize()
        totals[pane_id] = timing.total_ms() / 1000.0
    return totals


def gpu_gen_times_by_pane(payloads: list[PanePayload]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for payload in payloads:
        pane_id = payload.pane.spec.pane_id
        timing = payload.gpu_build_timing
        if timing is None:
            totals[pane_id] = 0.0
            continue
        if payload.copy_done_event is None:
            timing.synchronize()
        totals[pane_id] = timing.gen_ms() / 1000.0
    return totals


def shared_gpu_build_time(timing: GpuBuildTiming | None) -> float:
    if timing is None:
        return 0.0
    timing.synchronize()
    return timing.total_ms() / 1000.0


def shared_gpu_preprocess_time(timing: GpuBuildTiming | None) -> float:
    if timing is None:
        return 0.0
    timing.synchronize()
    return timing.preprocess_ms() / 1000.0


def build_pane_payloads_with_stats(
    panes: list[PaneRuntime],
    canvas: torch.Tensor,
) -> tuple[list[PanePayload], dict[str, float], dict[str, GpuBuildTiming | None]]:
    payloads: list[PanePayload] = []
    pane_build_times = {pane.spec.pane_id: 0.0 for pane in panes}
    pane_gpu_build_timings: dict[str, GpuBuildTiming | None] = {
        pane.spec.pane_id: None for pane in panes
    }
    for pane in panes:
        pane_build_start = time.perf_counter()
        crop = canvas[pane.spec.y0 : pane.spec.y1, pane.spec.x0 : pane.spec.x1]
        previous_frame = clone_previous_frame(pane)
        payload_result = pane.renderer.build_frame_payload(previous_frame, crop)
        payload, updated_previous, *payload_meta = payload_result
        gpu_build_timing = None
        if payload_meta and isinstance(payload_meta[-1], GpuBuildTiming):
            gpu_build_timing = payload_meta[-1]
        pane_gpu_build_timings[pane.spec.pane_id] = gpu_build_timing
        if payload is None:
            payloads.append(
                PanePayload(
                    pane=pane,
                    payload_ref=None,
                    payload_view=None,
                    copy_done_event=None,
                    gpu_build_timing=gpu_build_timing,
                    next_previous_frame=updated_previous,
                )
            )
            pane_build_times[pane.spec.pane_id] += (
                time.perf_counter() - pane_build_start
            )
            continue
        payload_ref, payload_view, copy_done_event = tensor_payload_view(pane, payload)
        payloads.append(
            PanePayload(
                pane=pane,
                payload_ref=payload_ref,
                payload_view=payload_view,
                copy_done_event=copy_done_event,
                gpu_build_timing=gpu_build_timing,
                next_previous_frame=updated_previous,
            )
        )
        pane_build_times[pane.spec.pane_id] += time.perf_counter() - pane_build_start
    return payloads, pane_build_times, pane_gpu_build_timings


def build_shared_pane_payloads_with_stats(
    panes: list[PaneRuntime],
    runtime: SharedBuildRuntime,
    canvas: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[PanePayload], dict[str, float], GpuBuildTiming | None, torch.Tensor]:
    config = runtime.renderer.config
    previous_frame = runtime.previous_frame

    preprocess_gpu_timing = GpuBuildTiming() if runtime.renderer.cuda_enabled else None
    preprocess_segment = (
        _new_gpu_segment() if preprocess_gpu_timing is not None else None
    )
    if preprocess_segment is not None:
        preprocess_segment[0].record(torch.cuda.current_stream(device=config.device))
    xs, ys, colors_rgb, updated_previous = pre_process_frame(
        previous_frame,
        canvas,
        config,
        quant_mask=int(config.quant_mask),
        diff_thresh_override=int(config.diff_thresh),
    )
    if preprocess_segment is not None and preprocess_gpu_timing is not None:
        preprocess_segment[1].record(torch.cuda.current_stream(device=config.device))
        preprocess_gpu_timing.preprocess_segments.append(preprocess_segment)

    payloads: list[PanePayload] = []
    pane_build_times = {pane.spec.pane_id: 0.0 for pane in panes}
    pane_bounds = {
        pane.spec.pane_id: pane.cell_bounds
        if pane.cell_bounds is not None
        else pane_cell_bounds(
            pane.spec,
            str(args.render_mode).lower(),
            int(args.quadrant_cell_divisor),
            int(args.octant_cell_width_divisor),
            int(args.octant_cell_height_divisor),
        )
        for pane in panes
    }
    pane_selectors: dict[str, torch.Tensor] | None = None
    x_edges = sorted(
        {bound[0] for bound in pane_bounds.values()}
        | {bound[1] for bound in pane_bounds.values()}
    )
    y_edges = sorted(
        {bound[2] for bound in pane_bounds.values()}
        | {bound[3] for bound in pane_bounds.values()}
    )
    if len(x_edges) == 3 and len(y_edges) == 3 and xs.numel() > 0:
        pane_index = (xs >= x_edges[1]).to(torch.int64)
        pane_index = pane_index + ((ys >= y_edges[1]).to(torch.int64) * 2)
        pane_selectors = {
            pane_id: torch.where(pane_index == pane_idx)[0]
            for pane_idx, pane_id in enumerate(PANE_ORDER)
        }

    for pane in panes:
        pane_build_start = time.perf_counter()
        pane_id = pane.spec.pane_id
        x0, x1, y0, y1 = pane_bounds[pane_id]
        next_previous_frame = updated_previous[y0:y1, x0:x1]
        pane_gpu_timing = GpuBuildTiming() if runtime.renderer.cuda_enabled else None

        if xs.numel() == 0:
            payloads.append(
                PanePayload(
                    pane=pane,
                    payload_ref=None,
                    payload_view=None,
                    copy_done_event=None,
                    gpu_build_timing=pane_gpu_timing,
                    next_previous_frame=next_previous_frame,
                )
            )
            pane_build_times[pane.spec.pane_id] += (
                time.perf_counter() - pane_build_start
            )
            continue

        pane_selector = None if pane_selectors is None else pane_selectors.get(pane_id)
        if pane_selector is None:
            pane_selector = torch.where(
                (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            )[0]

        if pane_selector.numel() == 0:
            payloads.append(
                PanePayload(
                    pane=pane,
                    payload_ref=None,
                    payload_view=None,
                    copy_done_event=None,
                    gpu_build_timing=pane_gpu_timing,
                    next_previous_frame=next_previous_frame,
                )
            )
            pane_build_times[pane.spec.pane_id] += (
                time.perf_counter() - pane_build_start
            )
            continue

        pane_xs = xs.index_select(0, pane_selector) - x0
        pane_ys = ys.index_select(0, pane_selector) - y0
        pane_colors = colors_rgb.index_select(0, pane_selector)

        build_stream = pane.build_stream
        if build_stream is not None:
            current_stream = torch.cuda.current_stream(device=config.device)
            with torch.cuda.stream(build_stream):
                build_stream.wait_stream(current_stream)
                gen_segment = (
                    _new_gpu_segment() if pane_gpu_timing is not None else None
                )
                if gen_segment is not None:
                    gen_segment[0].record(build_stream)
                payload = ansi_generate(
                    pane_xs,
                    pane_ys,
                    pane_colors,
                    pane.renderer.lookup_vals,
                    pane.renderer.lookup_lens,
                    pane.renderer.config,
                    run_color_diff_thresh_override=int(
                        pane.renderer.config.run_color_diff_thresh
                    ),
                )
                if gen_segment is not None and pane_gpu_timing is not None:
                    gen_segment[1].record(build_stream)
                    pane_gpu_timing.gen_segments.append(gen_segment)

                old_shape = (
                    pane.previous_frame.shape
                    if pane.previous_frame is not None
                    else None
                )
                shape_changed = (
                    old_shape is not None and old_shape != next_previous_frame.shape
                )
                if shape_changed:
                    clear_seq = torch.tensor(
                        list(b"\033[2J\033[H"),
                        dtype=torch.uint8,
                        device=config.device,
                    )
                    payload = torch.cat([clear_seq, payload])

                payload_ref, payload_view, copy_done_event = tensor_payload_view(
                    pane, payload
                )
        else:
            gen_segment = _new_gpu_segment() if pane_gpu_timing is not None else None
            if gen_segment is not None:
                gen_segment[0].record(torch.cuda.current_stream(device=config.device))
            payload = ansi_generate(
                pane_xs,
                pane_ys,
                pane_colors,
                pane.renderer.lookup_vals,
                pane.renderer.lookup_lens,
                pane.renderer.config,
                run_color_diff_thresh_override=int(
                    pane.renderer.config.run_color_diff_thresh
                ),
            )
            if gen_segment is not None and pane_gpu_timing is not None:
                gen_segment[1].record(torch.cuda.current_stream(device=config.device))
                pane_gpu_timing.gen_segments.append(gen_segment)

            old_shape = (
                pane.previous_frame.shape if pane.previous_frame is not None else None
            )
            shape_changed = (
                old_shape is not None and old_shape != next_previous_frame.shape
            )
            if shape_changed:
                clear_seq = torch.tensor(
                    list(b"\033[2J\033[H"),
                    dtype=torch.uint8,
                    device=config.device,
                )
                payload = torch.cat([clear_seq, payload])

            payload_ref, payload_view, copy_done_event = tensor_payload_view(
                pane, payload
            )
        payloads.append(
            PanePayload(
                pane=pane,
                payload_ref=payload_ref,
                payload_view=payload_view,
                copy_done_event=copy_done_event,
                gpu_build_timing=pane_gpu_timing,
                next_previous_frame=next_previous_frame,
            )
        )
        pane_build_times[pane.spec.pane_id] += time.perf_counter() - pane_build_start

    return payloads, pane_build_times, preprocess_gpu_timing, updated_previous


def commit_pane_payloads(payloads: list[PanePayload]) -> None:
    for payload in payloads:
        payload.pane.previous_frame = payload.next_previous_frame


def flush_pane_payloads(
    panes: list[PaneRuntime],
    payloads: list[PanePayload],
    sync_mode: str,
    shared_runtime: SharedBuildRuntime | None = None,
    shared_next_previous_frame: torch.Tensor | None = None,
) -> FlushStats:
    flush_start = time.perf_counter()
    per_pane_times = {pane.spec.pane_id: 0.0 for pane in panes}
    if sync_mode == "global":
        for pane in panes:
            pane_start = time.perf_counter()
            write_all(pane.fd, SYNC_OUTPUT_BEGIN)
            per_pane_times[pane.spec.pane_id] += time.perf_counter() - pane_start
        for payload in payloads:
            if payload.payload_view is not None:
                pane_start = time.perf_counter()
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                write_all(payload.pane.fd, payload.payload_view)
                per_pane_times[payload.pane.spec.pane_id] += (
                    time.perf_counter() - pane_start
                )
        for pane in panes:
            pane_start = time.perf_counter()
            write_all(pane.fd, SYNC_OUTPUT_END)
            per_pane_times[pane.spec.pane_id] += time.perf_counter() - pane_start
    elif sync_mode == "pane":
        active_payloads = [
            payload for payload in payloads if payload.payload_view is not None
        ]

        if len(active_payloads) > 1:

            def flush_one(payload: PanePayload) -> tuple[str, float]:
                pane_start = time.perf_counter()
                payload_view = payload.payload_view
                if payload_view is None:
                    return payload.pane.spec.pane_id, 0.0
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                try:
                    writev_all(
                        payload.pane.fd,
                        [SYNC_OUTPUT_BEGIN, payload_view, SYNC_OUTPUT_END],
                    )
                except PaneDisconnectedError as exc:
                    raise PaneDisconnectedError(
                        f"Pane '{payload.pane.spec.pane_id}' disconnected"
                    ) from exc
                return payload.pane.spec.pane_id, time.perf_counter() - pane_start

            for pane_id, pane_time in _get_flush_executor().map(
                flush_one, active_payloads
            ):
                per_pane_times[pane_id] += pane_time
        else:
            for payload in active_payloads:
                pane_start = time.perf_counter()
                payload_view = payload.payload_view
                if payload_view is None:
                    continue
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                try:
                    writev_all(
                        payload.pane.fd,
                        [SYNC_OUTPUT_BEGIN, payload_view, SYNC_OUTPUT_END],
                    )
                except PaneDisconnectedError as exc:
                    raise PaneDisconnectedError(
                        f"Pane '{payload.pane.spec.pane_id}' disconnected"
                    ) from exc
                per_pane_times[payload.pane.spec.pane_id] += (
                    time.perf_counter() - pane_start
                )
    elif sync_mode == "off":
        active_payloads = [
            payload for payload in payloads if payload.payload_view is not None
        ]

        if len(active_payloads) > 1:

            def flush_one(payload: PanePayload) -> tuple[str, float]:
                pane_start = time.perf_counter()
                payload_view = payload.payload_view
                if payload_view is None:
                    return payload.pane.spec.pane_id, 0.0
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                try:
                    write_all(payload.pane.fd, payload_view)
                except PaneDisconnectedError as exc:
                    raise PaneDisconnectedError(
                        f"Pane '{payload.pane.spec.pane_id}' disconnected"
                    ) from exc
                return payload.pane.spec.pane_id, time.perf_counter() - pane_start

            for pane_id, pane_time in _get_flush_executor().map(
                flush_one, active_payloads
            ):
                per_pane_times[pane_id] += pane_time
        else:
            for payload in active_payloads:
                pane_start = time.perf_counter()
                payload_view = payload.payload_view
                if payload_view is None:
                    continue
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                try:
                    write_all(payload.pane.fd, payload_view)
                except PaneDisconnectedError as exc:
                    raise PaneDisconnectedError(
                        f"Pane '{payload.pane.spec.pane_id}' disconnected"
                    ) from exc
                per_pane_times[payload.pane.spec.pane_id] += (
                    time.perf_counter() - pane_start
                )
    else:
        raise ValueError(f"Unsupported sync mode: {sync_mode}")

    commit_pane_payloads(payloads)
    if shared_runtime is not None and shared_next_previous_frame is not None:
        shared_runtime.previous_frame = shared_next_previous_frame
    return FlushStats(
        total_time=time.perf_counter() - flush_start,
        per_pane_times=per_pane_times,
    )


def timing_csv_header(pane_ids: tuple[str, ...]) -> str:
    columns = [
        "frame_idx",
        "dropped",
        "skipped_input_frames",
        "fetch_time",
        "upload_time",
        "build_time",
        "gpu_build_time",
        "render_sleep",
        "flush_time",
        "lateness",
        "total_payload_bytes",
    ]
    columns.extend(f"{pane_id}_build_time" for pane_id in pane_ids)
    columns.extend(f"{pane_id}_gpu_build_time" for pane_id in pane_ids)
    columns.extend(f"{pane_id}_bytes" for pane_id in pane_ids)
    columns.extend(f"{pane_id}_flush_time" for pane_id in pane_ids)
    return ",".join(columns) + "\n"


def timing_csv_row(
    frame_idx: int,
    dropped: bool,
    skipped_input_frames: int,
    fetch_time: float,
    upload_time: float,
    build_time: float,
    gpu_build_time: float,
    pane_build_times: dict[str, float],
    pane_gpu_build_times: dict[str, float],
    sleep_time: float,
    flush_stats: FlushStats,
    lateness: float,
    payload_bytes: dict[str, int],
    pane_ids: tuple[str, ...],
) -> str:
    total_payload_bytes = sum(
        int(payload_bytes.get(pane_id, 0)) for pane_id in pane_ids
    )
    values: list[str] = [
        str(frame_idx),
        "1" if dropped else "0",
        str(max(0, int(skipped_input_frames))),
        f"{fetch_time:.6f}",
        f"{upload_time:.6f}",
        f"{build_time:.6f}",
        f"{gpu_build_time:.6f}",
        f"{sleep_time:.6f}",
        f"{flush_stats.total_time:.6f}",
        f"{lateness:.6f}",
        str(total_payload_bytes),
    ]
    values.extend(
        f"{float(pane_build_times.get(pane_id, 0.0)):.6f}" for pane_id in pane_ids
    )
    values.extend(
        f"{float(pane_gpu_build_times.get(pane_id, 0.0)):.6f}" for pane_id in pane_ids
    )
    values.extend(str(int(payload_bytes.get(pane_id, 0))) for pane_id in pane_ids)
    values.extend(
        f"{float(flush_stats.per_pane_times.get(pane_id, 0.0)):.6f}"
        for pane_id in pane_ids
    )
    return ",".join(values) + "\n"


def emit_runtime_stats(
    stats: RuntimeStats,
    now: float,
    interval: float,
    fps: float,
) -> None:
    if interval <= 0 or now - stats.window_started_at < interval:
        return

    elapsed = max(now - stats.window_started_at, 1e-6)
    shown_fps = stats.presented / elapsed
    dropped_fps = stats.dropped / elapsed
    skipped_input_fps = stats.skipped_input_frames / elapsed
    avg_fetch_ms = (
        1000.0 * stats.fetch_time_sum / max(stats.presented + stats.dropped, 1)
    )
    avg_upload_ms = 1000.0 * stats.upload_time_sum / max(stats.presented, 1)
    avg_build_ms = (
        1000.0 * stats.build_time_sum / max(stats.presented + stats.dropped, 1)
    )
    avg_gpu_build_ms = 1000.0 * stats.gpu_build_time_sum / max(stats.presented, 1)
    avg_gpu_preprocess_ms = (
        1000.0 * stats.gpu_preprocess_time_sum / max(stats.presented, 1)
    )
    avg_gpu_gen_ms = 1000.0 * stats.gpu_gen_time_sum / max(stats.presented, 1)
    avg_flush_ms = 1000.0 * stats.flush_time_sum / max(stats.presented, 1)
    avg_lateness_ms = 1000.0 * stats.lateness_sum / max(stats.presented, 1)
    avg_bytes = stats.total_payload_bytes / max(stats.presented, 1)
    slowest_build_pane = max(
        stats.pane_ids,
        key=lambda pane_id: stats.pane_build_time_sum.get(pane_id, 0.0),
    )
    slowest_build_ms = (
        1000.0
        * stats.pane_build_time_sum.get(slowest_build_pane, 0.0)
        / max(stats.presented, 1)
    )
    slowest_gpu_build_pane = max(
        stats.pane_ids,
        key=lambda pane_id: stats.pane_gpu_build_time_sum.get(pane_id, 0.0),
    )
    slowest_gpu_build_ms = (
        1000.0
        * stats.pane_gpu_build_time_sum.get(slowest_gpu_build_pane, 0.0)
        / max(stats.presented, 1)
    )
    slowest_pane = max(
        stats.pane_ids,
        key=lambda pane_id: stats.pane_flush_time_sum.get(pane_id, 0.0),
    )
    slowest_ms = (
        1000.0
        * stats.pane_flush_time_sum.get(slowest_pane, 0.0)
        / max(stats.presented, 1)
    )
    pane_bytes_text = " ".join(
        f"{pane_id}={stats.pane_payload_bytes.get(pane_id, 0) // max(stats.presented, 1)}"
        for pane_id in stats.pane_ids
    )
    print(
        (
            f"stats shown_fps={shown_fps:.1f}/{fps:.1f} dropped_fps={dropped_fps:.1f} "
            f"skipped_input_fps={skipped_input_fps:.1f} avg_fetch_ms={avg_fetch_ms:.1f} "
            f"avg_upload_ms={avg_upload_ms:.1f} avg_build_ms={avg_build_ms:.1f} avg_gpu_build_ms={avg_gpu_build_ms:.1f} "
            f"avg_gpu_preprocess_ms={avg_gpu_preprocess_ms:.1f} avg_gpu_gen_ms={avg_gpu_gen_ms:.1f} avg_flush_ms={avg_flush_ms:.1f} "
            f"avg_late_ms={avg_lateness_ms:.1f} avg_bytes={int(avg_bytes)} "
            f"slowest_build_pane={slowest_build_pane}:{slowest_build_ms:.1f}ms "
            f"slowest_gpu_build_pane={slowest_gpu_build_pane}:{slowest_gpu_build_ms:.1f}ms "
            f"slowest_flush_pane={slowest_pane}:{slowest_ms:.1f}ms {pane_bytes_text}"
        ),
        file=sys.stderr,
        flush=True,
    )
    stats.reset_window(now)


def current_shown_fps(
    stats: RuntimeStats, now: float, next_presented: int = 0
) -> float:
    elapsed = max(now - stats.window_started_at, 1e-6)
    return (stats.presented + max(0, int(next_presented))) / elapsed


def fps_overlay_text(stats: RuntimeStats, now: float, fps: float) -> str:
    return f"{current_shown_fps(stats, now, next_presented=1):04.1f}/{fps:04.1f}"


@lru_cache(maxsize=256)
def render_fps_overlay_patch(text: str, scale: int) -> np.ndarray:
    font_height = 5
    gap = scale
    border = scale
    glyph_masks = [
        np.array(
            [list(row) for row in FPS_OVERLAY_FONT.get(char, FPS_OVERLAY_FONT[" "])],
            dtype="U1",
        )
        == "1"
        for char in text
    ]

    text_width = sum(mask.shape[1] * scale + gap for mask in glyph_masks) - gap
    text_height = font_height * scale
    patch = np.zeros(
        (text_height + (border * 2), text_width + (border * 2), 3),
        dtype=np.uint8,
    )
    cursor_x = border
    cursor_y = border
    for mask in glyph_masks:
        expanded = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        glyph_height, glyph_width = expanded.shape
        patch[
            cursor_y : cursor_y + glyph_height,
            cursor_x : cursor_x + glyph_width,
        ][expanded] = 255
        cursor_x += glyph_width + gap
    return patch


def draw_fps_overlay(frame: np.ndarray, text: str) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        return

    scale = max(1, min(frame.shape[0] // 24, frame.shape[1] // 80, 3))
    margin = scale
    overlay = render_fps_overlay_patch(text, scale)
    box_height, box_width = overlay.shape[:2]

    frame_height, frame_width = frame.shape[:2]
    if box_width + margin > frame_width or box_height + margin > frame_height:
        return

    x0 = margin
    y0 = margin
    x1 = x0 + box_width
    y1 = y0 + box_height
    frame[y0:y1, x0:x1] = overlay.astype(frame.dtype, copy=False)


def open_panes(
    specs: list[PaneSpec], args: argparse.Namespace, fps: float, device: torch.device
) -> list[PaneRuntime]:
    panes: list[PaneRuntime] = []
    use_cuda_copy_streams = device.type == "cuda" and torch.cuda.is_available()
    use_cuda_build_streams = device.type == "cuda" and torch.cuda.is_available()
    for spec in specs:
        fd = os.open(spec.fifo_path, os.O_WRONLY)
        renderer = build_renderer(spec, args, fps, device)
        build_stream = (
            torch.cuda.Stream(device=device) if use_cuda_build_streams else None
        )
        copy_stream = (
            torch.cuda.Stream(device=device) if use_cuda_copy_streams else None
        )
        panes.append(
            PaneRuntime(
                spec=spec,
                fd=fd,
                renderer=renderer,
                cell_bounds=pane_cell_bounds(
                    spec,
                    str(args.render_mode).lower(),
                    int(args.quadrant_cell_divisor),
                    int(args.octant_cell_width_divisor),
                    int(args.octant_cell_height_divisor),
                ),
                build_stream=build_stream,
                copy_stream=copy_stream,
            )
        )
    return panes


def close_panes(panes: list[PaneRuntime]) -> None:
    for pane in panes:
        try:
            write_all(pane.fd, FINAL_SEQUENCE)
        except OSError:
            pass
        try:
            os.close(pane.fd)
        except OSError:
            pass


def close_launched_session(session: dict) -> None:
    pane_pids = []
    for pane in session.get("panes", []):
        pid = pane.get("pid")
        if isinstance(pid, int) and pid > 0:
            pane_pids.append(pid)
    focus_window = session.get("focus_window")
    if isinstance(focus_window, dict):
        focus_pid = focus_window.get("pid")
        if isinstance(focus_pid, int) and focus_pid > 0:
            pane_pids.append(focus_pid)

    for pid in pane_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    deadline = time.time() + 2.0
    remaining = list(pane_pids)
    while remaining and time.time() < deadline:
        next_remaining = []
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except OSError:
                continue
            next_remaining.append(pid)
        if not next_remaining:
            break
        time.sleep(0.05)
        remaining = next_remaining

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass


def maybe_launch_audio(path: str) -> subprocess.Popen[bytes] | None:
    return subprocess.Popen(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-vn",
            "-sn",
            path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    for cmd in ("ffprobe", "ffmpeg", "ffplay"):
        require_cmd(cmd)

    video_path = resolve_path(args.video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    session_dir = (
        resolve_path(args.session_dir)
        if args.session_dir
        else tempfile.mkdtemp(prefix="terminal-renderer-session-")
    )
    session_file = os.path.join(session_dir, "session.json")
    launched_session = not os.path.exists(session_file)
    if not os.path.exists(session_file):
        session_file = launch_session(session_dir, args.launcher)

    session = load_session(session_file)
    pane_specs, total_width, total_height = build_pane_specs(session, args)

    source_width, source_height, fps = probe_video_stream(video_path)
    decode_vf = build_ffmpeg_canvas_filter(
        source_width,
        source_height,
        total_width,
        total_height,
        cell_aspect=args.cell_aspect,
    )
    reader_output_width = total_width if decode_vf is not None else source_width
    reader_output_height = total_height if decode_vf is not None else source_height
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panes = open_panes(pane_specs, args, fps, device)
    shared_runtime = build_shared_runtime(total_width, total_height, args, fps, device)
    pane_ids = tuple(pane.spec.pane_id for pane in panes)

    audio_process = None
    frame_reader = None
    timing_handle = None
    stats = RuntimeStats(pane_ids=pane_ids, window_started_at=time.perf_counter())
    build_time_ema = 0.0
    flush_time_ema = 0.0
    try:
        for pane in panes:
            write_all(pane.fd, INIT_SEQUENCE)

        if args.timing_file:
            timing_handle = open(args.timing_file, "w", encoding="utf-8")
            timing_handle.write(timing_csv_header(pane_ids))

        frame_reader = LatestFrameReader(
            video_path,
            source_width,
            source_height,
            output_width=reader_output_width,
            output_height=reader_output_height,
            vf=decode_vf,
        )

        warmup_canvas = torch.zeros(
            (total_height, total_width, 3), dtype=torch.uint8, device=device
        )
        for pane in panes:
            pane.previous_frame = None
        build_shared_pane_payloads_with_stats(
            panes, shared_runtime, warmup_canvas, args
        )
        for pane in panes:
            pane.previous_frame = None
        shared_runtime.previous_frame = None

        fetch_start = time.perf_counter()
        first_item = frame_reader.get()
        first_fetch_time = time.perf_counter() - fetch_start
        if first_item is None:
            return 0
        first_frame_idx, first_frame_np = first_item
        first_skipped_input_frames = max(0, int(first_frame_idx))
        draw_fps_overlay(
            first_frame_np,
            fps_overlay_text(stats, time.perf_counter(), fps),
        )
        first_upload_start = time.perf_counter()
        first_frame = torch.from_numpy(first_frame_np).to(device)
        first_upload_time = time.perf_counter() - first_upload_start
        first_build_start = time.perf_counter()
        first_canvas = ensure_canvas_frame(
            first_frame,
            total_height,
            total_width,
            cell_aspect=args.cell_aspect,
        )
        (
            first_payloads,
            first_pane_build_times,
            first_preprocess_gpu_timing,
            first_next_previous_frame,
        ) = build_shared_pane_payloads_with_stats(
            panes,
            shared_runtime,
            first_canvas,
            args,
        )
        first_build_time = time.perf_counter() - first_build_start

        audio_process = maybe_launch_audio(video_path)
        start_time = time.perf_counter()
        playback_frame_idx = 0

        first_target_time = playback_target_time(
            start_time,
            playback_frame_idx,
            fps,
            float(args.audio_delay),
        )
        first_sleep = max(0.0, first_target_time - time.perf_counter())
        if first_sleep > 0:
            time.sleep(first_sleep)
        first_payload_bytes = payload_bytes_by_pane(first_payloads)
        first_flush_stats = flush_pane_payloads(
            panes,
            first_payloads,
            args.sync_mode,
            shared_runtime=shared_runtime,
            shared_next_previous_frame=first_next_previous_frame,
        )
        first_gpu_preprocess_time = shared_gpu_preprocess_time(
            first_preprocess_gpu_timing
        )
        first_pane_gpu_gen_times = gpu_gen_times_by_pane(first_payloads)
        first_pane_gpu_build_times = gpu_build_times_by_pane(first_payloads)
        first_gpu_build_time = shared_gpu_build_time(first_preprocess_gpu_timing) + sum(
            first_pane_gpu_build_times.values()
        )
        first_gpu_gen_time = sum(first_pane_gpu_gen_times.values())
        first_present_at = time.perf_counter()
        first_lateness = max(0.0, first_present_at - first_target_time)
        stats.record(
            dropped=False,
            skipped_input_frames=first_skipped_input_frames,
            fetch_time=first_fetch_time,
            upload_time=first_upload_time,
            build_time=first_build_time,
            gpu_build_time=first_gpu_build_time,
            gpu_preprocess_time=first_gpu_preprocess_time,
            gpu_gen_time=first_gpu_gen_time,
            pane_build_times=first_pane_build_times,
            pane_gpu_build_times=first_pane_gpu_build_times,
            sleep_time=first_sleep,
            flush_stats=first_flush_stats,
            lateness=first_lateness,
            payload_bytes=first_payload_bytes,
        )
        build_time_ema = first_upload_time + first_build_time
        flush_time_ema = first_flush_stats.total_time
        previous_frame_idx = first_frame_idx
        playback_frame_idx += 1

        if timing_handle is not None:
            timing_handle.write(
                timing_csv_row(
                    first_frame_idx,
                    False,
                    first_skipped_input_frames,
                    first_fetch_time,
                    first_upload_time,
                    first_build_time,
                    first_gpu_build_time,
                    first_pane_build_times,
                    first_pane_gpu_build_times,
                    first_sleep,
                    first_flush_stats,
                    first_lateness,
                    first_payload_bytes,
                    pane_ids,
                )
            )
        emit_runtime_stats(stats, first_present_at, float(args.stats_interval), fps)

        while True:
            fetch_start = time.perf_counter()
            item = frame_reader.get()
            fetch_time = time.perf_counter() - fetch_start
            if item is None:
                break
            frame_idx, frame_np = item
            skipped_input_frames = max(0, frame_idx - previous_frame_idx - 1)
            previous_frame_idx = frame_idx
            if should_drop_frame(
                time.perf_counter(),
                start_time,
                playback_frame_idx,
                fps,
                float(args.audio_delay),
                float(args.max_frame_lag),
                lead_time=build_time_ema + flush_time_ema,
            ):
                if timing_handle is not None:
                    empty_flush_stats = FlushStats(
                        total_time=0.0,
                        per_pane_times={pane_id: 0.0 for pane_id in pane_ids},
                    )
                    empty_pane_build_times = {pane_id: 0.0 for pane_id in pane_ids}
                    empty_pane_gpu_build_times = {pane_id: 0.0 for pane_id in pane_ids}
                    empty_payload_bytes = {pane_id: 0 for pane_id in pane_ids}
                    timing_handle.write(
                        timing_csv_row(
                            frame_idx,
                            True,
                            skipped_input_frames,
                            fetch_time,
                            0.0,
                            0.0,
                            0.0,
                            empty_pane_build_times,
                            empty_pane_gpu_build_times,
                            0.0,
                            empty_flush_stats,
                            0.0,
                            empty_payload_bytes,
                            pane_ids,
                        )
                    )
                stats.record(
                    dropped=True,
                    skipped_input_frames=skipped_input_frames,
                    fetch_time=fetch_time,
                    upload_time=0.0,
                    build_time=0.0,
                    gpu_build_time=0.0,
                    gpu_preprocess_time=0.0,
                    gpu_gen_time=0.0,
                    pane_build_times={pane_id: 0.0 for pane_id in pane_ids},
                    pane_gpu_build_times={pane_id: 0.0 for pane_id in pane_ids},
                    sleep_time=0.0,
                    flush_stats=FlushStats(
                        total_time=0.0,
                        per_pane_times={pane_id: 0.0 for pane_id in pane_ids},
                    ),
                    lateness=0.0,
                    payload_bytes={pane_id: 0 for pane_id in pane_ids},
                )
                emit_runtime_stats(
                    stats,
                    time.perf_counter(),
                    float(args.stats_interval),
                    fps,
                )
                playback_frame_idx += 1
                continue

            draw_fps_overlay(
                frame_np,
                fps_overlay_text(stats, time.perf_counter(), fps),
            )
            upload_start = time.perf_counter()
            frame = torch.from_numpy(frame_np).to(device)
            upload_time = time.perf_counter() - upload_start
            build_start = time.perf_counter()
            canvas = ensure_canvas_frame(
                frame,
                total_height,
                total_width,
                cell_aspect=args.cell_aspect,
            )
            (
                payloads,
                pane_build_times,
                preprocess_gpu_timing,
                next_previous_frame,
            ) = build_shared_pane_payloads_with_stats(
                panes,
                shared_runtime,
                canvas,
                args,
            )
            build_time = time.perf_counter() - build_start
            frame_payload_bytes = payload_bytes_by_pane(payloads)

            target_time = playback_target_time(
                start_time,
                playback_frame_idx,
                fps,
                float(args.audio_delay),
            )
            sleep_time = max(0.0, target_time - time.perf_counter())
            if sleep_time > 0:
                time.sleep(sleep_time)

            flush_stats = flush_pane_payloads(
                panes,
                payloads,
                args.sync_mode,
                shared_runtime=shared_runtime,
                shared_next_previous_frame=next_previous_frame,
            )
            gpu_preprocess_time = shared_gpu_preprocess_time(preprocess_gpu_timing)
            pane_gpu_gen_times = gpu_gen_times_by_pane(payloads)
            pane_gpu_build_times = gpu_build_times_by_pane(payloads)
            gpu_build_time = shared_gpu_build_time(preprocess_gpu_timing) + sum(
                pane_gpu_build_times.values()
            )
            gpu_gen_time = sum(pane_gpu_gen_times.values())
            presented_at = time.perf_counter()
            lateness = max(0.0, presented_at - target_time)
            stats.record(
                dropped=False,
                skipped_input_frames=skipped_input_frames,
                fetch_time=fetch_time,
                upload_time=upload_time,
                build_time=build_time,
                gpu_build_time=gpu_build_time,
                gpu_preprocess_time=gpu_preprocess_time,
                gpu_gen_time=gpu_gen_time,
                pane_build_times=pane_build_times,
                pane_gpu_build_times=pane_gpu_build_times,
                sleep_time=sleep_time,
                flush_stats=flush_stats,
                lateness=lateness,
                payload_bytes=frame_payload_bytes,
            )
            frame_work_time = upload_time + build_time
            alpha = 0.2
            build_time_ema = (
                frame_work_time
                if build_time_ema <= 0
                else ((1.0 - alpha) * build_time_ema + alpha * frame_work_time)
            )
            flush_time_ema = (
                flush_stats.total_time
                if flush_time_ema <= 0
                else ((1.0 - alpha) * flush_time_ema + alpha * flush_stats.total_time)
            )
            playback_frame_idx += 1

            if timing_handle is not None:
                timing_handle.write(
                    timing_csv_row(
                        frame_idx,
                        False,
                        skipped_input_frames,
                        fetch_time,
                        upload_time,
                        build_time,
                        gpu_build_time,
                        pane_build_times,
                        pane_gpu_build_times,
                        sleep_time,
                        flush_stats,
                        lateness,
                        frame_payload_bytes,
                        pane_ids,
                    )
                )
            emit_runtime_stats(
                stats,
                presented_at,
                float(args.stats_interval),
                fps,
            )

    except KeyboardInterrupt:
        return 130
    except PaneDisconnectedError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        if timing_handle is not None:
            timing_handle.flush()
            timing_handle.close()
        if audio_process is not None:
            audio_process.terminate()
            audio_process.wait()
        if frame_reader is not None:
            frame_reader.close()
        close_panes(panes)
        if launched_session:
            close_launched_session(session)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
