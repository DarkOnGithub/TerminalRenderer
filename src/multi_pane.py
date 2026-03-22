import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Generator, Iterable

import torch
import torch.nn.functional as F

from .ansi_generator import ansi_generate
from .ansi_renderer import AnsiRenderer
from .config import (
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
from .frame_processing import pre_process_frame

PANE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
INIT_SEQUENCE = ENABLE_ALT_BUFFER + CLEAR_SCREEN + HIDE_CURSOR + b"\033[H"
FINAL_SEQUENCE = SHOW_CURSOR + DISABLE_ALT_BUFFER

_FLUSH_EXECUTOR: ThreadPoolExecutor | None = None


class PaneDisconnectedError(RuntimeError):
    pass


@dataclass
class MultiPaneOptions:
    launcher: str = "./open_four_alacritty.sh"
    session_dir: str | None = None
    sync_mode: str = "pane"
    cell_aspect: float = CELL_ASPECT
    stats_interval: float = 0.0


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
    cell_bounds: tuple[int, int, int, int]
    previous_frame: torch.Tensor | None = None
    build_stream: torch.cuda.Stream | None = None
    copy_stream: torch.cuda.Stream | None = None
    cpu_payload_buffer: torch.Tensor | None = None


@dataclass
class PanePayload:
    pane: PaneRuntime
    payload_ref: object | None
    payload_view: memoryview | None
    copy_done_event: torch.cuda.Event | None
    next_previous_frame: torch.Tensor


@dataclass
class SharedBuildRuntime:
    renderer: AnsiRenderer
    previous_frame: torch.Tensor | None = None


@dataclass
class MultiPaneStats:
    window_started_at: float
    frames_rendered: int = 0
    build_time_sum: float = 0.0
    flush_time_sum: float = 0.0
    payload_bytes_sum: int = 0

    def record(self, build_time: float, flush_time: float, payload_bytes: int) -> None:
        self.frames_rendered += 1
        self.build_time_sum += build_time
        self.flush_time_sum += flush_time
        self.payload_bytes_sum += int(payload_bytes)

    def reset_window(self, now: float) -> None:
        self.window_started_at = now
        self.frames_rendered = 0
        self.build_time_sum = 0.0
        self.flush_time_sum = 0.0
        self.payload_bytes_sum = 0


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
        (target_height, target_width, 3),
        dtype=frame.dtype,
        device=frame.device,
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


def pane_target_size(config: Config, columns: int, lines: int) -> tuple[int, int]:
    if config.render_mode == "pixel":
        return columns, lines
    if config.render_mode == "quadrant":
        factor = max(1, int(config.quadrant_cell_divisor))
        return columns * factor, lines * factor
    if config.render_mode == "octant":
        return (
            columns * max(1, int(config.octant_cell_width_divisor)),
            lines * max(1, int(config.octant_cell_height_divisor)),
        )
    raise ValueError(f"Unsupported render mode: {config.render_mode}")


def build_pane_specs(session: dict, config: Config) -> tuple[list[PaneSpec], int, int]:
    panes_by_id = {pane["id"]: pane for pane in session["panes"]}

    top_left = panes_by_id["top_left"]
    top_right = panes_by_id["top_right"]
    bottom_left = panes_by_id["bottom_left"]
    bottom_right = panes_by_id["bottom_right"]

    tl_width, tl_height = pane_target_size(
        config,
        int(top_left["columns"]),
        int(top_left["lines"]),
    )
    tr_width, tr_height = pane_target_size(
        config,
        int(top_right["columns"]),
        int(top_right["lines"]),
    )
    bl_width, bl_height = pane_target_size(
        config,
        int(bottom_left["columns"]),
        int(bottom_left["lines"]),
    )
    br_width, br_height = pane_target_size(
        config,
        int(bottom_right["columns"]),
        int(bottom_right["lines"]),
    )

    if bl_width != tl_width or br_width != tr_width:
        raise RuntimeError("Pane column widths do not align across the 4-pane layout")
    if tr_height != tl_height or br_height != bl_height:
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
            x1=tl_width,
            y0=0,
            y1=tl_height,
        ),
        PaneSpec(
            pane_id="top_right",
            fifo_path=top_right["fifo"],
            columns=int(top_right["columns"]),
            lines=int(top_right["lines"]),
            target_width=tr_width,
            target_height=tr_height,
            x0=tl_width,
            x1=tl_width + tr_width,
            y0=0,
            y1=tl_height,
        ),
        PaneSpec(
            pane_id="bottom_left",
            fifo_path=bottom_left["fifo"],
            columns=int(bottom_left["columns"]),
            lines=int(bottom_left["lines"]),
            target_width=bl_width,
            target_height=bl_height,
            x0=0,
            x1=bl_width,
            y0=tl_height,
            y1=tl_height + bl_height,
        ),
        PaneSpec(
            pane_id="bottom_right",
            fifo_path=bottom_right["fifo"],
            columns=int(bottom_right["columns"]),
            lines=int(bottom_right["lines"]),
            target_width=br_width,
            target_height=br_height,
            x0=bl_width,
            x1=bl_width + br_width,
            y0=tl_height,
            y1=tl_height + bl_height,
        ),
    ]
    return specs, tl_width + tr_width, tl_height + bl_height


def _empty_frame_generator(device: torch.device) -> Generator[torch.Tensor, None, None]:
    if False:
        yield torch.empty((0, 0, 3), dtype=torch.uint8, device=device)


def build_renderer_config(base_config: Config, width: int, height: int) -> Config:
    return replace(
        base_config,
        width=width,
        height=height,
        audio_path=None,
        sync_output=False,
    )


def pane_cell_bounds(spec: PaneSpec, config: Config) -> tuple[int, int, int, int]:
    if config.render_mode == "pixel":
        return spec.x0, spec.x1, spec.y0, spec.y1
    if config.render_mode == "quadrant":
        divisor = max(1, int(config.quadrant_cell_divisor))
        return (
            spec.x0 // divisor,
            spec.x1 // divisor,
            spec.y0 // divisor,
            spec.y1 // divisor,
        )
    if config.render_mode == "octant":
        x_divisor = max(1, int(config.octant_cell_width_divisor))
        y_divisor = max(1, int(config.octant_cell_height_divisor))
        return (
            spec.x0 // x_divisor,
            spec.x1 // x_divisor,
            spec.y0 // y_divisor,
            spec.y1 // y_divisor,
        )
    raise ValueError(f"Unsupported render mode: {config.render_mode}")


def build_renderer(base_config: Config, width: int, height: int) -> AnsiRenderer:
    return AnsiRenderer(
        _empty_frame_generator(base_config.device),
        build_renderer_config(base_config, width, height),
        autostart=False,
    )


def open_panes(specs: list[PaneSpec], config: Config) -> list[PaneRuntime]:
    panes: list[PaneRuntime] = []
    use_cuda_streams = config.device.type == "cuda" and torch.cuda.is_available()
    for spec in specs:
        fd = os.open(spec.fifo_path, os.O_WRONLY)
        panes.append(
            PaneRuntime(
                spec=spec,
                fd=fd,
                renderer=build_renderer(config, spec.target_width, spec.target_height),
                cell_bounds=pane_cell_bounds(spec, config),
                build_stream=(
                    torch.cuda.Stream(device=config.device)
                    if use_cuda_streams
                    else None
                ),
                copy_stream=(
                    torch.cuda.Stream(device=config.device)
                    if use_cuda_streams
                    else None
                ),
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
    pane_pids: list[int] = []
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
        except (OSError, ProcessLookupError):
            pass

    deadline = time.time() + 2.0
    remaining = list(pane_pids)
    while remaining and time.time() < deadline:
        next_remaining: list[int] = []
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                continue
            next_remaining.append(pid)
        if not next_remaining:
            break
        time.sleep(0.05)
        remaining = next_remaining

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


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


def _new_gpu_segment() -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def payload_bytes(payloads: list[PanePayload]) -> int:
    return sum(
        0 if payload.payload_view is None else int(payload.payload_view.nbytes)
        for payload in payloads
    )


def maybe_emit_runtime_stats(
    stats: MultiPaneStats,
    interval: float,
    now: float,
) -> None:
    if interval <= 0.0 or now - stats.window_started_at < interval:
        return

    elapsed = max(now - stats.window_started_at, 1e-6)
    frames = max(stats.frames_rendered, 1)
    shown_fps = stats.frames_rendered / elapsed
    avg_build_ms = 1000.0 * stats.build_time_sum / frames
    avg_flush_ms = 1000.0 * stats.flush_time_sum / frames
    avg_bytes = stats.payload_bytes_sum / frames
    print(
        (
            f"multi-pane stats fps={shown_fps:.1f} "
            f"avg_build_ms={avg_build_ms:.1f} "
            f"avg_flush_ms={avg_flush_ms:.1f} "
            f"avg_bytes={int(avg_bytes)}"
        ),
        file=sys.stderr,
        flush=True,
    )
    stats.reset_window(now)


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
    copy_done_event = None
    if pane.copy_stream is not None:
        current_stream = torch.cuda.current_stream(device=payload.device)
        with torch.cuda.stream(pane.copy_stream):
            pane.copy_stream.wait_stream(current_stream)
            cpu_payload_view.copy_(payload, non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(pane.copy_stream)
    else:
        cpu_payload_view.copy_(payload, non_blocking=False)

    return cpu_payload_view, memoryview(cpu_payload_view.numpy()), copy_done_event


def build_shared_pane_payloads(
    panes: list[PaneRuntime],
    runtime: SharedBuildRuntime,
    canvas: torch.Tensor,
    config: Config,
) -> tuple[list[PanePayload], torch.Tensor]:
    xs, ys, colors_rgb, updated_previous = pre_process_frame(
        runtime.previous_frame,
        canvas,
        runtime.renderer.config,
        quant_mask=int(runtime.renderer.config.quant_mask),
        diff_thresh_override=int(runtime.renderer.config.diff_thresh),
    )

    payloads: list[PanePayload] = []
    pane_bounds = {pane.spec.pane_id: pane.cell_bounds for pane in panes}
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
        pane_id = pane.spec.pane_id
        x0, x1, y0, y1 = pane_bounds[pane_id]
        next_previous_frame = updated_previous[y0:y1, x0:x1]

        if xs.numel() == 0:
            payloads.append(
                PanePayload(
                    pane=pane,
                    payload_ref=None,
                    payload_view=None,
                    copy_done_event=None,
                    next_previous_frame=next_previous_frame,
                )
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
                    next_previous_frame=next_previous_frame,
                )
            )
            continue

        pane_xs = xs.index_select(0, pane_selector) - x0
        pane_ys = ys.index_select(0, pane_selector) - y0
        pane_colors = colors_rgb.index_select(0, pane_selector)

        clear_prefix = None
        old_shape = (
            pane.previous_frame.shape if pane.previous_frame is not None else None
        )
        shape_changed = old_shape is not None and old_shape != next_previous_frame.shape

        if pane.build_stream is not None:
            current_stream = torch.cuda.current_stream(device=config.device)
            with torch.cuda.stream(pane.build_stream):
                pane.build_stream.wait_stream(current_stream)
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
                if shape_changed:
                    clear_prefix = torch.tensor(
                        list(b"\033[2J\033[H"),
                        dtype=torch.uint8,
                        device=config.device,
                    )
                    payload = torch.cat([clear_prefix, payload])
                payload_ref, payload_view, copy_done_event = tensor_payload_view(
                    pane, payload
                )
        else:
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
            if shape_changed:
                clear_prefix = torch.tensor(
                    list(b"\033[2J\033[H"),
                    dtype=torch.uint8,
                    device=config.device,
                )
                payload = torch.cat([clear_prefix, payload])
            payload_ref, payload_view, copy_done_event = tensor_payload_view(
                pane, payload
            )

        payloads.append(
            PanePayload(
                pane=pane,
                payload_ref=payload_ref,
                payload_view=payload_view,
                copy_done_event=copy_done_event,
                next_previous_frame=next_previous_frame,
            )
        )

    return payloads, updated_previous


def flush_pane_payloads(
    panes: list[PaneRuntime],
    payloads: list[PanePayload],
    sync_mode: str,
) -> None:
    if sync_mode == "global":
        for pane in panes:
            write_all(pane.fd, SYNC_OUTPUT_BEGIN)
        for payload in payloads:
            if payload.payload_view is None:
                continue
            if payload.copy_done_event is not None:
                payload.copy_done_event.synchronize()
            write_all(payload.pane.fd, payload.payload_view)
        for pane in panes:
            write_all(pane.fd, SYNC_OUTPUT_END)
    elif sync_mode == "pane":
        active_payloads = [
            payload for payload in payloads if payload.payload_view is not None
        ]

        if len(active_payloads) > 1:

            def flush_one(payload: PanePayload) -> None:
                payload_view = payload.payload_view
                if payload_view is None:
                    return
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                writev_all(
                    payload.pane.fd,
                    [SYNC_OUTPUT_BEGIN, payload_view, SYNC_OUTPUT_END],
                )

            list(_get_flush_executor().map(flush_one, active_payloads))
        else:
            for payload in active_payloads:
                payload_view = payload.payload_view
                if payload_view is None:
                    continue
                if payload.copy_done_event is not None:
                    payload.copy_done_event.synchronize()
                writev_all(
                    payload.pane.fd,
                    [SYNC_OUTPUT_BEGIN, payload_view, SYNC_OUTPUT_END],
                )
    elif sync_mode == "off":
        for payload in payloads:
            if payload.payload_view is None:
                continue
            if payload.copy_done_event is not None:
                payload.copy_done_event.synchronize()
            write_all(payload.pane.fd, payload.payload_view)
    else:
        raise ValueError(f"Unsupported sync mode: {sync_mode}")

    for payload in payloads:
        payload.pane.previous_frame = payload.next_previous_frame


class MultiPaneRenderer:
    def __init__(self, config: Config, options: MultiPaneOptions | None = None):
        self.config = config
        self.options = options or MultiPaneOptions()

        session_dir = self.options.session_dir
        self.session_dir = (
            resolve_path(session_dir)
            if session_dir
            else tempfile.mkdtemp(prefix="terminal-renderer-session-")
        )
        session_file = os.path.join(self.session_dir, "session.json")
        self.launched_session = not os.path.exists(session_file)
        if self.launched_session:
            session_file = launch_session(self.session_dir, self.options.launcher)

        self.session = load_session(session_file)
        self.pane_specs, self.total_width, self.total_height = build_pane_specs(
            self.session,
            self.config,
        )
        self.panes = open_panes(self.pane_specs, self.config)
        self.shared_runtime = SharedBuildRuntime(
            renderer=build_renderer(self.config, self.total_width, self.total_height)
        )
        self.stats = MultiPaneStats(window_started_at=time.perf_counter())
        self.closed = False

        for pane in self.panes:
            write_all(pane.fd, INIT_SEQUENCE)

    def render_frame(self, frame: torch.Tensor) -> None:
        build_start = time.perf_counter()
        if isinstance(frame, torch.Tensor):
            frame = frame.to(device=self.config.device, dtype=torch.uint8)
        else:
            frame = torch.as_tensor(frame, dtype=torch.uint8, device=self.config.device)
        canvas = ensure_canvas_frame(
            frame,
            self.total_height,
            self.total_width,
            cell_aspect=self.options.cell_aspect,
        )
        payloads, next_previous_frame = build_shared_pane_payloads(
            self.panes,
            self.shared_runtime,
            canvas,
            self.config,
        )
        build_time = time.perf_counter() - build_start

        flush_start = time.perf_counter()
        flush_pane_payloads(self.panes, payloads, self.options.sync_mode)
        flush_time = time.perf_counter() - flush_start
        self.shared_runtime.previous_frame = next_previous_frame
        self.stats.record(build_time, flush_time, payload_bytes(payloads))
        maybe_emit_runtime_stats(
            self.stats,
            float(self.options.stats_interval),
            time.perf_counter(),
        )

    def render_frames(self, frames: Iterable[torch.Tensor]) -> None:
        for frame in frames:
            self.render_frame(frame)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close_panes(self.panes)
        if self.launched_session:
            close_launched_session(self.session)

    def __enter__(self) -> "MultiPaneRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
