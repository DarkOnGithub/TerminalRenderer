import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Generator

import torch

from src.ansi_renderer import AnsiRenderer
from src.config import Config, DEVICE
from src.demos.common import (
    ROOT_DIR,
    build_multi_pane_options,
    resolve_project_path,
)
from src.multi_pane import MultiPaneRenderer
from src.terminal_router import cleanup_renderer
from src.video_playback import (
    LatestFrameReader,
    playback_target_time,
    probe_video_stream,
    should_drop_frame,
)

DEFAULT_VIDEO_PATH = ROOT_DIR / "iSpyWithMyLittleEye by Voxicat [Ow7nDnZTbDw].mp4"
MAX_FRAME_LAG = 1.0


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video_path", nargs="?", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--fps", type=float, help="Override detected video FPS.")
    parser.add_argument("--audio-delay", type=float, default=0.0)
    parser.add_argument("--max-frame-lag", type=float, default=MAX_FRAME_LAG)
    parser.add_argument(
        "--cursor-moves",
        choices=("absolute", "relative"),
        default="relative",
    )


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")


def empty_frame_generator(device: torch.device) -> Generator[torch.Tensor, None, None]:
    if False:
        yield torch.empty((0, 0, 3), dtype=torch.uint8, device=device)


def build_config(args: argparse.Namespace, fps: float) -> Config:
    return Config(
        width=int(args.width),
        height=int(args.height),
        device=DEVICE,
        fps=fps,
        audio_path=str(args.video_path) if args.terminal_mode == "single" else None,
        audio_delay=float(args.audio_delay),
        render_mode=str(args.render_mode),
        quadrant_cell_divisor=int(args.quadrant_cell_divisor),
        quant_mask=int(args.quant_mask),
        diff_thresh=int(args.diff_thresh),
        run_color_diff_thresh=int(args.run_color_diff_thresh),
        relative_cursor_moves=str(args.cursor_moves) == "relative",
        use_rep=True,
        rep_min_run=int(args.rep_min_run),
        sync_output=args.terminal_mode == "single",
    )


def payload_to_output(
    renderer: AnsiRenderer,
    payload: torch.Tensor | None,
) -> tuple[memoryview | None, torch.cuda.Event | None, torch.Tensor | None]:
    if payload is None:
        return None, None, None
    if payload.device.type == "cpu":
        return memoryview(payload.numpy()), None, None

    cpu_buf = renderer.free_buffers.get()
    if cpu_buf.size(0) < payload.size(0):
        cpu_buf = torch.empty(
            int(payload.size(0) * 1.2),
            dtype=torch.uint8,
            pin_memory=renderer.cuda_enabled,
        )

    cpu_view = cpu_buf[: payload.size(0)]
    copy_done_event = None
    if renderer.copy_stream is not None:
        current_stream = torch.cuda.current_stream(device=renderer.config.device)
        with torch.cuda.stream(renderer.copy_stream):
            renderer.copy_stream.wait_stream(current_stream)
            cpu_view.copy_(payload, non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(renderer.copy_stream)
    else:
        cpu_view.copy_(payload, non_blocking=renderer.cuda_enabled)
        if renderer.cuda_enabled:
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(
                torch.cuda.current_stream(device=renderer.config.device)
            )

    return memoryview(cpu_view.numpy()), copy_done_event, cpu_buf


def render_single(args: argparse.Namespace, fps: float) -> int:
    config = build_config(args, fps)
    renderer = AnsiRenderer(
        empty_frame_generator(config.device),
        config,
        autostart=False,
    )
    frame_reader = LatestFrameReader(
        str(args.video_path),
        int(args.source_width),
        int(args.source_height),
        realtime=True,
        pause_after_first_frame=True,
    )
    previous_frame: torch.Tensor | None = None
    decoder_resumed = False
    build_time_ema = 0.0
    flush_time_ema = 0.0

    try:
        while True:
            item = frame_reader.get()
            if item is None:
                break

            frame_idx, frame_np = item
            if renderer.start_time is not None and should_drop_frame(
                time.perf_counter(),
                renderer.start_time,
                frame_idx,
                config.fps,
                config.audio_delay,
                float(args.max_frame_lag),
                lead_time=build_time_ema + flush_time_ema,
            ):
                continue

            upload_start = time.perf_counter()
            frame = torch.from_numpy(frame_np).to(config.device)
            upload_time = time.perf_counter() - upload_start

            build_start = time.perf_counter()
            payload, next_previous_frame = renderer.build_frame_payload(
                previous_frame,
                frame,
            )
            build_time = time.perf_counter() - build_start
            output_view, copy_done_event, cpu_buf = payload_to_output(renderer, payload)

            if renderer.start_time is not None:
                target_time = playback_target_time(
                    renderer.start_time,
                    frame_idx,
                    config.fps,
                    config.audio_delay,
                )
                sleep_time = max(0.0, target_time - time.perf_counter())
                if sleep_time > 0:
                    time.sleep(sleep_time)

            flush_start = time.perf_counter()
            if copy_done_event is not None:
                copy_done_event.synchronize()
            if output_view is not None:
                renderer.render_frame(output_view, frame_idx)
                if not decoder_resumed:
                    frame_reader.resume()
                    decoder_resumed = True
            flush_time = time.perf_counter() - flush_start

            previous_frame = next_previous_frame
            if cpu_buf is not None:
                renderer.free_buffers.put(cpu_buf)

            alpha = 0.2
            frame_work_time = upload_time + build_time
            build_time_ema = (
                frame_work_time
                if build_time_ema <= 0.0
                else ((1.0 - alpha) * build_time_ema) + (alpha * frame_work_time)
            )
            flush_time_ema = (
                flush_time
                if flush_time_ema <= 0.0
                else ((1.0 - alpha) * flush_time_ema) + (alpha * flush_time)
            )
    finally:
        frame_reader.close()
        cleanup_renderer(renderer)

    return 0


def start_audio(path: str) -> subprocess.Popen:
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


def render_multi(args: argparse.Namespace, fps: float) -> int:
    config = build_config(args, fps)
    frame_reader = LatestFrameReader(
        str(args.video_path),
        int(args.source_width),
        int(args.source_height),
        realtime=True,
        pause_after_first_frame=True,
    )
    audio_process: subprocess.Popen | None = None
    start_time: float | None = None
    decoder_resumed = False

    try:
        with MultiPaneRenderer(config, build_multi_pane_options(args)) as renderer:
            while True:
                item = frame_reader.get()
                if item is None:
                    break

                frame_idx, frame_np = item
                if start_time is not None and should_drop_frame(
                    time.perf_counter(),
                    start_time,
                    frame_idx,
                    fps,
                    config.audio_delay,
                    float(args.max_frame_lag),
                ):
                    continue

                if start_time is not None:
                    target_time = playback_target_time(
                        start_time,
                        frame_idx,
                        fps,
                        config.audio_delay,
                    )
                    sleep_time = max(0.0, target_time - time.perf_counter())
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                renderer.render_frame(torch.from_numpy(frame_np).to(config.device))
                if start_time is None:
                    audio_process = start_audio(str(args.video_path))
                    start_time = time.perf_counter()
                if not decoder_resumed:
                    frame_reader.resume()
                    decoder_resumed = True
    finally:
        frame_reader.close()
        if audio_process is not None:
            audio_process.terminate()
            audio_process.wait()

    return 0


def run(args: argparse.Namespace) -> int:
    for cmd in ("ffprobe", "ffmpeg", "ffplay"):
        require_cmd(cmd)

    args.video_path = resolve_project_path(args.video_path)
    if not Path(args.video_path).exists():
        raise FileNotFoundError(args.video_path)

    args.source_width, args.source_height, fps = probe_video_stream(args.video_path)
    if args.fps is not None:
        fps = float(args.fps)

    if args.terminal_mode == "multi":
        return render_multi(args, fps)
    return render_single(args, fps)
