import argparse
import os
import shutil
import traceback
import time
from typing import Generator, Sequence

import torch

from src.ansi_renderer import AnsiRenderer
from src.config import Config
from src.terminal_router import cleanup_renderer
from src.video_playback import (
    LatestFrameReader,
    playback_target_time,
    probe_video_stream,
    should_drop_frame,
)

DEFAULT_VIDEO_PATH = "iSpyWithMyLittleEye by Voxicat [Ow7nDnZTbDw].mp4"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
MAX_FRAME_LAG = 1.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a video into one terminal.")
    parser.add_argument("video_path", nargs="?", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args(argv)


def resolve_video_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def build_config(
    video_path: str,
    fps: float,
    width: int,
    height: int,
) -> Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Config(
        width=width,
        height=height,
        device=device,
        fps=fps,
        audio_path=video_path,
        render_mode="quadrant",
        quadrant_cell_divisor=2,
        quant_mask=0xFF,
        diff_thresh=0,
        run_color_diff_thresh=0,
        relative_cursor_moves=True,
        use_rep=False,
        rep_min_run=12,
        sync_output=True,
    )


def empty_frame_generator(
    device: torch.device,
) -> Generator[torch.Tensor, None, None]:
    if False:
        yield torch.empty((0, 0, 3), dtype=torch.uint8, device=device)


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


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")


def run_video_demo(args: argparse.Namespace) -> int:
    video_path = resolve_video_path(str(args.video_path))
    for cmd in ("ffprobe", "ffmpeg", "ffplay"):
        require_cmd(cmd)

    source_width, source_height, fps = probe_video_stream(video_path)
    config = build_config(
        video_path,
        fps,
        int(args.width),
        int(args.height),
    )
    renderer = AnsiRenderer(
        frame_generator=empty_frame_generator(config.device),
        config=config,
        autostart=False,
    )
    frame_reader = LatestFrameReader(video_path, source_width, source_height)
    previous_frame: torch.Tensor | None = None
    playback_frame_idx = 0
    build_time_ema = 0.0
    flush_time_ema = 0.0

    try:
        while True:
            item = frame_reader.get()
            if item is None:
                break

            _, frame_np = item

            if renderer.start_time is not None and should_drop_frame(
                time.perf_counter(),
                renderer.start_time,
                playback_frame_idx,
                config.fps,
                config.audio_delay,
                MAX_FRAME_LAG,
                lead_time=build_time_ema + flush_time_ema,
            ):
                playback_frame_idx += 1
                continue

            upload_start = time.perf_counter()
            frame = torch.from_numpy(frame_np).to(config.device)
            upload_time = time.perf_counter() - upload_start

            build_start = time.perf_counter()
            payload, next_previous_frame = renderer.build_frame_payload(
                previous_frame, frame
            )
            build_time = time.perf_counter() - build_start

            output_view, copy_done_event, cpu_buf = payload_to_output(renderer, payload)

            sleep_time = 0.0
            target_time = None
            if renderer.start_time is not None:
                target_time = playback_target_time(
                    renderer.start_time,
                    playback_frame_idx,
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
                renderer.render_frame(output_view, playback_frame_idx)
            flush_time = time.perf_counter() - flush_start

            previous_frame = next_previous_frame
            if cpu_buf is not None:
                renderer.free_buffers.put(cpu_buf)

            frame_work_time = upload_time + build_time
            alpha = 0.2
            build_time_ema = (
                frame_work_time
                if build_time_ema <= 0.0
                else ((1.0 - alpha) * build_time_ema + alpha * frame_work_time)
            )
            flush_time_ema = (
                flush_time
                if flush_time_ema <= 0.0
                else ((1.0 - alpha) * flush_time_ema + alpha * flush_time)
            )
            playback_frame_idx += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        return 130
    except Exception as exc:
        print(f"Main thread caught exception: {exc}")
        traceback.print_exc()
        print("Program exiting due to thread crash")
        return 1
    finally:
        frame_reader.close()
        cleanup_renderer(renderer)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_video_demo(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
