import argparse
import os
import sys
import traceback
from typing import Sequence

from src.demos import cube, shader, video
from src.demos.common import (
    DEFAULT_FPS,
    add_multi_pane_args,
    add_render_args,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="terminal-renderer",
        description="Render videos, shaders, and procedural scenes in the terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    video_parser = subparsers.add_parser("video", help="Render a video file.")
    video.add_cli_args(video_parser)
    add_render_args(video_parser)
    add_multi_pane_args(video_parser)
    video_parser.set_defaults(func=video.run)

    cube_parser = subparsers.add_parser("cube", help="Render a procedural cube.")
    cube_parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    cube_parser.add_argument("--depth", type=float, default=cube.DEFAULT_DEPTH)
    add_render_args(cube_parser)
    add_multi_pane_args(cube_parser)
    cube_parser.set_defaults(func=cube.run)

    shader_parser = subparsers.add_parser("shader", help="Render a GLSL shader.")
    shader_parser.add_argument("shader_path", nargs="?")
    shader_parser.add_argument("--fps", type=int, default=shader.FPS)
    shader_parser.add_argument("--time-scale", type=float, default=1.0)
    add_render_args(shader_parser)
    add_multi_pane_args(shader_parser)
    shader_parser.set_defaults(func=shader.run)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"terminal-renderer: {exc}", file=sys.stderr)
        if os.environ.get("TERMINAL_RENDERER_DEBUG"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
