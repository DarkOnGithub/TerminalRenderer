import argparse
from typing import Sequence

from example.object import main as object_demo_main
from multi_terminal_player import main as multi_terminal_main
from video_demo import DEFAULT_VIDEO_PATH, main as video_demo_main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TerminalRenderer demo router.")
    subparsers = parser.add_subparsers(dest="demo", required=True)

    video_parser = subparsers.add_parser("video", help="Play the video demo")
    video_parser.add_argument(
        "--terminal-mode",
        choices=("single", "multi"),
        default="single",
    )
    video_parser.add_argument("video_path", nargs="?", default=DEFAULT_VIDEO_PATH)
    video_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args for the selected video mode. Prefix with --.",
    )

    object_parser = subparsers.add_parser("object", help="Play the cube demo")
    object_parser.add_argument(
        "--terminal-mode",
        choices=("single", "multi"),
        default="single",
    )
    object_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args for the cube demo. Prefix with --.",
    )

    return parser.parse_args(argv)


def _normalize_extra_args(extra_args: list[str]) -> list[str]:
    return extra_args[1:] if extra_args and extra_args[0] == "--" else extra_args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    extra_args = _normalize_extra_args(list(args.extra_args))

    if args.demo == "video":
        if args.terminal_mode == "multi":
            return multi_terminal_main([str(args.video_path), *extra_args])
        return video_demo_main([str(args.video_path), *extra_args])

    return object_demo_main(
        [
            "--terminal-mode",
            str(args.terminal_mode),
            *extra_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
