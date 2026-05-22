import argparse
import os
from pathlib import Path

from src.config import CELL_ASPECT, Config, DEVICE
from src.multi_pane import MultiPaneOptions

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 60


def resolve_project_path(path: str | os.PathLike[str]) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((ROOT_DIR / candidate).resolve())


def build_render_config(args: argparse.Namespace, fps: float | None = None) -> Config:
    return Config(
        width=int(args.width),
        height=int(args.height),
        device=DEVICE,
        fps=float(args.fps if fps is None else fps),
        render_mode=str(args.render_mode),
        quadrant_cell_divisor=int(args.quadrant_cell_divisor),
        diff_thresh=int(args.diff_thresh),
        quant_mask=int(args.quant_mask),
        run_color_diff_thresh=int(args.run_color_diff_thresh),
        use_rep=True,
        rep_min_run=int(args.rep_min_run),
    )


def build_multi_pane_options(args: argparse.Namespace) -> MultiPaneOptions:
    return MultiPaneOptions(
        launcher=str(args.launcher),
        session_dir=str(args.session_dir) if args.session_dir else None,
        sync_mode=str(args.sync_mode),
        cell_aspect=float(args.cell_aspect),
        stats_interval=float(args.stats_interval),
    )


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--terminal-mode",
        choices=("single", "multi"),
        default="single",
        help="Render into one terminal or a four-pane launcher session.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--render-mode",
        choices=("pixel", "quadrant"),
        default="quadrant",
    )
    parser.add_argument("--quadrant-cell-divisor", type=int, default=2)
    parser.add_argument("--diff-thresh", type=int, default=0)
    parser.add_argument("--run-color-diff-thresh", type=int, default=0)
    parser.add_argument("--quant-mask", type=lambda value: int(value, 0), default=0xFF)
    parser.add_argument("--rep-min-run", type=int, default=12)


def add_multi_pane_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launcher", default="./open_four_terminals.sh")
    parser.add_argument("--session-dir")
    parser.add_argument(
        "--sync-mode",
        choices=("pane", "global", "off"),
        default="pane",
    )
    parser.add_argument("--cell-aspect", type=float, default=CELL_ASPECT)
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=0.0,
        help="Print multi-pane runtime stats every N seconds; 0 disables it.",
    )
