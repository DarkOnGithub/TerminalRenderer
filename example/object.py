import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import torch

try:
    from src.config import Config, DEVICE
    from src.multi_pane import MultiPaneOptions
    from src.terminal_router import render_with_terminal_mode
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.config import Config, DEVICE
    from src.multi_pane import MultiPaneOptions
    from src.terminal_router import render_with_terminal_mode

FPS = 60
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_DEPTH = 1.0

Point3 = tuple[float, float, float]
ScreenPoint = tuple[int, int, float]

VERTICES: tuple[Point3, ...] = (
    (-0.25, -0.25, -0.25),
    (0.25, -0.25, -0.25),
    (0.25, 0.25, -0.25),
    (-0.25, 0.25, -0.25),
    (-0.25, -0.25, 0.25),
    (0.25, -0.25, 0.25),
    (0.25, 0.25, 0.25),
    (-0.25, 0.25, 0.25),
)

FACES: tuple[tuple[int, ...], ...] = (
    (4, 5, 6, 7),
    (1, 0, 3, 2),
    (7, 6, 2, 3),
    (0, 1, 5, 4),
    (0, 4, 7, 3),
    (5, 1, 2, 6),
)


@dataclass
class SceneSettings:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = FPS
    depth: float = DEFAULT_DEPTH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the procedural cube demo.")
    parser.add_argument(
        "--terminal-mode",
        choices=("single", "multi"),
        default="single",
        help="Render into one terminal or a multi-pane launcher session.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--render-mode",
        choices=("pixel", "quadrant"),
        default="quadrant",
    )
    parser.add_argument("--quadrant-cell-divisor", type=int, default=2)
    parser.add_argument("--launcher", default="./open_four_alacritty.sh")
    parser.add_argument("--session-dir")
    parser.add_argument(
        "--sync-mode",
        choices=("pane", "global", "off"),
        default="pane",
    )
    parser.add_argument("--cell-aspect", type=float, default=0.5)
    return parser.parse_args(argv)


def rotate_yz(point: Point3, angle: float) -> Point3:
    x, y, z = point
    c = math.cos(angle)
    s = math.sin(angle)
    return x, (y * c) - (z * s), (y * s) + (z * c)


def rotate_xz(point: Point3, angle: float) -> Point3:
    x, y, z = point
    c = math.cos(angle)
    s = math.sin(angle)
    return (x * c) - (z * s), y, (x * s) + (z * c)


def rotate_xy(point: Point3, angle: float) -> Point3:
    x, y, z = point
    c = math.cos(angle)
    s = math.sin(angle)
    return (x * c) - (y * s), (x * s) + (y * c), z


def translate_z(point: Point3, dz: float) -> Point3:
    x, y, z = point
    return x, y, z + dz


def project(point: Point3) -> Point3:
    x, y, z = point
    if z == 0:
        z = 0.001
    return x / z, y / z, z


def screen_space(point: Point3, width: int, height: int) -> ScreenPoint:
    x, y, z = point
    return (
        int(((x + 1.0) / 2.0) * (width - 1)),
        int((1.0 - ((y + 1.0) / 2.0)) * (height - 1)),
        z,
    )


def fill_triangle(
    buffer: torch.Tensor,
    z_buffer: torch.Tensor,
    p0: ScreenPoint,
    p1: ScreenPoint,
    p2: ScreenPoint,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    width: int,
    height: int,
) -> None:
    min_x = max(0, int(min(p0[0], p1[0], p2[0])))
    max_x = min(width - 1, int(max(p0[0], p1[0], p2[0])))
    min_y = max(0, int(min(p0[1], p1[1], p2[1])))
    max_y = min(height - 1, int(max(p0[1], p1[1], p2[1])))
    if min_x > max_x or min_y > max_y:
        return

    yy, xx = torch.meshgrid(
        torch.arange(min_y, max_y + 1, device=DEVICE),
        torch.arange(min_x, max_x + 1, device=DEVICE),
        indexing="ij",
    )

    v0x, v0y = p1[0] - p0[0], p1[1] - p0[1]
    v1x, v1y = p2[0] - p0[0], p2[1] - p0[1]
    v2x, v2y = xx - p0[0], yy - p0[1]

    den = (v0x * v1y) - (v1x * v0y)
    if abs(den) < 1e-6:
        return

    v = ((v2x * v1y) - (v1x * v2y)) / den
    w = ((v0x * v2y) - (v2x * v0y)) / den
    u = 1.0 - v - w
    mask = (u >= 0) & (v >= 0) & (w >= 0)

    z_interp = (u * p0[2]) + (v * p1[2]) + (w * p2[2])
    current_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
    depth_mask = mask & (z_interp < current_z)
    current_z[depth_mask] = z_interp[depth_mask]

    u_m = u[depth_mask].unsqueeze(-1)
    v_m = v[depth_mask].unsqueeze(-1)
    w_m = w[depth_mask].unsqueeze(-1)
    colors_interp = (u_m * c0 + v_m * c1 + w_m * c2).to(torch.uint8)
    buffer[min_y : max_y + 1, min_x : max_x + 1][depth_mask] = colors_interp


def dynamic_vertex_colors(angle: float) -> torch.Tensor:
    vertex_ids = torch.arange(len(VERTICES), device=DEVICE, dtype=torch.float32)
    red = 127.5 + 127.5 * torch.sin(angle + (vertex_ids * 0.5))
    green = 127.5 + 127.5 * torch.sin((angle * 1.5) + (vertex_ids * 0.7) + 2.0)
    blue = 127.5 + 127.5 * torch.sin((angle * 0.7) + (vertex_ids * 0.9) + 4.0)
    return torch.stack([red, green, blue], dim=1)


def transform_vertex(
    vertex: Point3,
    angle: float,
    depth: float,
    width: int,
    height: int,
) -> ScreenPoint:
    rotated = rotate_yz(vertex, angle)
    rotated = rotate_xz(rotated, angle * 0.7)
    rotated = rotate_xy(rotated, angle * 0.3)
    translated = translate_z(rotated, depth)
    return screen_space(project(translated), width, height)


def render_frame_to_tensor(
    angle: float,
    depth: float,
    width: int,
    height: int,
) -> torch.Tensor:
    buffer = torch.zeros((height, width, 3), dtype=torch.uint8, device=DEVICE)
    z_buffer = torch.full((height, width), float("inf"), device=DEVICE)
    vertex_colors = dynamic_vertex_colors(angle)

    for face in FACES:
        face_vertices = [
            transform_vertex(VERTICES[index], angle, depth, width, height)
            for index in face
        ]
        face_colors = [vertex_colors[index] for index in face]

        for triangle_index in range(1, len(face_vertices) - 1):
            fill_triangle(
                buffer,
                z_buffer,
                face_vertices[0],
                face_vertices[triangle_index],
                face_vertices[triangle_index + 1],
                face_colors[0],
                face_colors[triangle_index],
                face_colors[triangle_index + 1],
                width,
                height,
            )

    return buffer


def frame_generator(
    settings: SceneSettings,
) -> Generator[torch.Tensor, None, None]:
    angle = 0.0
    delta = math.pi * (1.0 / max(settings.fps, 1))
    while True:
        yield render_frame_to_tensor(
            angle,
            settings.depth,
            settings.width,
            settings.height,
        )
        angle += delta


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        width=int(args.width),
        height=int(args.height),
        device=DEVICE,
        fps=float(args.fps),
        render_mode=str(args.render_mode),
        quadrant_cell_divisor=int(args.quadrant_cell_divisor),
        diff_thresh=0,
        quant_mask=0xFF,
        run_color_diff_thresh=0,
        use_rep=True,
        rep_min_run=4,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = SceneSettings(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        depth=float(args.depth),
    )
    multi_pane_options = MultiPaneOptions(
        launcher=str(args.launcher),
        session_dir=str(args.session_dir) if args.session_dir else None,
        sync_mode=str(args.sync_mode),
        cell_aspect=float(args.cell_aspect),
    )
    render_with_terminal_mode(
        frame_generator(settings),
        build_config(args),
        terminal_mode=str(args.terminal_mode),
        multi_pane_options=multi_pane_options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
