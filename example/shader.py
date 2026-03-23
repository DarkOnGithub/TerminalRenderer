import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
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

FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

VERTEX_SHADER = """
#version 330

in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

DEFAULT_FRAGMENT_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform float u_time;

out vec4 f_color;

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution.xy) * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;

    float radius = length(uv);
    float wave = 0.5 + 0.5 * sin((radius * 14.0) - (u_time * 3.5));

    float red = 0.5 + 0.5 * sin((uv.x * 4.0) + (u_time * 1.2));
    float green = 0.5 + 0.5 * sin((uv.y * 5.0) - (u_time * 1.4));
    float blue = mix(0.15, 1.0, wave);

    f_color = vec4(red, green, blue, 1.0);
}
"""

FULLSCREEN_TRIANGLE_STRIP = np.array(
    [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    dtype="f4",
)

DEFAULT_GLSL_VERSION = "#version 330"
VERSION_LINE_PATTERN = re.compile(r"^\s*#version[^\n]*")


@dataclass
class SceneSettings:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = FPS
    shader_path: str | None = None
    time_scale: float = 1.0


class ShaderRunner:
    def __init__(self, width: int, height: int, fragment_shader: str) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise RuntimeError(
                "Shader rendering requires moderngl and glcontext. Install them with `uv pip install -e .`."
            ) from exc

        self._moderngl = moderngl
        self.width = int(width)
        self.height = int(height)
        self.frame_index = 0
        self.last_elapsed_seconds = 0.0
        self.ctx = moderngl.create_standalone_context()
        self.framebuffer = self.ctx.simple_framebuffer(
            (self.width, self.height),
            components=3,
        )
        self.framebuffer.use()
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=build_fragment_shader(fragment_shader),
        )
        self.resolution_uniform = self._get_uniform("u_resolution")
        self.time_uniform = self._get_uniform("u_time")
        self.shader_toy_resolution_uniform = self._get_uniform("iResolution")
        self.shader_toy_time_uniform = self._get_uniform("iTime")
        self.shader_toy_delta_uniform = self._get_uniform("iTimeDelta")
        self.shader_toy_frame_uniform = self._get_uniform("iFrame")
        self.shader_toy_frame_rate_uniform = self._get_uniform("iFrameRate")
        self.shader_toy_mouse_uniform = self._get_uniform("iMouse")
        self.vertex_buffer = self.ctx.buffer(FULLSCREEN_TRIANGLE_STRIP.tobytes())
        self.vertex_array = self.ctx.simple_vertex_array(
            self.program,
            self.vertex_buffer,
            "in_vert",
        )

    def _get_uniform(self, name: str):
        try:
            return self.program[name]
        except KeyError:
            return None

    def render(self, elapsed_seconds: float) -> np.ndarray:
        self.framebuffer.use()
        self.framebuffer.clear(0.0, 0.0, 0.0, 1.0)
        delta_seconds = max(0.0, elapsed_seconds - self.last_elapsed_seconds)

        if self.resolution_uniform is not None:
            self.resolution_uniform.value = (float(self.width), float(self.height))

        if self.time_uniform is not None:
            self.time_uniform.value = float(elapsed_seconds)

        if self.shader_toy_resolution_uniform is not None:
            self.shader_toy_resolution_uniform.value = (
                float(self.width),
                float(self.height),
                1.0,
            )

        if self.shader_toy_time_uniform is not None:
            self.shader_toy_time_uniform.value = float(elapsed_seconds)

        if self.shader_toy_delta_uniform is not None:
            self.shader_toy_delta_uniform.value = float(delta_seconds)

        if self.shader_toy_frame_uniform is not None:
            self.shader_toy_frame_uniform.value = int(self.frame_index)

        if self.shader_toy_frame_rate_uniform is not None:
            self.shader_toy_frame_rate_uniform.value = (
                0.0 if delta_seconds <= 0.0 else 1.0 / delta_seconds
            )

        if self.shader_toy_mouse_uniform is not None:
            self.shader_toy_mouse_uniform.value = (0.0, 0.0, 0.0, 0.0)

        self.vertex_array.render(self._moderngl.TRIANGLE_STRIP)
        self.frame_index += 1
        self.last_elapsed_seconds = elapsed_seconds

        frame_bytes = self.framebuffer.read(components=3, alignment=1)
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            self.height,
            self.width,
            3,
        )
        return np.flipud(frame).copy()

    def close(self) -> None:
        for resource_name in (
            "vertex_array",
            "vertex_buffer",
            "program",
            "framebuffer",
            "ctx",
        ):
            resource = getattr(self, resource_name, None)
            if resource is None:
                continue
            release = getattr(resource, "release", None)
            if callable(release):
                release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an offscreen GLSL fragment shader through TerminalRenderer.",
    )
    parser.add_argument(
        "shader_path",
        nargs="?",
        help="Path to a GLSL fragment shader. Defaults to a built-in shader.",
    )
    parser.add_argument(
        "--terminal-mode",
        choices=("single", "multi"),
        default="single",
        help="Render into one terminal or a multi-pane launcher session.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Scale the u_time uniform passed to the shader.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("pixel", "quadrant", "octant"),
        default="quadrant",
    )
    parser.add_argument("--quadrant-cell-divisor", type=int, default=2)
    parser.add_argument("--octant-cell-width-divisor", type=int, default=2)
    parser.add_argument("--octant-cell-height-divisor", type=int, default=4)
    parser.add_argument("--launcher", default="./open_four_alacritty.sh")
    parser.add_argument("--session-dir")
    parser.add_argument(
        "--sync-mode",
        choices=("pane", "global", "off"),
        default="pane",
    )
    parser.add_argument("--cell-aspect", type=float, default=0.5)
    return parser.parse_args(argv)


def load_fragment_shader(shader_path: str | None) -> str:
    if not shader_path:
        return DEFAULT_FRAGMENT_SHADER
    return Path(shader_path).read_text(encoding="utf-8")


def _extract_version_line(fragment_shader: str) -> tuple[str, str]:
    stripped_shader = fragment_shader.lstrip()
    if not stripped_shader.startswith("#version"):
        return DEFAULT_GLSL_VERSION, fragment_shader

    first_line, _, remainder = stripped_shader.partition("\n")
    return first_line.strip(), remainder


def _declares_uniform(fragment_shader: str, uniform_name: str) -> bool:
    return (
        re.search(rf"\buniform\s+\w+\s+{re.escape(uniform_name)}\b", fragment_shader)
        is not None
    )


def _has_main_function(fragment_shader: str) -> bool:
    return re.search(r"\bvoid\s+main\s*\(", fragment_shader) is not None


def _has_main_image_function(fragment_shader: str) -> bool:
    return re.search(r"\bvoid\s+mainImage\s*\(", fragment_shader) is not None


def _has_fragment_output(fragment_shader: str) -> bool:
    return re.search(r"\bout\s+vec4\s+\w+\s*;", fragment_shader) is not None


def build_fragment_shader(fragment_shader: str) -> str:
    version_line, shader_body = _extract_version_line(fragment_shader)
    prelude_lines: list[str] = []

    for declaration in (
        ("u_resolution", "uniform vec2 u_resolution;"),
        ("u_time", "uniform float u_time;"),
        ("iResolution", "uniform vec3 iResolution;"),
        ("iTime", "uniform float iTime;"),
        ("iTimeDelta", "uniform float iTimeDelta;"),
        ("iFrame", "uniform int iFrame;"),
        ("iFrameRate", "uniform float iFrameRate;"),
        ("iMouse", "uniform vec4 iMouse;"),
    ):
        uniform_name, uniform_declaration = declaration
        if not _declares_uniform(shader_body, uniform_name):
            prelude_lines.append(uniform_declaration)

    processed_body = shader_body
    wrapper_lines: list[str] = []
    has_output = _has_fragment_output(shader_body)

    if _has_main_image_function(shader_body) and not _has_main_function(shader_body):
        if not has_output:
            prelude_lines.append("out vec4 f_color;")
        wrapper_lines.extend(
            (
                "void main() {",
                "    vec4 color = vec4(0.0);",
                "    mainImage(color, gl_FragCoord.xy);",
                "    f_color = color;",
                "}",
            )
        )
    elif "gl_FragColor" in shader_body and not has_output:
        prelude_lines.append("out vec4 f_color;")
        processed_body = shader_body.replace("gl_FragColor", "f_color")

    sections = [version_line, ""]
    if prelude_lines:
        sections.append("\n".join(prelude_lines))
        sections.append("")
    sections.append(processed_body.strip())
    if wrapper_lines:
        sections.extend(("", "\n".join(wrapper_lines)))

    return "\n".join(sections).strip() + "\n"


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        width=int(args.width),
        height=int(args.height),
        device=DEVICE,
        fps=float(args.fps),
        timing_enabled=False,
        timing_file="timing_shader.csv",
        render_mode=str(args.render_mode),
        quadrant_cell_divisor=int(args.quadrant_cell_divisor),
        octant_cell_width_divisor=int(args.octant_cell_width_divisor),
        octant_cell_height_divisor=int(args.octant_cell_height_divisor),
        diff_thresh=0,
        quant_mask=0xFF,
        run_color_diff_thresh=0,
        adaptive_quality=False,
        target_frame_bytes=0,
        frame_byte_buffer_frames=4,
        max_frame_bytes=0,
        write_chunk_size=2_097_152,
        use_rep=True,
        rep_min_run=4,
    )


def frame_generator(
    settings: SceneSettings,
) -> Generator[torch.Tensor, None, None]:
    runner: ShaderRunner | None = None
    start_time = time.perf_counter()

    try:
        runner = ShaderRunner(
            width=settings.width,
            height=settings.height,
            fragment_shader=load_fragment_shader(settings.shader_path),
        )
        while True:
            elapsed = (time.perf_counter() - start_time) * settings.time_scale
            frame = runner.render(elapsed)
            yield torch.as_tensor(frame, dtype=torch.uint8, device=DEVICE)
    finally:
        if runner is not None:
            runner.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = SceneSettings(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        shader_path=str(args.shader_path) if args.shader_path else None,
        time_scale=float(args.time_scale),
    )
    multi_pane_options = MultiPaneOptions(
        launcher=str(args.launcher),
        session_dir=str(args.session_dir) if args.session_dir else None,
        sync_mode=str(args.sync_mode),
        cell_aspect=float(args.cell_aspect),
    )

    try:
        render_with_terminal_mode(
            frame_generator(settings),
            build_config(args),
            terminal_mode=str(args.terminal_mode),
            multi_pane_options=multi_pane_options,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
