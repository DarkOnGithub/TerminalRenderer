# TerminalRenderer

TerminalRenderer is a GPU-accelerated terminal video renderer built with PyTorch + Triton.
It renders RGB frames as ANSI escape sequences and streams them directly to your terminal.
In practice, it can handle 720p-equivalent output and 1080p vertical content (hardware and terminal dependent).

Current render modes in this repo:
- `pixel`
- `quadrant`
- `octant` (experimental)

Notes:
- This project currently targets CUDA/NVIDIA workflows.
- `octant` relies on Unicode block-octant glyphs and is terminal/font dependent. In practice, keep `quadrant` as the safe default and use `octant` only when your terminal/font stack renders those symbols correctly.

## Demos

<img src="assets/output.gif" width="360" alt="TerminalRenderer demo" />

### Yaosobi - Idol

<video src="assets/idol.mp4" controls width="360"></video>

Source: https://youtu.be/7995X3B275g

### 3D Cube

<video src="assets/3d_cube.mp4" controls width="360"></video>

Source: https://youtu.be/7Zr2gqd8iPI

### Bad Apple

<video src="assets/bad_apple.mp4" controls width="360"></video>

Source: https://youtu.be/EVdXZdDUfWs

### Multi-pane example

<video src="assets/multi_pane.mp4" controls width="360"></video>

Source: https://www.youtube.com/watch?v=ftHQEd0QApc

## Requirements

- Python `>=3.13`
- NVIDIA GPU + CUDA-capable PyTorch build
- A fast terminal emulator (Alacritty/Kitty/WezTerm recommended)
- FFmpeg tools available in `PATH`:
  - `ffmpeg`
  - `ffprobe`
  - `ffplay`

### Alacritty preset (720p, quadrant, divisor 2)

Use this in your Alacritty config for a fullscreen, borderless setup tuned for 720p with `render_mode="quadrant"` and `quadrant_cell_divisor=2`:

```toml
[font]
size = 3.7

[font.offset]
x = 0
y = -5

[window]
startup_mode = "Fullscreen"
padding = { x = 0, y = 0 }
```

For a 720p quadrant multi-pane setup aimed at higher frame rate, use a tight grid like this:

```toml
[font]
size = 3.6

[font.offset]
x = 0
y = -2

[window]
dimensions = { columns = 320, lines = 90 }
decorations = "none"
startup_mode = "Windowed"
padding = { x = 0, y = 0 }
```

## Installation (uv)

1) Create a virtual environment with `uv` (activation is optional; `uv run` will auto-use `.venv`)

```bash
uv venv .venv
source .venv/bin/activate
```

2) Install a CUDA-enabled PyTorch build (pick your CUDA version)

```bash
# example (CUDA 12.8)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

3) Install project dependencies

```bash
uv pip install -e .
```

## Run

### Demo router

Use the shared CLI to choose the demo and whether it renders in one terminal or a multi-pane launcher session:

```bash
uv run terminal-renderer-demo video --terminal-mode single
uv run terminal-renderer-demo video --terminal-mode multi -- path/to/video.mp4 --stats-interval 0.5
uv run terminal-renderer-demo object --terminal-mode single
uv run terminal-renderer-demo object --terminal-mode multi -- --width 1280 --height 720
uv run terminal-renderer-demo shader --terminal-mode single
uv run terminal-renderer-demo shader --terminal-mode multi -- example/shaders/plasma.frag --width 1280 --height 720
```

Arguments after `--` are forwarded to the selected demo.

Multi-pane mode is useful when you want higher frame rate by splitting the output across multiple terminals.

### Video demo (`video_demo.py`)

```bash
uv run video_demo.py path/to/video.mp4
```

This demo:
- decodes frames with FFmpeg
- plays audio with `ffplay`
- renders into one terminal with `AnsiRenderer`

### 3D cube demo (`example/object.py`)

```bash
uv run python -m example.object --terminal-mode multi
```

This is a procedural GPU-rendered cube scene and it can now render to either a single terminal or the reusable multi-pane path.

### Shader demo (`example/shader.py`)

```bash
uv run python -m example.shader
uv run python -m example.shader example/shaders/plasma.frag --terminal-mode multi
```

This demo renders an offscreen GLSL fragment shader with `moderngl`, reads the result back as an `H x W x 3` RGB frame, and feeds that frame into TerminalRenderer.

Shader files can be either regular GLSL fragment shaders or simple Shadertoy-style shaders. The wrapper provides these uniforms when referenced:

```glsl
uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform float iFrameRate;
uniform vec4 iMouse;
```

It also supports shaders that define `mainImage(out vec4 fragColor, in vec2 fragCoord)` and shaders that write to `gl_FragColor`.

### Timing analysis

If you enable timing in config, analyze the CSV with:

```bash
uv run analyze_timing.py timing_object.csv 1
```

## Minimal API usage

Single terminal:

```python
import torch
from src.config import Config
from src.terminal_router import render_single_terminal

def frame_generator():
    while True:
        # H x W x 3, uint8, device should match config.device
        yield torch.zeros((720, 1280, 3), dtype=torch.uint8, device=torch.device("cuda"))

cfg = Config(width=1280, height=720, device=torch.device("cuda"), render_mode="pixel")
render_single_terminal(frame_generator(), cfg)
```

Reusable multi-pane rendering from any frame source:

```python
import torch
from src.config import Config
from src.multi_pane import MultiPaneOptions
from src.terminal_router import render_with_terminal_mode

def frame_generator():
    while True:
        yield torch.zeros((720, 1280, 3), dtype=torch.uint8, device=torch.device("cuda"))

cfg = Config(width=1280, height=720, device=torch.device("cuda"), render_mode="quadrant")
options = MultiPaneOptions(
    launcher="./open_four_alacritty.sh",
    sync_mode="pane",
    stats_interval=0.5,
)

render_with_terminal_mode(frame_generator(), cfg, terminal_mode="multi", multi_pane_options=options)
```

Set `stats_interval` to print reusable multi-pane runtime stats to stderr every N seconds. Use `0` to disable them.

## Project layout

- `src/ansi_renderer.py`: render loop, pacing, buffering, output writing
- `src/frame_processing.py`: resize, diffing, and mode-specific preprocessing
- `src/ansi_generator.py`: Triton kernels and ANSI sequence generation
- `src/config.py`: runtime configuration and ANSI constants
- `cli.py`: demo router for single-terminal vs multi-pane playback
- `video_demo.py`: single-terminal video playback demo
- `example/object.py`: procedural cube demo
- `example/shader.py`: offscreen GLSL shader demo via ModernGL
