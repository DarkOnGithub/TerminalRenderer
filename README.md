# TerminalRenderer

TerminalRenderer is a GPU-accelerated terminal renderer built with PyTorch and Triton.
It converts RGB frames into ANSI escape sequences and can render videos, procedural scenes, and GLSL shaders into either one terminal or a four-pane terminal layout.

Current render modes:
- `pixel`
- `quadrant`

Notes:
- This project currently targets CUDA/NVIDIA workflows.
- Multi-pane mode uses four independent terminal windows by default.

## Demos

<img src="assets/output.gif" width="480" alt="TerminalRenderer demo" />

### Yaosobi - Idol

<video src="https://github.com/user-attachments/assets/1395815b-4413-4f64-a4ce-ef7f11bd425e" controls width="480"></video>

Source: https://youtu.be/7995X3B275g

### 3D Cube

<video src="https://github.com/user-attachments/assets/0991a9a2-b65a-462a-bd22-da86c4ae1fe9" controls width="480"></video>

Source: https://youtu.be/7Zr2gqd8iPI

### Bad Apple

<video src="https://github.com/user-attachments/assets/3c7ba2d2-8e33-474b-ac3c-54801b48787e" controls width="480"></video>

Source: https://youtu.be/EVdXZdDUfWs

### Tidal wave (multi-pane)

<video src="https://github.com/user-attachments/assets/0b8ea27f-6c64-4cdb-b61c-fa0743d6de2e" controls width="360"></video>

Source: https://www.youtube.com/watch?v=ftHQEd0QApc

## Requirements

- Python `>=3.13`
- NVIDIA GPU + CUDA-capable PyTorch build
- A fast terminal emulator, such as Alacritty, Kitty, or WezTerm
- FFmpeg tools available in `PATH`: `ffmpeg`, `ffprobe`, `ffplay`
Shader rendering also uses `moderngl` and `glcontext`, which are included in the project dependencies.

## Installation

```bash
uv venv .venv
source .venv/bin/activate

# Example for CUDA 12.8. Pick the build that matches your system.
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## CLI

`main.py` is the user entry point. The installed command is:

```bash
uv run terminal-renderer --help
```

Choose what to render with a subcommand:

```bash
uv run terminal-renderer video
uv run terminal-renderer video video.mp4
uv run terminal-renderer video --terminal-mode multi video.mp4

uv run terminal-renderer cube
uv run terminal-renderer cube --terminal-mode multi --width 1280 --height 720

uv run terminal-renderer shader
uv run terminal-renderer shader example/shaders/plasma.frag --terminal-mode multi
```

Common render options:
- `--terminal-mode single|multi`
- `--width` / `--height`
- `--render-mode pixel|quadrant`
- `--quadrant-cell-divisor`
- `--diff-thresh`
- `--run-color-diff-thresh`
- `--quant-mask`

Multi-pane options:
- `--launcher`
- `--session-dir`
- `--sync-mode pane|global|off`
- `--cell-aspect`
- `--stats-interval`

The default launcher is `./open_four_terminals.sh`. It creates four independent
terminal windows, so frame flushing can happen in parallel. The launcher avoids
compositor-specific placement APIs for window creation and works across X11 and
Wayland. Placement is manual by default. KDE Wayland placement is available
through KWin scripting when explicitly requested; other compositors currently
require manual tiling or compositor/window rules.
It uses the first available terminal from `TERMINAL`, Alacritty, Kitty, WezTerm,
Foot, Konsole, GNOME Terminal, or xterm. You can pass a different launcher with
`--launcher`.

Launcher placement options:

```bash
./open_four_terminals.sh --placement manual
./open_four_terminals.sh --placement kde-wayland
```

## Shader Support

Shader files can be regular GLSL fragment shaders or simple Shadertoy-style shaders. The wrapper provides these uniforms when referenced:

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

## Alacritty Presets

Single-terminal 720p quadrant preset:

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

Four-pane 720p quadrant preset:

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

## API Usage

Single terminal:

```python
import torch
from src.config import Config
from src.terminal_router import render_single_terminal

def frame_generator():
    while True:
        yield torch.zeros((720, 1280, 3), dtype=torch.uint8, device=torch.device("cuda"))

cfg = Config(width=1280, height=720, device=torch.device("cuda"), render_mode="pixel")
render_single_terminal(frame_generator(), cfg)
```

Reusable multi-pane rendering:

```python
import torch
from src.config import Config
from src.multi_pane import MultiPaneOptions
from src.terminal_router import render_with_terminal_mode

def frame_generator():
    while True:
        yield torch.zeros((720, 1280, 3), dtype=torch.uint8, device=torch.device("cuda"))

cfg = Config(width=1280, height=720, device=torch.device("cuda"), render_mode="quadrant")
options = MultiPaneOptions(launcher="./open_four_terminals.sh", sync_mode="pane")

render_with_terminal_mode(
    frame_generator(),
    cfg,
    terminal_mode="multi",
    multi_pane_options=options,
)
```

