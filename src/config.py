import sys
import torch
from dataclasses import dataclass

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESC_VALS = (27, 91)
SEP_VAL = 59
CUR_END_VAL = 72
FG_COL_PREF_VALS = (27, 91, 51, 56, 59, 50, 59)  # \e[38;2;
COL_PREF_VALS = (27, 91, 52, 56, 59, 50, 59)  # \e[48;2;
COL_SUF_VALS = (109, 32)
RESET_VALS = (27, 91, 48, 109)
CUR_UP_PREFIX = (27, 91)  # \e[
CUR_DOWN_PREFIX = (27, 91)  # \e[
CUR_RIGHT_PREFIX = (27, 91)  # \e[
CUR_LEFT_PREFIX = (27, 91)  # \e[
CUR_UP_SUFFIX = (65,)  # A
CUR_DOWN_SUFFIX = (66,)  # B
CUR_RIGHT_SUFFIX = (67,)  # C
CUR_LEFT_SUFFIX = (68,)  # D
HIDE_CURSOR = b"\033[?25l"
SHOW_CURSOR = b"\033[?25h"
CLEAR_SCREEN = b"\033[2J"
ENABLE_ALT_BUFFER = b"\033[?1049h"
DISABLE_ALT_BUFFER = b"\033[?1049l"
SYNC_OUTPUT_BEGIN = b"\033[?2026h"
SYNC_OUTPUT_END = b"\033[?2026l"
CELL_ASPECT = 0.5
QUADRANT_GLYPHS = (
    " ",
    "▘",
    "▝",
    "▀",
    "▖",
    "▌",
    "▞",
    "▛",
    "▗",
    "▚",
    "▐",
    "▜",
    "▄",
    "▙",
    "▟",
    "█",
)


@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    fps: float = 30.0
    audio_path: str = None
    diff_thresh: int = 0
    quant_mask: int = 0xFF
    run_color_diff_thresh: int = 0
    relative_cursor_moves: bool = True
    use_rep: bool = False
    rep_min_run: int = 12
    output_fd: int = sys.stdout.fileno()
    sync_output: bool = True
    audio_delay: float = 0.0

    render_mode: str = "pixel"
    quadrant_cell_divisor: int = 2
