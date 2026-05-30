#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat >&2 <<'EOF'
Usage: open_four_terminals.sh [--session-dir DIR] [--terminal CMD] [--placement MODE]

Creates four independent terminal windows for TerminalRenderer and writes
session.json. This avoids compositor-specific placement APIs; use your window
manager/compositor rules if you want automatic tiling.

Options:
  --session-dir DIR  Directory for FIFOs, logs, and session.json.
  --terminal CMD     Terminal command to launch. Defaults to $TERMINAL, then
                     alacritty, kitty, wezterm, foot, konsole, gnome-terminal, xterm.
    --placement MODE   manual or kde-wayland. Default: $PLACEMENT or manual.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

session_dir=""
terminal_cmd="${TERMINAL:-}"
placement="${PLACEMENT:-manual}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --session-dir" >&2
                exit 1
            fi
            session_dir=$2
            shift 2
            ;;
        --terminal)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --terminal" >&2
                exit 1
            fi
            terminal_cmd=$2
            shift 2
            ;;
        --placement)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --placement" >&2
                exit 1
            fi
            placement=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$placement" in
    kde-wayland|manual) ;;
    *)
        echo "Unsupported placement mode: $placement" >&2
        usage
        exit 1
        ;;
esac

if [[ -z "$terminal_cmd" ]]; then
    for candidate in alacritty kitty wezterm foot konsole gnome-terminal xterm; do
        if command -v "$candidate" >/dev/null 2>&1; then
            terminal_cmd=$candidate
            break
        fi
    done
fi

if [[ -z "$terminal_cmd" ]]; then
    echo "No terminal command found. Set TERMINAL or pass --terminal." >&2
    exit 1
fi

require_cmd "$terminal_cmd"
require_cmd python3

if [[ -z "$session_dir" ]]; then
    session_dir=$(mktemp -d)
else
    mkdir -p "$session_dir"
fi

session_file="$session_dir/session.json"

declare -A titles=(
    [top_left]="TerminalRenderer Top Left"
    [top_right]="TerminalRenderer Top Right"
    [bottom_left]="TerminalRenderer Bottom Left"
    [bottom_right]="TerminalRenderer Bottom Right"
)

declare -A classes=(
    [top_left]="terminal-renderer-top-left"
    [top_right]="terminal-renderer-top-right"
    [bottom_left]="terminal-renderer-bottom-left"
    [bottom_right]="terminal-renderer-bottom-right"
)

launch_terminal() {
    local pane_id=$1
    local size_file=$2
    local fifo_path=$3
    local log_file=$4
    local debug_file=$5
    local title=${titles[$pane_id]}
    local class_name=${classes[$pane_id]}
    local terminal_name

    terminal_name=$(basename -- "$terminal_cmd")

    case "$terminal_name" in
        alacritty)
            "$terminal_cmd" \
                --title "$title" \
                --class "$class_name","$class_name" \
                --option window.decorations="None" \
                --option cursor.unfocused_hollow=false \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        kitty)
            "$terminal_cmd" \
                --title "$title" \
                --class "$class_name" \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        wezterm)
            "$terminal_cmd" start \
                --class "$class_name" \
                --cwd "$PWD" \
                -- "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        foot)
            "$terminal_cmd" \
                --title "$title" \
                --app-id "$class_name" \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        konsole)
            "$terminal_cmd" \
                --title "$title" \
                --workdir "$PWD" \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        gnome-terminal)
            "$terminal_cmd" \
                --title "$title" \
                -- "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        xterm)
            "$terminal_cmd" \
                -T "$title" \
                -class "$class_name" \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
        *)
            "$terminal_cmd" \
                -e "$script_dir/terminal_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
                >"$log_file" 2>&1 &
            ;;
    esac

    printf '%s\n' "$!"
}

can_place_kde_wayland() {
    [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || return 1
    [[ "${XDG_CURRENT_DESKTOP:-}" == *KDE* || "${DESKTOP_SESSION:-}" == *plasma* ]] || return 1
    command -v qdbus >/dev/null 2>&1 || return 1
}

place_kde_wayland() {
    local plugin_name="terminal_renderer_tile_four_${$}"
    local kwin_script

    kwin_script=$(mktemp)
    cat >"$kwin_script" <<EOF
const TOP_LEFT_TITLE = ${titles[top_left]@Q};
const TOP_RIGHT_TITLE = ${titles[top_right]@Q};
const BOTTOM_LEFT_TITLE = ${titles[bottom_left]@Q};
const BOTTOM_RIGHT_TITLE = ${titles[bottom_right]@Q};
const PLUGIN_NAME = ${plugin_name@Q};
const PREVIOUS_ACTIVE_WINDOW = workspace.activeWindow;
const FIND_WINDOWS_INTERVAL_MS = 50;
const MAX_FIND_ATTEMPTS = 100;
const OUTLINE_OVERLAP_PX = 6;
const FOCUS_GUARD_DURATION_MS = 5000;
const REFOCUS_INTERVAL_MS = 150;

let findAttempts = 0;
let completed = false;
let focusGuardActive = false;

function unloadSelf() {
    callDBus("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting", "unloadScript", PLUGIN_NAME);
}

function restorePreviousFocus() {
    if (PREVIOUS_ACTIVE_WINDOW && PREVIOUS_ACTIVE_WINDOW.normalWindow) {
        workspace.raiseWindow(PREVIOUS_ACTIVE_WINDOW);
        workspace.activeWindow = PREVIOUS_ACTIVE_WINDOW;
    }
}

function isRenderWindow(window) {
    if (!window || !window.caption) return false;
    return (
        window.caption.indexOf(TOP_LEFT_TITLE) !== -1 ||
        window.caption.indexOf(TOP_RIGHT_TITLE) !== -1 ||
        window.caption.indexOf(BOTTOM_LEFT_TITLE) !== -1 ||
        window.caption.indexOf(BOTTOM_RIGHT_TITLE) !== -1
    );
}

function findWindow(title) {
    const windows = workspace.stackingOrder;
    for (let i = windows.length - 1; i >= 0; --i) {
        const window = windows[i];
        if (window && window.normalWindow && window.caption && window.caption.indexOf(title) !== -1) {
            return window;
        }
    }
    return null;
}

function tile(window, position) {
    if (!window || !window.normalWindow) return;
    window.noBorder = true;
    workspace.activeWindow = window;
    if (position === "top-left") workspace.slotWindowQuickTileTopLeft();
    else if (position === "top-right") workspace.slotWindowQuickTileTopRight();
    else if (position === "bottom-left") workspace.slotWindowQuickTileBottomLeft();
    else if (position === "bottom-right") workspace.slotWindowQuickTileBottomRight();
}

function overlapWindow(window) {
    if (!window || !window.normalWindow) return;
    const geometry = window.frameGeometry;
    if (!geometry) return;
    geometry.x -= OUTLINE_OVERLAP_PX;
    geometry.y -= OUTLINE_OVERLAP_PX;
    geometry.width += OUTLINE_OVERLAP_PX * 2;
    geometry.height += OUTLINE_OVERLAP_PX * 2;
    window.frameGeometry = geometry;
}

function onWindowActivated(window) {
    if (!focusGuardActive || !isRenderWindow(window)) return;
    const bounceTimer = new QTimer();
    bounceTimer.setSingleShot(true);
    bounceTimer.timeout.connect(() => restorePreviousFocus());
    bounceTimer.start(0);
}

function finishSetup(topLeftWindow, topRightWindow, bottomLeftWindow, bottomRightWindow) {
    if (completed) return;
    completed = true;

    const overlapTimer = new QTimer();
    overlapTimer.setSingleShot(true);
    overlapTimer.timeout.connect(() => {
        overlapWindow(topLeftWindow);
        overlapWindow(topRightWindow);
        overlapWindow(bottomLeftWindow);
        overlapWindow(bottomRightWindow);
    });
    overlapTimer.start(0);

    focusGuardActive = true;
    workspace.windowActivated.connect(onWindowActivated);
    restorePreviousFocus();

    const settleTimer = new QTimer();
    settleTimer.timeout.connect(() => restorePreviousFocus());
    settleTimer.start(REFOCUS_INTERVAL_MS);

    const stopTimer = new QTimer();
    stopTimer.setSingleShot(true);
    stopTimer.timeout.connect(() => {
        focusGuardActive = false;
        settleTimer.stop();
        restorePreviousFocus();
        unloadSelf();
    });
    stopTimer.start(FOCUS_GUARD_DURATION_MS);
}

const findWindowsTimer = new QTimer();
findWindowsTimer.timeout.connect(() => {
    findAttempts += 1;

    const topLeftWindow = findWindow(TOP_LEFT_TITLE);
    const topRightWindow = findWindow(TOP_RIGHT_TITLE);
    const bottomLeftWindow = findWindow(BOTTOM_LEFT_TITLE);
    const bottomRightWindow = findWindow(BOTTOM_RIGHT_TITLE);

    if (topLeftWindow && topRightWindow && bottomLeftWindow && bottomRightWindow) {
        findWindowsTimer.stop();
        tile(topLeftWindow, "top-left");
        tile(topRightWindow, "top-right");
        tile(bottomLeftWindow, "bottom-left");
        tile(bottomRightWindow, "bottom-right");
        finishSetup(topLeftWindow, topRightWindow, bottomLeftWindow, bottomRightWindow);
        return;
    }

    if (findAttempts >= MAX_FIND_ATTEMPTS) {
        findWindowsTimer.stop();
        finishSetup(topLeftWindow, topRightWindow, bottomLeftWindow, bottomRightWindow);
    }
});
findWindowsTimer.start(FIND_WINDOWS_INTERVAL_MS);
EOF

    qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$plugin_name" >/dev/null 2>&1 || true
    qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$kwin_script" "$plugin_name" >/dev/null
    qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.start >/dev/null
    rm -f "$kwin_script"
}

place_windows() {
    case "$placement" in
        manual)
            echo "Placement is manual; tile the four TerminalRenderer windows yourself." >&2
            ;;
        kde-wayland)
            if ! can_place_kde_wayland; then
                echo "KDE Wayland placement requested, but this session does not support it." >&2
                echo "Tile the four TerminalRenderer windows manually or use --placement manual." >&2
                return
            fi
            place_kde_wayland
            ;;
    esac
}

for pane_id in top_left top_right bottom_left bottom_right; do
    rm -f "$session_dir/${pane_id}.fifo"
    mkfifo "$session_dir/${pane_id}.fifo"
done

top_left_pid=$(launch_terminal top_left "$session_dir/top_left.json" "$session_dir/top_left.fifo" "$session_dir/top_left.log" "$session_dir/top_left.debug")
top_right_pid=$(launch_terminal top_right "$session_dir/top_right.json" "$session_dir/top_right.fifo" "$session_dir/top_right.log" "$session_dir/top_right.debug")
bottom_left_pid=$(launch_terminal bottom_left "$session_dir/bottom_left.json" "$session_dir/bottom_left.fifo" "$session_dir/bottom_left.log" "$session_dir/bottom_left.debug")
bottom_right_pid=$(launch_terminal bottom_right "$session_dir/bottom_right.json" "$session_dir/bottom_right.fifo" "$session_dir/bottom_right.log" "$session_dir/bottom_right.debug")

place_windows

for pane_id in top_left top_right bottom_left bottom_right; do
    size_file="$session_dir/${pane_id}.json"
    for _ in $(seq 1 80); do
        [[ -s "$size_file" ]] && break
        sleep 0.1
    done
    if [[ ! -s "$size_file" ]]; then
        log_file="$session_dir/${pane_id}.log"
        debug_file="$session_dir/${pane_id}.debug"
        echo "Timed out waiting for $pane_id terminal size metadata." >&2
        echo "Session dir: $session_dir" >&2
        if [[ -s "$log_file" ]]; then
            echo "Last log lines from $log_file:" >&2
            tail -n 20 "$log_file" >&2 || true
        fi
        if [[ -s "$debug_file" ]]; then
            echo "Debug trace from $debug_file:" >&2
            tail -n 20 "$debug_file" >&2 || true
        fi
        exit 1
    fi
done

python3 - <<PY
import json
from pathlib import Path

session_dir = Path(${session_dir@Q})
entries = [
    ("top_left", ${titles[top_left]@Q}, ${classes[top_left]@Q}, int(${top_left_pid@Q})),
    ("top_right", ${titles[top_right]@Q}, ${classes[top_right]@Q}, int(${top_right_pid@Q})),
    ("bottom_left", ${titles[bottom_left]@Q}, ${classes[bottom_left]@Q}, int(${bottom_left_pid@Q})),
    ("bottom_right", ${titles[bottom_right]@Q}, ${classes[bottom_right]@Q}, int(${bottom_right_pid@Q})),
]

panes = []
for pane_id, title, class_name, pid in entries:
    size_path = session_dir / f"{pane_id}.json"
    fifo_path = session_dir / f"{pane_id}.fifo"
    size_data = json.loads(size_path.read_text())
    panes.append(
        {
            "id": pane_id,
            "pid": pid,
            "title": title,
            "class": class_name,
            "size_file": str(size_path),
            "fifo": str(fifo_path),
            **size_data,
        }
    )

payload = {
    "layout": "4-window",
    "launcher": "terminal-windows",
    "terminal": ${terminal_cmd@Q},
    "session_dir": str(session_dir),
    "session_file": str(Path(${session_file@Q})),
    "panes": panes,
}

Path(${session_file@Q}).write_text(json.dumps(payload, indent=2) + "\n")

print(f"Session ready: {payload['session_file']}")
print(f"Session dir: {payload['session_dir']}")
print(f"terminal: {payload['terminal']}")
print()
print(f"{'Position':<15} | {'Grid (Chars)':<15} | {'Window (Pixels)':<15} | FIFO")
print("-" * 96)
for pane in panes:
    grid = f"{pane['columns']}x{pane['lines']}"
    pixels = f"{pane['width']}x{pane['height']}"
    print(f"{pane['id']:<15} | {grid:<15} | {pixels:<15} | {pane['fifo']}")
PY
