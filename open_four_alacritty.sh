#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_cmd alacritty
require_cmd qdbus
require_cmd python3

if [[ "${XDG_SESSION_TYPE:-}" != "wayland" ]]; then
    echo "This script is intended for Wayland sessions." >&2
    exit 1
fi

if [[ "${XDG_CURRENT_DESKTOP:-}" != *KDE* && "${DESKTOP_SESSION:-}" != *plasma* ]]; then
    echo "This script currently supports KDE Plasma on Wayland." >&2
    exit 1
fi

session_dir=""
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
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$session_dir" ]]; then
    session_dir=$(mktemp -d)
else
    mkdir -p "$session_dir"
fi

plugin_name="terminal_renderer_tile_four_alacritty_$$"
top_left_title="TerminalRenderer Top Left"
top_right_title="TerminalRenderer Top Right"
bottom_left_title="TerminalRenderer Bottom Left"
bottom_right_title="TerminalRenderer Bottom Right"
top_left_class="terminal-renderer-top-left"
top_right_class="terminal-renderer-top-right"
bottom_left_class="terminal-renderer-bottom-left"
bottom_right_class="terminal-renderer-bottom-right"

kwin_script=$(mktemp)
cleanup() {
    rm -f "$kwin_script"
}
trap cleanup EXIT

launch_window() {
    local title=$1
    local class_name=$2
    local size_file=$3
    local fifo_path=$4
    local log_file=$5
    local debug_file=$6

    env \
        alacritty \
        --title "$title" \
        --class "$class_name","$class_name" \
        --option window.decorations="None" \
        --option cursor.unfocused_hollow=false \
        -e "$script_dir/alacritty_pane_runner.sh" "$size_file" "$fifo_path" "$debug_file" \
        >"$log_file" 2>&1 &

    printf '%s\n' "$!"
}

cat >"$kwin_script" <<EOF
const TOP_LEFT_TITLE = ${top_left_title@Q};
const TOP_RIGHT_TITLE = ${top_right_title@Q};
const BOTTOM_LEFT_TITLE = ${bottom_left_title@Q};
const BOTTOM_RIGHT_TITLE = ${bottom_right_title@Q};
const PLUGIN_NAME = ${plugin_name@Q};
const PREVIOUS_ACTIVE_WINDOW = workspace.activeWindow;
const FIND_WINDOWS_INTERVAL_MS = 100;
const MAX_FIND_ATTEMPTS = 80;
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

function overlapWindow(window, position) {
    if (!window || !window.normalWindow) return;

    const geometry = window.frameGeometry;
    if (!geometry) return;

    if (position === "top-left") {
        geometry.x -= OUTLINE_OVERLAP_PX;
        geometry.y -= OUTLINE_OVERLAP_PX;
        geometry.width += OUTLINE_OVERLAP_PX * 2;
        geometry.height += OUTLINE_OVERLAP_PX * 2;
    } else if (position === "top-right") {
        geometry.x -= OUTLINE_OVERLAP_PX;
        geometry.y -= OUTLINE_OVERLAP_PX;
        geometry.width += OUTLINE_OVERLAP_PX * 2;
        geometry.height += OUTLINE_OVERLAP_PX * 2;
    } else if (position === "bottom-left") {
        geometry.x -= OUTLINE_OVERLAP_PX;
        geometry.y -= OUTLINE_OVERLAP_PX;
        geometry.width += OUTLINE_OVERLAP_PX * 2;
        geometry.height += OUTLINE_OVERLAP_PX * 2;
    } else if (position === "bottom-right") {
        geometry.x -= OUTLINE_OVERLAP_PX;
        geometry.y -= OUTLINE_OVERLAP_PX;
        geometry.width += OUTLINE_OVERLAP_PX * 2;
        geometry.height += OUTLINE_OVERLAP_PX * 2;
    }

    window.frameGeometry = geometry;
}

function onWindowActivated(window) {
    if (!focusGuardActive || !isRenderWindow(window)) {
        return;
    }

    const bounceTimer = new QTimer();
    bounceTimer.setSingleShot(true);
    bounceTimer.timeout.connect(() => {
        restorePreviousFocus();
    });
    bounceTimer.start(0);
}

function finishSetup(topLeftWindow, topRightWindow, bottomLeftWindow, bottomRightWindow) {
    if (completed) return;
    completed = true;

    const overlapTimer = new QTimer();
    overlapTimer.setSingleShot(true);
    overlapTimer.timeout.connect(() => {
        overlapWindow(topLeftWindow, "top-left");
        overlapWindow(topRightWindow, "top-right");
        overlapWindow(bottomLeftWindow, "bottom-left");
        overlapWindow(bottomRightWindow, "bottom-right");
    });
    overlapTimer.start(0);

    focusGuardActive = true;
    workspace.windowActivated.connect(onWindowActivated);
    restorePreviousFocus();

    const settleTimer = new QTimer();
    settleTimer.timeout.connect(() => {
        restorePreviousFocus();
    });
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

top_left_size_file="$session_dir/top_left.json"
top_right_size_file="$session_dir/top_right.json"
bottom_left_size_file="$session_dir/bottom_left.json"
bottom_right_size_file="$session_dir/bottom_right.json"
top_left_fifo="$session_dir/top_left.fifo"
top_right_fifo="$session_dir/top_right.fifo"
bottom_left_fifo="$session_dir/bottom_left.fifo"
bottom_right_fifo="$session_dir/bottom_right.fifo"
top_left_log="$session_dir/top_left.log"
top_right_log="$session_dir/top_right.log"
bottom_left_log="$session_dir/bottom_left.log"
bottom_right_log="$session_dir/bottom_right.log"
top_left_debug="$session_dir/top_left.debug"
top_right_debug="$session_dir/top_right.debug"
bottom_left_debug="$session_dir/bottom_left.debug"
bottom_right_debug="$session_dir/bottom_right.debug"
session_file="$session_dir/session.json"

rm -f "$top_left_fifo" "$top_right_fifo" "$bottom_left_fifo" "$bottom_right_fifo"
mkfifo "$top_left_fifo" "$top_right_fifo" "$bottom_left_fifo" "$bottom_right_fifo"

top_left_pid=$(launch_window "$top_left_title" "$top_left_class" "$top_left_size_file" "$top_left_fifo" "$top_left_log" "$top_left_debug")
top_right_pid=$(launch_window "$top_right_title" "$top_right_class" "$top_right_size_file" "$top_right_fifo" "$top_right_log" "$top_right_debug")
bottom_left_pid=$(launch_window "$bottom_left_title" "$bottom_left_class" "$bottom_left_size_file" "$bottom_left_fifo" "$bottom_left_log" "$bottom_left_debug")
bottom_right_pid=$(launch_window "$bottom_right_title" "$bottom_right_class" "$bottom_right_size_file" "$bottom_right_fifo" "$bottom_right_log" "$bottom_right_debug")

sleep 1.2

qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$plugin_name" >/dev/null 2>&1 || true
qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$kwin_script" "$plugin_name" >/dev/null
qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.start >/dev/null

for label in top_left top_right bottom_left bottom_right; do
    file="$session_dir/${label}.json"
    for _ in $(seq 1 50); do
        [[ -s "$file" ]] && break
        sleep 0.1
    done
    if [[ ! -s "$file" ]]; then
        log_file="$session_dir/${label}.log"
        echo "Timed out waiting for $label terminal size metadata." >&2
        echo "Session dir: $session_dir" >&2
        if [[ -s "$log_file" ]]; then
            echo "Last log lines from $log_file:" >&2
            tail -n 20 "$log_file" >&2 || true
        fi
        debug_file="$session_dir/${label}.debug"
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
    ("top_left", ${top_left_title@Q}, ${top_left_class@Q}, Path(${top_left_size_file@Q}), Path(${top_left_fifo@Q}), int(${top_left_pid@Q})),
    ("top_right", ${top_right_title@Q}, ${top_right_class@Q}, Path(${top_right_size_file@Q}), Path(${top_right_fifo@Q}), int(${top_right_pid@Q})),
    ("bottom_left", ${bottom_left_title@Q}, ${bottom_left_class@Q}, Path(${bottom_left_size_file@Q}), Path(${bottom_left_fifo@Q}), int(${bottom_left_pid@Q})),
    ("bottom_right", ${bottom_right_title@Q}, ${bottom_right_class@Q}, Path(${bottom_right_size_file@Q}), Path(${bottom_right_fifo@Q}), int(${bottom_right_pid@Q})),
]

panes = []
for pane_id, title, class_name, size_path, fifo_path, pid in entries:
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
    "layout": "4-pane",
    "session_dir": str(session_dir),
    "session_file": str(Path(${session_file@Q})),
    "panes": panes,
}

Path(${session_file@Q}).write_text(json.dumps(payload, indent=2) + "\n")

print(f"Session ready: {payload['session_file']}")
print(f"Session dir: {payload['session_dir']}")
print()
print(f"{'Position':<15} | {'Grid (Chars)':<15} | {'Window (Pixels)':<15} | FIFO")
print("-" * 96)
for pane in panes:
    grid = f"{pane['columns']}x{pane['lines']}"
    pixels = f"{pane['width']}x{pane['height']}"
    print(f"{pane['id']:<15} | {grid:<15} | {pixels:<15} | {pane['fifo']}")
PY
