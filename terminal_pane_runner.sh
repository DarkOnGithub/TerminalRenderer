#!/usr/bin/env bash

set -euo pipefail

size_file=${1:?missing size_file}
fifo_path=${2:?missing fifo_path}
debug_file=${3:-}

debug() {
    if [[ -n "$debug_file" ]]; then
        printf "%s\n" "$1" >&3
    fi
}

if [[ -n "$debug_file" ]]; then
    exec 3>>"$debug_file"
else
    exec 3>/dev/null
fi

debug "shell-start"
sleep 1.0

if read -r rows cols < <(stty size < /dev/tty); then
    debug "stty-size rows=$rows cols=$cols"
else
    rows=${LINES:-0}
    cols=${COLUMNS:-0}
    debug "stty-size-failed rows=$rows cols=$cols"
fi

python3 - "$size_file" "$rows" "$cols" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = int(sys.argv[2])
cols = int(sys.argv[3])
payload = {
    "lines": rows,
    "columns": cols,
    "width": 0,
    "height": 0,
}
path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
print(f"metadata-written path={path} payload={payload}", file=sys.stderr)
PY

debug "python-finished"

while true; do
    cat "$fifo_path"
done
