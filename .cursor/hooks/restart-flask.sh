#!/usr/bin/env bash
# Cursor afterFileEdit：Agent 保存 backend/*.py 后自动重启本项目的 Flask（5001）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEBOUNCE_SEC="${FLASK_RESTART_DEBOUNCE_SEC:-2}"
MARKER="$ROOT/.cursor/.flask_last_restart"

input=$(cat)
path="$(printf '%s' "$input" | python3 -c "
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw) if raw.strip() else {}
    print(d.get('file_path', '') or '', end='')
except Exception:
    print('', end='')
")"

if [[ -z "$path" ]]; then exit 0; fi
if [[ "$path" != "$ROOT/backend/"* ]]; then exit 0; fi
if [[ "$path" != *.py ]]; then exit 0; fi

now=$(date +%s)
last=0
if [[ -f "$MARKER" ]]; then
  last=$(cat "$MARKER" 2>/dev/null || echo 0)
fi
if (( now - last < DEBOUNCE_SEC )); then
  exit 0
fi

PY=python3
if [[ -x "$ROOT/backend/.venv/bin/python" ]]; then
  PY="$ROOT/backend/.venv/bin/python"
fi

pids=$(lsof -nP -iTCP:5001 -sTCP:LISTEN -t 2>/dev/null || true)
if [[ -n "${pids:-}" ]]; then
  echo "$pids" | xargs kill 2>/dev/null || true
  sleep 0.4
fi

mkdir -p "$ROOT/.cursor"
cd "$ROOT/backend"
nohup "$PY" app.py >>"$ROOT/.cursor/flask-dev.log" 2>&1 &
disown 2>/dev/null || true
echo "$now" >"$MARKER"
exit 0
