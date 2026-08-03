#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-local-test"
PYTHON="$VENV/bin/python"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip
fi
if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi
if ! "$PYTHON" -c 'import numpy' >/dev/null 2>&1; then
  "$PYTHON" -m pip install numpy
fi

exec "$PYTHON" "$ROOT/infer.py" \
  --checkpoint "$ROOT/checkpoints/reference_models/260K.pt" \
  --prompt 1 10 20 30 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --device cpu
