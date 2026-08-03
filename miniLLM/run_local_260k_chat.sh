#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-local-test"
PYTHON="$VENV/bin/python"

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
if ! "$PYTHON" -c 'import sentencepiece' >/dev/null 2>&1; then
  "$PYTHON" -m pip install sentencepiece
fi

exec "$PYTHON" "$ROOT/chat.py" \
  --checkpoint "$ROOT/checkpoints/reference_models/260K.pt" \
  --tokenizer "$ROOT/checkpoints/reference_models/tok512.model" \
  --device cpu
