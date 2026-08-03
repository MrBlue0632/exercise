#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Projects/miniLLM"
PYTHON="$ROOT/.conda/bin/python"
TOKENS_DIR="$ROOT/data/tinystories_llama2"

cd "$ROOT"
exec "$PYTHON" sweep_15m.py \
  --train-data "$TOKENS_DIR/train.bin" \
  --val-data "$TOKENS_DIR/val.bin" \
  --out-root "$ROOT/out/sweep_15m_phase_a" \
  --steps 10000 \
  --swanlab-mode online
