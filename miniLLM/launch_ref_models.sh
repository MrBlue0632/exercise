#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Projects/miniLLM"
mkdir -p "$ROOT/logs"
cd "$ROOT"
export PYTHONUNBUFFERED=1
./prepare_ref_260k.sh >> "$ROOT/logs/prepare_ref_260k.log" 2>&1
exec "$ROOT/.conda/bin/python" train_ref_models.py --root "$ROOT" --swanlab-mode online >> "$ROOT/logs/reference_models.log" 2>&1
