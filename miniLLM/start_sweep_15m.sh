#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Projects/miniLLM"
mkdir -p "$ROOT/logs"
cd "$ROOT"
export PYTHONUNBUFFERED=1
exec ./launch_sweep_15m.sh >> "$ROOT/logs/sweep_15m_phase_a.log" 2>&1
