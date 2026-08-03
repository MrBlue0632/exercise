#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Projects/miniLLM"
PYTHON="$ROOT/.conda/bin/python"
RAW_DATA="$HOME/Projects/Datasets/TinyStories/data"
TOK_DIR="$ROOT/data/tinystories_tok512"
TOKENIZER_PREFIX="$ROOT/data/tok512"

mkdir -p "$ROOT/logs" "$TOK_DIR"
if [[ ! -s "$TOKENIZER_PREFIX.model" ]]; then
  "$PYTHON" "$ROOT/train_tokenizer.py" --data-dir "$RAW_DATA" --out-prefix "$TOKENIZER_PREFIX" --vocab-size 512
fi
if [[ ! -s "$TOK_DIR/train.bin" || ! -s "$TOK_DIR/val.bin" ]]; then
  "$PYTHON" "$ROOT/prepare_tinystories.py" --data-dir "$RAW_DATA" --tokenizer-model "$TOKENIZER_PREFIX.model" --out-dir "$TOK_DIR"
fi
