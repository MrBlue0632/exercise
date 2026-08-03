#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Projects/miniLLM"
ENV_DIR="$ROOT/.conda"
PYTHON="$ENV_DIR/bin/python"
LOG_DIR="$ROOT/logs"
DATA_DIR="$HOME/Projects/Datasets/TinyStories/data"
TOKENS_DIR="$ROOT/data/tinystories_llama2"

mkdir -p "$LOG_DIR"
if [[ ! -x "$PYTHON" ]]; then
  "$HOME/anaconda3/bin/conda" create --prefix "$ENV_DIR" --clone "$HOME/anaconda3/envs/Evo1" -y
fi

"$PYTHON" -m pip install sentencepiece
"$PYTHON" -c 'import torch, sentencepiece, pyarrow, swanlab; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

if [[ ! -s "$TOKENS_DIR/train.bin" || ! -s "$TOKENS_DIR/val.bin" ]]; then
  "$PYTHON" "$ROOT/prepare_tinystories.py" \
    --data-dir "$DATA_DIR" \
    --tokenizer-model "$ROOT/tokenizer.model" \
    --out-dir "$TOKENS_DIR"
fi

echo "setup complete"
