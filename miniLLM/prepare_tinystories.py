"""Convert TinyStories parquet files into contiguous uint16 token files."""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_stories(paths: Iterable[Path]):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from error
    for path in paths:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(columns=["text"], batch_size=4096):
            for text in record_batch.column(0).to_pylist():
                if isinstance(text, str) and text.strip():
                    yield text.strip()


def write_tokens(paths: list[Path], output: Path, tokenizer_model: Path) -> int:
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError("sentencepiece is required: pip install sentencepiece") from error
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    if not 0 < processor.vocab_size() <= np.iinfo(np.uint16).max:
        raise ValueError(f"tokenizer vocabulary must fit uint16, got {processor.vocab_size()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("wb") as handle:
        for index, story in enumerate(iter_stories(paths), start=1):
            token_ids = [processor.bos_id(), *processor.encode(story)]
            token_array = np.asarray(token_ids, dtype=np.uint16)
            handle.write(token_array.tobytes())
            total += token_array.size
            if index % 100_000 == 0:
                print(f"{output.name}: stories={index:,} tokens={total:,}", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing TinyStories parquet files")
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train_paths = sorted(args.data_dir.glob("train-*.parquet"))
    val_paths = sorted(args.data_dir.glob("validation-*.parquet"))
    if not train_paths or not val_paths:
        raise FileNotFoundError("expected train-*.parquet and validation-*.parquet under data-dir")
    if not args.tokenizer_model.is_file():
        raise FileNotFoundError(args.tokenizer_model)
    train_tokens = write_tokens(train_paths, args.out_dir / "train.bin", args.tokenizer_model)
    val_tokens = write_tokens(val_paths, args.out_dir / "val.bin", args.tokenizer_model)
    print(f"complete: train_tokens={train_tokens:,}; val_tokens={val_tokens:,}")


if __name__ == "__main__":
    main()
