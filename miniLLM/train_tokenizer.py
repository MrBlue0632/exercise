"""Train a TinyStories SentencePiece BPE tokenizer, matching llama2.c settings."""

import argparse
import os
from pathlib import Path

from prepare_tinystories import iter_stories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--max-stories", type=int, default=300_000)
    args = parser.parse_args()
    if args.vocab_size < 512:
        raise ValueError("vocab-size must be at least 512 when byte fallback is enabled")
    train_paths = sorted(args.data_dir.glob("train-*.parquet"))
    if not train_paths:
        raise FileNotFoundError("expected train-*.parquet under data-dir")
    text_path = args.out_prefix.with_suffix(".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("w", encoding="utf-8") as handle:
        for count, story in enumerate(iter_stories(train_paths), start=1):
            handle.write(story + "\n")
            if count >= args.max_stories:
                break
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError("sentencepiece is required") from error
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(args.out_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=" ⁇ ",
        normalization_rule_name="identity",
    )
    print(f"saved tokenizer to {args.out_prefix}.model")


if __name__ == "__main__":
    main()
