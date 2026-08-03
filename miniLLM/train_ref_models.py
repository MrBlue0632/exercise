"""Train the four TinyStories Llama configurations published by llama2.c."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceRun:
    size: str
    batch_size: int
    grad_accum: int
    learning_rate: float
    steps: int
    data_dir: str
    extra_args: tuple[str, ...]

    @property
    def experiment_name(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("e-0", "e-")
        return f"{self.size} - adamw - bs{self.batch_size} - lr{lr}"


# 260K, 15M and 110M use their published launch arguments. 42M has a published
# architecture but no full launch command; its batch/LR follow the repository's
# medium-model training guide.
REFERENCE_RUNS = (
    ReferenceRun(
        "260K", 128, 1, 1e-3, 100_000, "data/tinystories_tok512",
        ("--seq-len", "512", "--dim", "64", "--n-layers", "5", "--n-heads", "8", "--n-kv-heads", "4", "--vocab-size", "512", "--multiple-of", "4", "--dropout", "0.05", "--weight-decay", "0.01", "--beta2", "0.99", "--eval-interval", "2000", "--eval-batches", "100", "--compile"),
    ),
    ReferenceRun(
        "15M", 128, 4, 5e-4, 100_000, "data/tinystories_llama2",
        ("--seq-len", "256", "--dim", "288", "--n-layers", "6", "--n-heads", "6", "--n-kv-heads", "6", "--vocab-size", "32000", "--multiple-of", "32", "--dropout", "0.0", "--weight-decay", "0.1", "--compile"),
    ),
    ReferenceRun(
        "42M", 16, 8, 3e-4, 100_000, "data/tinystories_llama2",
        ("--seq-len", "1024", "--dim", "512", "--n-layers", "8", "--n-heads", "8", "--n-kv-heads", "8", "--vocab-size", "32000", "--multiple-of", "32", "--dropout", "0.1", "--weight-decay", "0.1", "--compile"),
    ),
    ReferenceRun(
        "110M", 16, 8, 4e-4, 200_000, "data/tinystories_llama2",
        ("--seq-len", "1024", "--dim", "768", "--n-layers", "12", "--n-heads", "12", "--n-kv-heads", "12", "--vocab-size", "32000", "--multiple-of", "32", "--dropout", "0.1", "--weight-decay", "0.1", "--compile"),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    for run in REFERENCE_RUNS:
        data_dir = args.root / run.data_dir
        command = [
            sys.executable, "train.py",
            "--train-data", str(data_dir / "train.bin"),
            "--val-data", str(data_dir / "val.bin"),
            "--out-dir", str(args.root / "out" / "reference_models" / run.size),
            "--steps", str(run.steps),
            "--batch-size", str(run.batch_size),
            "--grad-accum", str(run.grad_accum),
            "--learning-rate", str(run.learning_rate),
            "--warmup-steps", "1000",
            "--swanlab-project", "mini-llama",
            "--swanlab-experiment-name", run.experiment_name,
            "--swanlab-mode", args.swanlab_mode,
            *run.extra_args,
        ]
        print(f"starting: {run.experiment_name}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
