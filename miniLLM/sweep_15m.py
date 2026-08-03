"""Launch the first-stage 15M AdamW sweep sequentially."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    learning_rate: float
    dropout: float
    weight_decay: float

    @property
    def name(self) -> str:
        optimizer = f"adamw-wd{self.weight_decay:g}-do{self.dropout:g}"
        lr = f"{self.learning_rate:.0e}".replace("e-0", "e-")
        return f"15M - {optimizer} - bs128 - lr{lr}"


EXPERIMENTS = [
    Experiment(lr, dropout, weight_decay)
    for lr in (3e-4, 5e-4, 7e-4)
    for dropout in (0.0, 0.1)
    for weight_decay in (0.03, 0.1)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("out/sweep_15m_phase_a"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        run_dir = args.out_root / f"run_{index:02d}"
        command = [
            sys.executable, "train.py", "--train-data", str(args.train_data), "--val-data", str(args.val_data),
            "--out-dir", str(run_dir), "--steps", str(args.steps), "--batch-size", "128", "--grad-accum", "4",
            "--seq-len", "256", "--dim", "288", "--n-layers", "6", "--n-heads", "6", "--n-kv-heads", "6",
            "--multiple-of", "32", "--learning-rate", str(experiment.learning_rate),
            "--weight-decay", str(experiment.weight_decay), "--dropout", str(experiment.dropout),
            "--swanlab-project", "mini-llama", "--swanlab-experiment-name", experiment.name,
            "--swanlab-mode", args.swanlab_mode,
        ]
        print(f"[{index}/{len(EXPERIMENTS)}] {experiment.name}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
