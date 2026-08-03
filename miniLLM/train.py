"""Train a TinyStories Llama-style model from pre-tokenized uint16 files."""

import argparse
import inspect
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer


class TokenMemmap:
    """Random fixed-length language-model samples from a uint16 token file."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"{path} is not a uint16 token file")
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_tokens <= seq_len:
            raise ValueError(f"{self.path} contains too few tokens for seq_len={seq_len}")
        starts = rng.integers(0, self.num_tokens - seq_len - 1, size=batch_size)
        chunks = np.stack([self.tokens[start : start + seq_len + 1] for start in starts])
        x = torch.from_numpy(chunks[:, :-1].astype(np.int64, copy=False))
        y = torch.from_numpy(chunks[:, 1:].astype(np.int64, copy=False))
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def configure_optimizers(
    model: Transformer,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    decay_params = [param for param in trainable_params if param.dim() >= 2]
    no_decay_params = [param for param in trainable_params if param.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = {"fused": True} if fused_available and device_type == "cuda" else {}
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


def learning_rate_at_step(step: int, args: argparse.Namespace) -> float:
    if not args.decay_lr:
        return args.learning_rate
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    if step >= args.steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.inference_mode()
def estimate_loss(
    model: Transformer,
    data: TokenMemmap,
    args: argparse.Namespace,
    device: torch.device,
    autocast_context,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = data.batch(args.batch_size, args.seq_len, device, rng)
        with autocast_context():
            model(x, y)
        assert model.last_loss is not None
        losses.append(model.last_loss.item())
    return float(np.mean(losses))


def init_swanlab(cli_args: argparse.Namespace, model_args: ModelArgs, tokens_per_update: int):
    if cli_args.swanlab_mode == "disabled":
        return None
    try:
        import swanlab
    except ImportError as error:
        raise RuntimeError("SwanLab is not installed. Install it with: pip install swanlab") from error
    return swanlab.init(
        project=cli_args.swanlab_project,
        experiment_name=cli_args.swanlab_experiment_name,
        mode=cli_args.swanlab_mode,
        config={
            "model": asdict(model_args),
            "optimizer": "adamw",
            "batch_size": cli_args.batch_size,
            "gradient_accumulation_steps": cli_args.grad_accum,
            "tokens_per_update": tokens_per_update,
            "steps": cli_args.steps,
            "learning_rate": cli_args.learning_rate,
            "min_lr": cli_args.min_lr,
            "warmup_steps": cli_args.warmup_steps,
            "weight_decay": cli_args.weight_decay,
            "beta1": cli_args.beta1,
            "beta2": cli_args.beta2,
            "dropout": cli_args.dropout,
            "seed": cli_args.seed,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="uint16 training token file")
    parser.add_argument("--val-data", type=Path, required=True, help="uint16 validation token file")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128, help="micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=288)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-kv-heads", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--multiple-of", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--decay-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--swanlab-project", default="mini-llama")
    parser.add_argument("--swanlab-experiment-name", required=True)
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.grad_accum < 1 or args.steps < 1:
        raise ValueError("batch-size, grad-accum, and steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_data = TokenMemmap(args.train_data)
    val_data = TokenMemmap(args.val_data)
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        multiple_of=args.multiple_of,
        dropout=args.dropout,
    )
    model = Transformer(model_args).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = configure_optimizers(model, args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device.type)
    tokens_per_update = args.batch_size * args.grad_accum * args.seq_len
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    autocast_context = (lambda: torch.autocast(device_type=device.type, dtype=dtype)) if device.type == "cuda" else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.dtype == "float16")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run = init_swanlab(args, model_args, tokens_per_update)
    best_val_loss = float("inf")
    started = time.perf_counter()
    try:
        print(f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        print(f"tokens_per_update={tokens_per_update:,}; train_tokens={train_data.num_tokens:,}; val_tokens={val_data.num_tokens:,}", flush=True)
        for step in range(1, args.steps + 1):
            lr = learning_rate_at_step(step - 1, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = train_data.batch(args.batch_size, args.seq_len, device, train_rng)
                with autocast_context():
                    model(x, y)
                    assert model.last_loss is not None
                    loss = model.last_loss / args.grad_accum
                scaler.scale(loss).backward()
                total_loss += loss.detach().item()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            else:
                grad_norm = 0.0
            scaler.step(optimizer)
            scaler.update()

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.perf_counter() - started
                throughput = step * tokens_per_update / max(elapsed, 1e-9)
                metrics = {
                    "train/loss": total_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens": step * tokens_per_update,
                    "system/tokens_per_second": throughput,
                }
                print(f"step={step}/{args.steps} loss={total_loss:.4f} lr={lr:.3e} tok/s={throughput:,.0f}", flush=True)
                if run is not None:
                    run.log(metrics, step=step)

            is_eval = step % args.eval_interval == 0 or step == args.steps
            if is_eval:
                val_loss = estimate_loss(model, val_data, args, device, autocast_context, val_rng)
                print(f"step={step}/{args.steps} val_loss={val_loss:.4f}", flush=True)
                if run is not None:
                    run.log({"val/loss": val_loss}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model_args": asdict(model_args), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_val_loss": best_val_loss, "train_args": vars(args)}, args.out_dir / "best.pt")

            if step % args.checkpoint_interval == 0 or step == args.steps:
                torch.save({"model_args": asdict(model_args), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_val_loss": best_val_loss, "train_args": vars(args)}, args.out_dir / "last.pt")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
