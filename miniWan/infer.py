"""CLI for Wan2.2 TI2V-5B text-to-video and image-to-video inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from model import TI2VRequest, build_inference_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ti2v.mp4"))
    parser.add_argument("--image", type=Path, help="Optional image for image-to-video mode")
    parser.add_argument("--size", default="1280*720")
    parser.add_argument("--frames", type=int, default=121, help="Must be 4n+1")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--no-dtype-conversion", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the official command only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = TI2VRequest(
        prompt=args.prompt,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        image_path=args.image,
        size=args.size,
        frame_num=args.frames,
        seed=args.seed,
        sample_steps=args.steps,
        offload_model=not args.no_offload,
        convert_model_dtype=not args.no_dtype_conversion,
        t5_cpu=args.t5_cpu,
    )
    command = build_inference_command(request)
    print(shlex.join(command))
    if args.dry_run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
