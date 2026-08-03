"""Prepare and validate a TI2V fine-tuning dataset manifest.

Wan2.2's official repository publishes inference code and weights, not an
official TI2V-5B training loop.  This script intentionally prepares the
portable ``jsonl`` manifest that a chosen fine-tuning implementation (such as
DiffSynth-Studio) can consume, rather than pretending that upstream supports
training directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/ti2v_train_manifest.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    records: list[dict[str, str]] = []
    for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("video"), str) or not isinstance(record.get("prompt"), str):
            raise ValueError(f"line {line_number}: each row needs string fields 'video' and 'prompt'")
        video = args.dataset_root / record["video"]
        if not video.is_file():
            raise FileNotFoundError(f"line {line_number}: video not found: {video}")
        records.append({"video": str(video.resolve()), "prompt": record["prompt"]})

    if not records:
        raise ValueError("the manifest contains no training records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Validated {len(records)} records and wrote {args.output}")
    print("Use this manifest with a TI2V-compatible LoRA/full-training implementation; upstream Wan2.2 has no training CLI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
