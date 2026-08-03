"""Small, dependency-free adapter around Wan2.2 TI2V-5B.

This module deliberately delegates model loading and sampling to the official
Wan2.2 checkout in ``ref/Wan2.2``.  Keeping the adapter thin avoids copying or
forking the upstream implementation while giving this project a stable local
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "ref" / "Wan2.2"


@dataclass(frozen=True)
class TI2VRequest:
    """Inputs accepted by the Wan2.2 TI2V-5B inference entry point."""

    prompt: str
    checkpoint_dir: Path
    output_path: Path
    image_path: Optional[Path] = None
    size: str = "1280*720"
    frame_num: int = 121
    seed: int = -1
    sample_steps: Optional[int] = None
    offload_model: bool = True
    convert_model_dtype: bool = True
    t5_cpu: bool = False

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.frame_num < 1 or self.frame_num % 4 != 1:
            raise ValueError("frame_num must be of the form 4n+1 (for example 121)")
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"TI2V checkpoint directory not found: {self.checkpoint_dir}"
            )
        if self.image_path is not None and not self.image_path.is_file():
            raise FileNotFoundError(f"input image not found: {self.image_path}")


def upstream_root(path: Path = DEFAULT_UPSTREAM_DIR) -> Path:
    """Return the official checkout, failing early with an actionable error."""

    path = path.resolve()
    if not (path / "generate.py").is_file():
        raise FileNotFoundError(
            "Wan2.2 source was not found. Expected ref/Wan2.2/generate.py; "
            "clone the official repository there first."
        )
    return path


def build_inference_command(
    request: TI2VRequest,
    *,
    python: str = sys.executable,
    source_dir: Path = DEFAULT_UPSTREAM_DIR,
) -> list[str]:
    """Build the official command without executing it."""

    request.validate()
    source_dir = upstream_root(source_dir)
    command = [
        python,
        str(source_dir / "generate.py"),
        "--task",
        "ti2v-5B",
        "--ckpt_dir",
        str(request.checkpoint_dir.resolve()),
        "--prompt",
        request.prompt,
        "--size",
        request.size,
        "--frame_num",
        str(request.frame_num),
        "--base_seed",
        str(request.seed),
        "--offload_model",
        str(request.offload_model),
        "--save_file",
        str(request.output_path.resolve()),
    ]
    if request.convert_model_dtype:
        command.append("--convert_model_dtype")
    if request.t5_cpu:
        command.append("--t5_cpu")
    if request.image_path is not None:
        command.extend(["--image", str(request.image_path.resolve())])
    if request.sample_steps is not None:
        command.extend(["--sample_steps", str(request.sample_steps)])
    return command
