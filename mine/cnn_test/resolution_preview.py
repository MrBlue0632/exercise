"""Compare ordinary RGB downsampling with high-frequency-enhanced downsampling."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "outputs" / "01_original_224.png"
OUTPUT_DIR = ROOT / "outputs" / "resolution_compression"


def rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def high_frequency_prefilter(image: Image.Image, amount: float = 1.35) -> Image.Image:
    """Boost local RGB residuals before downsampling so edges survive more clearly."""
    source = rgb_array(image)
    blurred = rgb_array(image.filter(ImageFilter.GaussianBlur(radius=1.15)))
    enhanced = np.clip(source + amount * (source - blurred), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.round(enhanced * 255.0)), mode="RGB")


def psnr(reference: Image.Image, reconstructed: Image.Image) -> float:
    ref = rgb_array(reference)
    rec = rgb_array(reconstructed)
    mse = float(np.mean((ref - rec) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * np.log10(mse)


def add_panel(axis, image: Image.Image, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def build_preview() -> dict[str, object]:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_PATH}. Run cnn_preview.py before this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = Image.open(SOURCE_PATH).convert("RGB").resize(
        (224, 224), Image.Resampling.LANCZOS
    )
    prefiltered = high_frequency_prefilter(original)

    results: dict[int, dict[str, Image.Image | float]] = {}
    for size in (32, 16):
        ordinary = original.resize((size, size), Image.Resampling.LANCZOS)
        enhanced = prefiltered.resize((size, size), Image.Resampling.LANCZOS)
        ordinary_bicubic = ordinary.resize((224, 224), Image.Resampling.BICUBIC)
        enhanced_bicubic = enhanced.resize((224, 224), Image.Resampling.BICUBIC)

        ordinary.save(OUTPUT_DIR / f"rgb_{size}x{size}_ordinary.png")
        enhanced.save(OUTPUT_DIR / f"rgb_{size}x{size}_highfreq.png")
        ordinary_bicubic.save(OUTPUT_DIR / f"rgb_{size}x{size}_ordinary_upscaled.png")
        enhanced_bicubic.save(OUTPUT_DIR / f"rgb_{size}x{size}_highfreq_upscaled.png")

        results[size] = {
            "ordinary": ordinary,
            "enhanced": enhanced,
            "ordinary_bicubic": ordinary_bicubic,
            "enhanced_bicubic": enhanced_bicubic,
            "ordinary_psnr_db": psnr(original, ordinary_bicubic),
            "enhanced_psnr_db": psnr(original, enhanced_bicubic),
        }

    figure, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for row, size in enumerate((32, 16)):
        item = results[size]
        add_panel(axes[row, 0], original, "Original RGB\n224 x 224")
        add_panel(
            axes[row, 1],
            item["ordinary"].resize((224, 224), Image.Resampling.NEAREST),
            f"Ordinary {size} x {size}\nnearest-neighbor zoom",
        )
        add_panel(
            axes[row, 2],
            item["enhanced"].resize((224, 224), Image.Resampling.NEAREST),
            f"High-frequency enhanced {size} x {size}\nnearest-neighbor zoom",
        )
        add_panel(
            axes[row, 3],
            item["enhanced_bicubic"],
            f"Same enhanced image\nbicubic zoom to 224",
        )

    figure.suptitle(
        "RGB downsampling: color survives; fine spatial detail does not",
        fontsize=15,
    )
    figure.savefig(OUTPUT_DIR / "rgb_resolution_comparison.png", dpi=170)
    plt.close(figure)

    metadata = {
        "source_shape": [224, 224, 3],
        "method": (
            "RGB unsharp prefilter (Gaussian radius 1.15, residual amount 1.35), "
            "then Lanczos downsampling"
        ),
        "outputs": {
            str(size): {
                "shape": [size, size, 3],
                "values": size * size * 3,
                "ordinary_psnr_db_after_bicubic_upscale": round(
                    float(results[size]["ordinary_psnr_db"]), 3
                ),
                "highfreq_psnr_db_after_bicubic_upscale": round(
                    float(results[size]["enhanced_psnr_db"]), 3
                ),
            }
            for size in (32, 16)
        },
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    info = build_preview()
    print(json.dumps(info, indent=2))
    print(OUTPUT_DIR / "rgb_resolution_comparison.png")
