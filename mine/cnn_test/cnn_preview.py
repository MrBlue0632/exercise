"""Generate an intuitive preview of fixed image kernels and a real CNN layer."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "outputs"
ASTRONAUT_PATH = ASSET_DIR / "astronaut.png"
ASTRONAUT_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.25.2/skimage/data/astronaut.png"
)


def ensure_input_image() -> Path:
    """Download the agreed scikit-image astronaut sample once."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if not ASTRONAUT_PATH.exists():
        print(f"Downloading sample image to {ASTRONAUT_PATH}")
        urllib.request.urlretrieve(ASTRONAUT_URL, ASTRONAUT_PATH)
    return ASTRONAUT_PATH


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )


def tensor_to_gray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().squeeze().cpu().numpy()


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def convolve_gray(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel.to(dtype=image.dtype).view(1, 1, *kernel.shape)
    return F.conv2d(image, kernel, padding="same")


def convolve_rgb(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel.to(dtype=image.dtype).view(1, 1, *kernel.shape)
    kernel = kernel.repeat(3, 1, 1, 1)
    return F.conv2d(image, kernel, padding="same", groups=3)


def robust_unit(array: np.ndarray) -> np.ndarray:
    ceiling = float(np.percentile(np.abs(array), 99.0))
    if ceiling < 1e-8:
        return np.zeros_like(array)
    return np.clip(np.abs(array) / ceiling, 0.0, 1.0)


def signed_preview(array: np.ndarray) -> np.ndarray:
    ceiling = float(np.percentile(np.abs(array), 99.0))
    if ceiling < 1e-8:
        return np.full_like(array, 0.5)
    return np.clip(0.5 + 0.5 * array / ceiling, 0.0, 1.0)


def feature_montage(maps: np.ndarray, columns: int = 4) -> np.ndarray:
    count, height, width = maps.shape
    rows = int(np.ceil(count / columns))
    gap = 3
    canvas = np.zeros(
        (rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap),
        dtype=np.float32,
    )
    for index, feature in enumerate(maps):
        row, column = divmod(index, columns)
        feature = feature - feature.min()
        maximum = float(feature.max())
        if maximum > 1e-8:
            feature = feature / maximum
        y = row * (height + gap)
        x = column * (width + gap)
        canvas[y : y + height, x : x + width] = feature
    return canvas


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0)).save(path)


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L").save(path)


def add_panel(axis, image, title: str, *, cmap=None) -> None:
    axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=11)
    axis.axis("off")


def build_preview(image_size: int = 224) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Image.open(ensure_input_image()).convert("RGB")
    source = source.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image = pil_to_tensor(source)
    gray = rgb_to_gray(image)

    gaussian = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=torch.float32,
    ) / 256.0
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    )
    sobel_y = sobel_x.T.contiguous()
    laplacian = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
    )
    sharpen_kernel = torch.tensor(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32
    )

    blurred = convolve_rgb(image, gaussian).clamp(0.0, 1.0)
    residual = image - blurred
    unsharp = (image + 1.5 * residual).clamp(0.0, 1.0)
    sharpened = convolve_rgb(image, sharpen_kernel).clamp(0.0, 1.0)
    sx = convolve_gray(gray, sobel_x)
    sy = convolve_gray(gray, sobel_y)
    sobel = torch.sqrt(sx.square() + sy.square() + 1e-12)
    lap = convolve_gray(gray, laplacian)

    # A pretrained first CNN stage: Conv2d -> BatchNorm -> ReLU.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()
    mean = image.new_tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = image.new_tensor(weights.transforms().std).view(1, 3, 1, 1)
    normalized = (image - mean) / std
    with torch.no_grad():
        cnn_features = model.relu(model.bn1(model.conv1(normalized)))

    # Select the 16 channels with the largest spatial variance for a useful view.
    scores = cnn_features.var(dim=(-2, -1)).squeeze(0)
    selected = scores.topk(16).indices.sort().values
    selected_maps = cnn_features[0, selected].cpu().numpy()
    montage = feature_montage(selected_maps)

    rgb = tensor_to_rgb(image)
    blurred_rgb = tensor_to_rgb(blurred)
    unsharp_rgb = tensor_to_rgb(unsharp)
    sharpened_rgb = tensor_to_rgb(sharpened)
    sobel_view = robust_unit(tensor_to_gray(sobel))
    lap_view = robust_unit(tensor_to_gray(lap))
    residual_view = signed_preview(tensor_to_gray(rgb_to_gray(residual)))

    save_rgb(OUTPUT_DIR / "01_original_224.png", rgb)
    save_rgb(OUTPUT_DIR / "02_gaussian_low_pass.png", blurred_rgb)
    save_gray(OUTPUT_DIR / "03_sobel_edges.png", sobel_view)
    save_gray(OUTPUT_DIR / "04_laplacian_high_pass.png", lap_view)
    save_gray(OUTPUT_DIR / "05_high_frequency_residual.png", residual_view)
    save_rgb(OUTPUT_DIR / "06_unsharp_mask.png", unsharp_rgb)
    save_rgb(OUTPUT_DIR / "07_sharpen_kernel.png", sharpened_rgb)
    save_gray(OUTPUT_DIR / "08_resnet18_feature_montage.png", montage)

    figure, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    add_panel(axes[0, 0], rgb, "Original (224 x 224)")
    add_panel(axes[0, 1], blurred_rgb, "Gaussian 5x5: low-pass")
    add_panel(axes[0, 2], sobel_view, "Sobel: edge magnitude", cmap="gray")
    add_panel(axes[0, 3], lap_view, "Laplacian: high-pass", cmap="gray")
    add_panel(
        axes[1, 0], residual_view, "High-frequency residual (gray=0)", cmap="gray"
    )
    add_panel(axes[1, 1], unsharp_rgb, "Original + 1.5 x residual")
    add_panel(axes[1, 2], sharpened_rgb, "Sharpen 3x3")
    add_panel(axes[1, 3], montage, "Pretrained ResNet-18 features", cmap="magma")
    figure.suptitle("What convolution keeps, removes, and learns", fontsize=16)
    figure.savefig(OUTPUT_DIR / "cnn_preview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    for axis, feature, channel in zip(axes.flat, selected_maps, selected.tolist()):
        add_panel(axis, robust_unit(feature), f"channel {channel}", cmap="magma")
    figure.suptitle("ResNet-18 first-stage feature maps (top variance channels)")
    figure.savefig(OUTPUT_DIR / "resnet18_feature_maps.png", dpi=160)
    plt.close(figure)

    metadata = {
        "input": "scikit-image astronaut",
        "input_size": [image_size, image_size, 3],
        "fixed_kernels": {
            "gaussian_5x5_sum": float(gaussian.sum()),
            "sobel_x_sum": float(sobel_x.sum()),
            "sobel_y_sum": float(sobel_y.sum()),
            "laplacian_sum": float(laplacian.sum()),
            "sharpen_sum": float(sharpen_kernel.sum()),
        },
        "cnn": {
            "model": "torchvision ResNet-18 DEFAULT pretrained weights",
            "stage": "conv1 -> bn1 -> relu",
            "feature_shape": list(cnn_features.shape),
            "selected_channels": selected.tolist(),
        },
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=224, help="square input size")
    args = parser.parse_args()
    metadata = build_preview(args.size)
    print(json.dumps(metadata, indent=2))
    print(f"Preview written to {OUTPUT_DIR / 'cnn_preview.png'}")


if __name__ == "__main__":
    main()
