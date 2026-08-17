"""加载检查点并导出重建或生成样本。"""

import argparse

import torch

from data import build_loader
from model import Generator, build_model
from other import get_device, save_tensor


def evaluate(kind: str, checkpoint: str, output: str, batch_size: int = 8) -> dict:
    """输入：类型、检查点、路径；返回：输入输出张量。"""
    device = get_device()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)

    # 根据类型恢复对应权重
    if kind == "gan":
        generator = Generator().to(device)
        generator.load_state_dict(payload["generator"])
        generator.eval()
        with torch.no_grad():
            sample = generator(torch.randn(batch_size, generator.noise_dim, device=device))
        result = {"sample": sample.cpu()}
    else:
        model = build_model(kind).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        image, label = next(iter(build_loader(batch_size=batch_size, seed=99)))
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            if kind == "cvae":
                recon, _, _ = model(image, label)
            else:
                recon, _, _ = model(image)
        result = {"input": image.cpu(), "reconstruction": recon.cpu(), "label": label.cpu()}

    save_tensor(result, output)
    return result


def main() -> None:
    """输入：命令行参数；返回：评估张量文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("vae", "cvae", "vqvae", "gan"), default="vae")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--output", default="samples.pt")
    args = parser.parse_args()
    result = evaluate(args.kind, args.checkpoint, args.output)
    print({key: tuple(value.shape) for key, value in result.items()})


if __name__ == "__main__":
    main()
