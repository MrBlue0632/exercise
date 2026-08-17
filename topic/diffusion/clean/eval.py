"""从不同生成目标导出玩具样本。"""

import argparse

import torch

from data import build_loader
from model import TinyDiT, TinyUNet, TinyVAE
from other import get_device, sample_ddpm, sample_flow, save_tensor


def evaluate(checkpoint: str, output: str, batch_size: int = 4, flow_steps: int = 16) -> dict:
    """输入：检查点、路径、采样参数；返回：样本字典。"""
    device = get_device()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    mode = payload["mode"]
    betas = payload["betas"].to(device)
    label = torch.arange(batch_size, device=device) % 4

    # 按检查点类型恢复采样链
    if mode == "vae":
        vae = TinyVAE().to(device)
        vae.load_state_dict(payload["vae"])
        image, _ = next(iter(build_loader(batch_size=batch_size, seed=99)))
        with torch.no_grad():
            recon, _, _ = vae(image.to(device))
        result = {"input": image, "sample": recon.cpu(), "label": label.cpu()}
    elif mode == "ldm":
        vae, model = TinyVAE().to(device), TinyUNet(in_channels=4).to(device)
        vae.load_state_dict(payload["vae"])
        model.load_state_dict(payload["model"])
        model.eval()
        latent = sample_ddpm(model, (batch_size, 4, 4, 4), label, betas)
        result = {"sample": vae.decode(latent).cpu(), "label": label.cpu()}
    else:
        model = TinyUNet().to(device) if mode == "ddpm" else TinyDiT().to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        if mode == "flow":
            sample = sample_flow(model, (batch_size, 1, 16, 16), label, flow_steps)
        else:
            sample = sample_ddpm(model, (batch_size, 1, 16, 16), label, betas)
        result = {"sample": sample.cpu(), "label": label.cpu()}

    save_tensor(result, output)
    return result


def main() -> None:
    """输入：命令行参数；返回：样本张量文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--output", default="samples.pt")
    parser.add_argument("--flow-steps", type=int, default=16)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.output, flow_steps=args.flow_steps)
    print({key: tuple(value.shape) for key, value in result.items()})


if __name__ == "__main__":
    main()
