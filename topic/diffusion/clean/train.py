"""训练 DDPM、潜空间扩散、DiT 或流匹配。"""

import argparse
from itertools import cycle

import torch
from torch.nn import functional as F

from data import build_loader
from model import TinyDiT, TinyUNet, TinyVAE
from other import get_device, linear_betas, make_flow_pair, q_sample, save_tensor, set_seed


def vae_loss(recon: torch.Tensor, image: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """输入：重建、图像、统计量；返回：VAE 标量损失。"""
    recon_loss = F.mse_loss(recon, image)
    kl_loss = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    return recon_loss + 0.02 * kl_loss


def train_vae(model: TinyVAE, loader, steps: int, device: torch.device) -> list[float]:
    """输入：VAE、加载器、步数；返回：训练损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    model.train()
    for _, (image, _) in zip(range(steps), cycle(loader)):
        image = image.to(device)
        recon, mean, logvar = model(image)
        loss = vae_loss(recon, image, mean, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def train_pixel_generator(
    model: torch.nn.Module, loader, betas: torch.Tensor, steps: int, device: torch.device, flow: bool = False
) -> list[float]:
    """输入：预测器、加载器、日程；返回：训练损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    model.train()

    # 按选择的目标回归噪声或速度
    for step, (image, label) in zip(range(steps), cycle(loader)):
        image, label = image.to(device), label.to(device)
        if flow:
            time = torch.rand(image.size(0), device=device)
            noisy, target = make_flow_pair(image, time)
        else:
            index = torch.randint(len(betas), (image.size(0),), device=device)
            target = torch.randn_like(image)
            noisy = q_sample(image, index, betas, target)
            time = index.float() / max(len(betas) - 1, 1)
        loss = F.mse_loss(model(noisy, time, label), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step={step + 1:03d} loss={history[-1]:.4f}")
    return history


def train_latent_diffusion(loader, betas: torch.Tensor, steps: int, device: torch.device) -> tuple[TinyVAE, TinyUNet, list[float]]:
    """输入：加载器、日程、步数；返回：VAE、潜空间 U-Net、损失。"""
    vae = TinyVAE().to(device)
    warmup = max(steps // 4, 1)

    # 先拟合小型图像码本
    history = train_vae(vae, loader, warmup, device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    model = TinyUNet(in_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for step, (image, label) in zip(range(steps), cycle(loader)):
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            latent, _ = vae.encode(image)
        index = torch.randint(len(betas), (image.size(0),), device=device)
        target = torch.randn_like(latent)
        noisy = q_sample(latent, index, betas, target)
        time = index.float() / max(len(betas) - 1, 1)
        loss = F.mse_loss(model(noisy, time, label), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step={step + 1:03d} loss={history[-1]:.4f}")
    return vae, model, history


def main() -> None:
    """输入：命令行参数；返回：生成模型检查点。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vae", "ddpm", "ldm", "dit", "flow"), default="ddpm")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoint.pt")
    args = parser.parse_args()

    # 统一初始化训练状态
    set_seed(args.seed)
    device = get_device()
    loader = build_loader(batch_size=args.batch_size, seed=args.seed)
    betas = linear_betas(args.timesteps, device)

    if args.mode == "vae":
        vae = TinyVAE().to(device)
        history = train_vae(vae, loader, args.steps, device)
        payload = {"mode": "vae", "vae": vae.state_dict(), "history": history}
    elif args.mode == "ldm":
        vae, model, history = train_latent_diffusion(loader, betas, args.steps, device)
        payload = {"mode": "ldm", "vae": vae.state_dict(), "model": model.state_dict(), "history": history}
    else:
        model = TinyUNet().to(device) if args.mode == "ddpm" else TinyDiT().to(device)
        history = train_pixel_generator(model, loader, betas, args.steps, device, flow=args.mode == "flow")
        payload = {"mode": args.mode, "model": model.state_dict(), "history": history}

    payload["betas"] = betas.cpu()
    print(f"saved={save_tensor(payload, args.output)}")


if __name__ == "__main__":
    main()
