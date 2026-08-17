"""训练 VAE、CVAE、VQ-VAE 或 GAN。"""

import argparse
from itertools import cycle

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import build_loader
from model import Discriminator, Generator, build_model
from other import get_device, save_tensor, set_seed


def vae_loss(recon: Tensor, image: Tensor, mean: Tensor, logvar: Tensor) -> Tensor:
    """输入：重建、图像、统计量；返回：VAE 标量损失。"""
    recon_loss = F.mse_loss(recon, image)
    kl_loss = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    return recon_loss + 0.05 * kl_loss


def train_autoencoder(
    kind: str, model: nn.Module, loader, steps: int, device: torch.device
) -> list[float]:
    """输入：类型、模型、加载器；返回：每步损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    model.train()

    # 按批次执行重建学习
    for step, (image, label) in zip(range(steps), cycle(loader)):
        image, label = image.to(device), label.to(device)
        if kind == "cvae":
            recon, mean, logvar = model(image, label)
            loss = vae_loss(recon, image, mean, logvar)
        elif kind == "vqvae":
            recon, quant_loss, _ = model(image)
            loss = F.mse_loss(recon, image) + quant_loss
        else:
            recon, mean, logvar = model(image)
            loss = vae_loss(recon, image, mean, logvar)

        # 反向更新全部模块
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step={step + 1:03d} loss={history[-1]:.4f}")
    return history


def train_gan(
    generator: Generator, discriminator: Discriminator, loader, steps: int, device: torch.device
) -> list[float]:
    """输入：GAN 模块、加载器；返回：生成器损失。"""
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.9))
    history: list[float] = []
    generator.train()
    discriminator.train()

    # 先更新判别器再更新生成器
    for step, (real, _) in zip(range(steps), cycle(loader)):
        real = real.to(device)
        noise = torch.randn(real.size(0), generator.noise_dim, device=device)
        fake = generator(noise)
        real_score = discriminator(real)
        fake_score = discriminator(fake.detach())
        d_loss = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
        d_loss = d_loss + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        noise = torch.randn(real.size(0), generator.noise_dim, device=device)
        fake = generator(noise)
        g_loss = F.binary_cross_entropy_with_logits(discriminator(fake), torch.ones_like(discriminator(fake)))
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        history.append(float(g_loss.detach()))
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step={step + 1:03d} g_loss={history[-1]:.4f}")
    return history


def main() -> None:
    """输入：命令行参数；返回：检查点文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("vae", "cvae", "vqvae", "gan"), default="vae")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoint.pt")
    args = parser.parse_args()

    # 初始化数据和训练设备
    set_seed(args.seed)
    device = get_device()
    loader = build_loader(batch_size=args.batch_size, seed=args.seed)

    if args.kind == "gan":
        generator, discriminator = Generator().to(device), Discriminator().to(device)
        history = train_gan(generator, discriminator, loader, args.steps, device)
        payload = {"kind": "gan", "generator": generator.state_dict(), "discriminator": discriminator.state_dict()}
    else:
        model = build_model(args.kind).to(device)
        history = train_autoencoder(args.kind, model, loader, args.steps, device)
        payload = {"kind": args.kind, "model": model.state_dict()}

    payload["history"] = history
    print(f"saved={save_tensor(payload, args.output)}")


if __name__ == "__main__":
    main()
