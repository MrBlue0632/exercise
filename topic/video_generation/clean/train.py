"""训练最小视频流匹配 Transformer。"""

import argparse
from itertools import cycle

import torch
from torch.nn import functional as F

from data import build_loader
from model import VideoFlowTransformer
from other import get_device, make_flow_pair, save_tensor, set_seed


def flow_loss(model: VideoFlowTransformer, video: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """输入：模型、视频、方向；返回：速度回归损失。"""
    time = torch.rand(video.size(0), device=video.device)
    noisy_video, target = make_flow_pair(video, time)
    prediction = model(noisy_video, time, direction)
    return F.mse_loss(prediction, target)


def train(model: VideoFlowTransformer, loader, steps: int, device: torch.device) -> list[float]:
    """输入：模型、加载器、步数；返回：每步损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    history: list[float] = []
    model.train()

    # 在噪声到视频路径上拟合速度
    for step, (video, direction) in zip(range(steps), cycle(loader)):
        video, direction = video.to(device), direction.to(device)
        loss = flow_loss(model, video, direction)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step={step + 1:03d} loss={history[-1]:.4f}")
    return history


def main() -> None:
    """输入：命令行参数；返回：视频模型检查点。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoint.pt")
    args = parser.parse_args()

    # 构建玩具数据和模型
    set_seed(args.seed)
    device = get_device()
    model = VideoFlowTransformer().to(device)
    history = train(model, build_loader(batch_size=args.batch_size, seed=args.seed), args.steps, device)
    print(f"saved={save_tensor({'model': model.state_dict(), 'history': history}, args.output)}")


if __name__ == "__main__":
    main()
