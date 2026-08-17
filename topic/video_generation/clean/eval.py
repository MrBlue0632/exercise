"""采样条件短视频并保存张量。"""

import argparse

import torch

from model import VideoFlowTransformer
from other import euler_sample, get_device, save_tensor


def evaluate(checkpoint: str, output: str, batch_size: int = 4, steps: int = 16) -> torch.Tensor:
    """输入：检查点、路径、采样参数；返回：生成视频。"""
    device = get_device()
    model = VideoFlowTransformer().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()

    # 为四个运动方向生成样本
    direction = torch.arange(batch_size, device=device) % 4
    video = euler_sample(model, direction, steps=steps)
    save_tensor({"video": video.cpu(), "direction": direction.cpu()}, output)
    return video


def main() -> None:
    """输入：命令行参数；返回：视频张量文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--output", default="samples.pt")
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    video = evaluate(args.checkpoint, args.output, steps=args.steps)
    print(tuple(video.shape))


if __name__ == "__main__":
    main()
