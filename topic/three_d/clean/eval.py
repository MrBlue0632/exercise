"""输入: NeRF模型和合成射线。
返回: 渲染误差指标。"""

import torch
from torch import Tensor

from model import TinyNeRF
from other import render_rays


@torch.no_grad()
def evaluate(model: TinyNeRF, origins: Tensor, directions: Tensor, targets: Tensor) -> dict[str, float]:
    """输入: 模型、射线和目标色。
    返回: MSE与PSNR指标。"""
    prediction = render_rays(model, origins, directions)
    mse = (prediction - targets).square().mean().clamp_min(1e-8)
    # 使用标准图像重建指标
    return {"mse": float(mse), "psnr": float(-10 * mse.log10())}
