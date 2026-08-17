"""输入: NeRF模型或高斯集合。
返回: 光线颜色或二维高斯图。"""

import torch
from torch import Tensor

from data import make_samples
from model import GaussianCloud, TinyNeRF


def render_rays(model: TinyNeRF, origins: Tensor, directions: Tensor) -> Tensor:
    """输入: 模型、光线原点和方向。
    返回: [B,3] 体渲染颜色。"""
    positions, deltas = make_samples(origins, directions)
    colors, density = model(positions)
    alpha = 1 - (-density.squeeze(-1) * deltas).exp()
    transmittance = torch.cumprod(1 - alpha + 1e-6, dim=1)
    transmittance = torch.cat((torch.ones_like(alpha[:, :1]), transmittance[:, :-1]), dim=1)
    # 依照体渲染权重混色
    return (transmittance * alpha).unsqueeze(-1).mul(colors).sum(dim=1)


def splat_points(cloud: GaussianCloud, side: int = 16) -> Tensor:
    """输入: 三维高斯集合与边长。
    返回: [3,H,W] 投影颜色图。"""
    means, colors, scales = cloud()
    grid = torch.linspace(-1, 1, side, device=means.device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((xx, yy), dim=-1)
    distance = (points[None] - means[:, None, None, :2]).square().sum(dim=-1)
    weights = (-distance / (2 * scales[:, None, None, 0].square())).exp()
    # 将投影高斯累加成图
    image = torch.einsum("nhw,nc->chw", weights, colors)
    return image / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
