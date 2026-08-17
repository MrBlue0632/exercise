"""DDPM、潜空间扩散与 DiT 的最小骨干。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from other import timestep_embedding


class TimestepEmbedder(nn.Module):
    """将连续时间映射为条件向量。"""

    def __init__(self, dim: int):
        """输入：嵌入维度；返回：时间嵌入器。"""
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: Tensor) -> Tensor:
        """输入：[B]连续时间；返回：[B,D]条件向量。"""
        return self.mlp(timestep_embedding(time, self.dim))


class LabelConditioner(nn.Module):
    """把类别展开为多个上下文词元。"""

    def __init__(self, classes: int, dim: int, tokens: int = 4):
        """输入：类别数、维度、词元数；返回：条件器。"""
        super().__init__()
        self.dim = dim
        self.tokens = tokens
        self.embedding = nn.Embedding(classes, dim * tokens)

    def forward(self, label: Tensor) -> Tensor:
        """输入：[B]类别；返回：[B,N,D]条件词元。"""
        return self.embedding(label).view(label.size(0), self.tokens, self.dim)


class ResBlock(nn.Module):
    """将时间条件加到卷积残差块。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        """输入：输入通道、输出通道、时间维；返回：残差块。"""
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, feature: Tensor, time: Tensor) -> Tensor:
        """输入：特征图、时间向量；返回：更新特征图。"""
        hidden = self.conv1(F.silu(self.norm1(feature)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(feature)


class CrossAttention2d(nn.Module):
    """让空间查询读取条件词元。"""

    def __init__(self, channels: int, context_dim: int):
        """输入：通道数、条件维度；返回：交叉注意力。"""
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.to_query = nn.Linear(channels, channels)
        self.to_key = nn.Linear(context_dim, channels)
        self.to_value = nn.Linear(context_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, feature: Tensor, context: Tensor) -> Tensor:
        """输入：特征图、条件词元；返回：条件特征图。"""
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        query = self.to_query(self.norm(tokens))
        key = self.to_key(context)
        value = self.to_value(context)

        # 计算空间词元到条件词元的权重
        weight = (query @ key.transpose(-2, -1)) * (channels ** -0.5)
        update = torch.softmax(weight, dim=-1) @ value
        update = self.proj(update).transpose(1, 2).view(batch, channels, height, width)
        return feature + update


class TinyUNet(nn.Module):
    """支持像素或潜空间噪声预测。"""

    def __init__(self, in_channels: int = 1, width: int = 32, time_dim: int = 128):
        """输入：通道、宽度、时间维；返回：条件 U-Net。"""
        super().__init__()
        self.time = TimestepEmbedder(time_dim)
        self.condition = LabelConditioner(4, time_dim)
        self.input = nn.Conv2d(in_channels, width, 3, padding=1)
        self.down_block = ResBlock(width, width, time_dim)
        self.downsample = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.middle = ResBlock(width * 2, width * 2, time_dim)
        self.cross = CrossAttention2d(width * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.up_block = ResBlock(width * 2, width, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, in_channels, 3, padding=1))

    def forward(self, noisy: Tensor, time: Tensor, label: Tensor) -> Tensor:
        """输入：加噪样本、时间、类别；返回：噪声或速度。"""
        time_embed = self.time(time)
        context = self.condition(label)

        # 下采样、条件混合、上采样
        skip = self.down_block(self.input(noisy), time_embed)
        hidden = self.middle(self.downsample(skip), time_embed)
        hidden = self.cross(hidden, context)
        hidden = self.upsample(hidden)
        hidden = self.up_block(torch.cat((hidden, skip), dim=1), time_embed)
        return self.output(hidden)


class TinyVAE(nn.Module):
    """为潜空间扩散提供四通道码。"""

    def __init__(self, latent_channels: int = 4):
        """输入：潜变量通道；返回：小型 VAE。"""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, latent_channels * 2, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """输入：[B,1,16,16]图像；返回：均值、对数方差。"""
        mean, logvar = self.encoder(image).chunk(2, dim=1)
        return mean, logvar

    def decode(self, latent: Tensor) -> Tensor:
        """输入：[B,4,4,4]潜变量；返回：重建图像。"""
        return self.decoder(latent)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入：图像；返回：重建、均值、对数方差。"""
        mean, logvar = self.encode(image)
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return self.decode(latent), mean, logvar


def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """输入：词元、偏移、缩放；返回：调制词元。"""
    return tokens * (1.0 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """使用 adaLN 调制的 Transformer 块。"""

    def __init__(self, dim: int, heads: int):
        """输入：维度、头数；返回：DiT 块。"""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        """输入：[B,N,D]词元、条件；返回：更新词元。"""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=1)
        hidden = _modulate(self.norm1(tokens), shift_a, scale_a)
        tokens = tokens + gate_a[:, None] * self.attn(hidden, hidden, hidden, need_weights=False)[0]
        hidden = _modulate(self.norm2(tokens), shift_m, scale_m)
        return tokens + gate_m[:, None] * self.mlp(hidden)


class TinyDiT(nn.Module):
    """在图像补丁上预测噪声或速度。"""

    def __init__(self, image_size: int = 16, patch: int = 4, dim: int = 64, depth: int = 2, heads: int = 4):
        """输入：尺寸、补丁、维度、层数；返回：小型 DiT。"""
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch
        self.grid = image_size // patch
        self.patch_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.position = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        self.time = TimestepEmbedder(dim)
        self.label = nn.Embedding(4, dim)
        self.blocks = nn.ModuleList(DiTBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, patch * patch)
        nn.init.normal_(self.position, std=0.02)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        """输入：[B,N,P²]补丁；返回：[B,1,H,W]图像。"""
        batch = patches.size(0)
        patch = self.patch_size
        grid = self.grid
        image = patches.view(batch, grid, grid, patch, patch, 1)
        return image.permute(0, 5, 1, 3, 2, 4).reshape(batch, 1, self.image_size, self.image_size)

    def forward(self, noisy: Tensor, time: Tensor, label: Tensor) -> Tensor:
        """输入：加噪图像、时间、类别；返回：噪声或速度。"""
        if noisy.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("TinyDiT 仅支持设定图像尺寸")
        tokens = self.patch_embed(noisy).flatten(2).transpose(1, 2)
        condition = self.time(time) + self.label(label)

        # 叠加位置后执行 adaLN 块
        tokens = tokens + self.position
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self._unpatchify(self.out(self.norm(tokens)))
