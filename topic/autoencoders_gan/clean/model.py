"""VAE、CVAE、VQ-VAE 与 GAN 的最小模型。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 16


def _image_encoder(out_channels: int) -> nn.Sequential:
    """输入：输出通道数；返回：图像编码器。"""
    return nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, out_channels, 4, stride=2, padding=1),
        nn.SiLU(),
    )


def _image_decoder(in_channels: int) -> nn.Sequential:
    """输入：输入通道数；返回：图像解码器。"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, 32, 4, stride=2, padding=1),
        nn.SiLU(),
        nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        nn.Sigmoid(),
    )


class VAE(nn.Module):
    """使用高斯潜变量重建玩具图像。"""

    def __init__(self, latent_dim: int = 16):
        """输入：潜变量维度；返回：VAE 实例。"""
        super().__init__()
        self.encoder = _image_encoder(64)
        self.to_stats = nn.Linear(64 * 4 * 4, latent_dim * 2)
        self.from_latent = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = _image_decoder(64)

    def encode(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """输入：[B,1,16,16]图像；返回：均值、对数方差。"""
        feature = self.encoder(image).flatten(1)
        mean, logvar = self.to_stats(feature).chunk(2, dim=1)
        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """输入：均值、对数方差；返回：高斯采样。"""
        noise = torch.randn_like(mean)
        return mean + noise * (0.5 * logvar).exp()

    def decode(self, latent: Tensor) -> Tensor:
        """输入：[B,D]潜变量；返回：重建图像。"""
        feature = self.from_latent(latent).view(-1, 64, 4, 4)
        return self.decoder(feature)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入：[B,1,16,16]图像；返回：重建、均值、方差。"""
        mean, logvar = self.encode(image)
        return self.decode(self.reparameterize(mean, logvar)), mean, logvar


class CVAE(nn.Module):
    """将类别嵌入接入编码器和解码器。"""

    def __init__(self, latent_dim: int = 16, classes: int = 4, condition_dim: int = 8):
        """输入：潜维、类别数、条件维；返回：CVAE 实例。"""
        super().__init__()
        self.encoder = _image_encoder(64)
        self.condition = nn.Embedding(classes, condition_dim)
        self.to_stats = nn.Linear(64 * 4 * 4 + condition_dim, latent_dim * 2)
        self.from_latent = nn.Linear(latent_dim + condition_dim, 64 * 4 * 4)
        self.decoder = _image_decoder(64)

    def encode(self, image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        """输入：图像、类别；返回：均值、对数方差。"""
        feature = self.encoder(image).flatten(1)
        feature = torch.cat((feature, self.condition(label)), dim=1)
        mean, logvar = self.to_stats(feature).chunk(2, dim=1)
        return mean, logvar

    def decode(self, latent: Tensor, label: Tensor) -> Tensor:
        """输入：潜变量、类别；返回：条件重建图像。"""
        feature = torch.cat((latent, self.condition(label)), dim=1)
        return self.decoder(self.from_latent(feature).view(-1, 64, 4, 4))

    def forward(self, image: Tensor, label: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入：图像、类别；返回：重建、均值、方差。"""
        mean, logvar = self.encode(image, label)
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return self.decode(latent, label), mean, logvar


class VectorQuantizer(nn.Module):
    """用直通估计量离散化潜特征。"""

    def __init__(self, codes: int = 32, dim: int = 16, beta: float = 0.25):
        """输入：码本数、维度、权重；返回：量化器。"""
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codes, dim) * 0.1)
        self.beta = beta

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入：[B,D,H,W]特征；返回：量化特征、损失、索引。"""
        batch, channels, height, width = latent.shape
        flat = latent.permute(0, 2, 3, 1).reshape(-1, channels)
        distance = flat.square().sum(1, keepdim=True)
        distance = distance - 2 * flat @ self.codebook.t() + self.codebook.square().sum(1)
        indices = distance.argmin(dim=1)
        quantized = F.embedding(indices, self.codebook).view(batch, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # 分别更新编码器和码本
        codebook_loss = F.mse_loss(quantized, latent.detach())
        commit_loss = F.mse_loss(latent, quantized.detach())
        loss = codebook_loss + self.beta * commit_loss
        straight = latent + (quantized - latent).detach()
        return straight, loss, indices.view(batch, height, width)


class VQVAE(nn.Module):
    """使用离散潜码重建玩具图像。"""

    def __init__(self, latent_dim: int = 16, codes: int = 32):
        """输入：潜维、码本数；返回：VQ-VAE 实例。"""
        super().__init__()
        self.encoder = _image_encoder(latent_dim)
        self.quantizer = VectorQuantizer(codes=codes, dim=latent_dim)
        self.decoder = _image_decoder(latent_dim)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入：[B,1,16,16]图像；返回：重建、量化损失、索引。"""
        quantized, loss, indices = self.quantizer(self.encoder(image))
        return self.decoder(quantized), loss, indices


class Generator(nn.Module):
    """将高斯噪声映射为图像。"""

    def __init__(self, noise_dim: int = 32):
        """输入：噪声维度；返回：生成器。"""
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, IMAGE_SIZE * IMAGE_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, noise: Tensor) -> Tensor:
        """输入：[B,D]高斯噪声；返回：[B,1,16,16]图像。"""
        return self.net(noise).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


class Discriminator(nn.Module):
    """为图像输出未归一化真假分数。"""

    def __init__(self):
        """输入：无；返回：判别器。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """输入：[B,1,16,16]图像；返回：[B,1]分数。"""
        return self.net(image)


def build_model(kind: str) -> nn.Module:
    """输入：模型名称；返回：对应生成模型。"""
    models = {"vae": VAE, "cvae": CVAE, "vqvae": VQVAE}
    if kind not in models:
        raise ValueError("kind 必须是 vae、cvae 或 vqvae")
    return models[kind]()
