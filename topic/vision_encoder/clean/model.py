"""输入: 图像和词元张量。
返回: 归一化视觉或文本特征。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyVisionEncoder(nn.Module):
    """输入: [B,C,H,W] 图像。
    返回: [B,D] 视觉特征。"""

    def __init__(self, image_size: int = 16, width: int = 32) -> None:
        """输入：图像尺寸与宽度；返回：初始化编码器。"""
        super().__init__()
        self.patch = nn.Conv2d(3, width, kernel_size=4, stride=4)
        patch_count = (image_size // 4) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(width)

    def forward(self, images: Tensor) -> Tensor:
        """输入: [B,3,16,16] 图像。
        返回: [B,D] 归一化特征。"""
        tokens = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(images.size(0), -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.pos
        # 聚合补丁间上下文
        return self.norm(self.blocks(tokens)[:, 0])


class TinyClip(nn.Module):
    """输入: 图像和词元序列。
    返回: 两路归一化嵌入。"""

    def __init__(self, vocab_size: int = 32, width: int = 32) -> None:
        """输入：词表与宽度；返回：初始化图文模型。"""
        super().__init__()
        self.image = TinyVisionEncoder(width=width)
        self.text = nn.Embedding(vocab_size, width)
        self.text_norm = nn.LayerNorm(width)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images: Tensor) -> Tensor:
        """输入: 图像批次张量。
        返回: 单位长度图像特征。"""
        return F.normalize(self.image(images), dim=-1)

    def encode_text(self, tokens: Tensor) -> Tensor:
        """输入: [B,T] 整数词元。
        返回: 单位长度文本特征。"""
        features = self.text_norm(self.text(tokens).mean(dim=1))
        return F.normalize(features, dim=-1)

    def forward(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """输入: 成对图像和词元。
        返回: 图像与文本特征。"""
        return self.encode_image(images), self.encode_text(tokens)
