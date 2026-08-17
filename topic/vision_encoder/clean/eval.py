"""输入: 图文模型和配对样本。
返回: 检索准确率字典。"""

import torch
from torch import Tensor

from model import TinyClip


@torch.no_grad()
def retrieval_accuracy(model: TinyClip, images: Tensor, tokens: Tensor) -> dict[str, float]:
    """输入: 模型与配对数据。
    返回: 双向检索准确率。"""
    image_features, text_features = model(images, tokens)
    scores = image_features @ text_features.T
    labels = torch.arange(scores.size(0), device=scores.device)
    # 统计成对样本命中率
    image_to_text = (scores.argmax(dim=1) == labels).float().mean()
    text_to_image = (scores.argmax(dim=0) == labels).float().mean()
    return {"image_to_text": float(image_to_text), "text_to_image": float(text_to_image)}
