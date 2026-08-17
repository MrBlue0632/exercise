"""基于上下文的下一个字符预测模型。"""

from __future__ import annotations

import torch
from torch import nn


class TokenPredictor(nn.Module):
    """将上下文 token 平均后预测下一个 token。"""

    def __init__(self, vocab_size: int, embedding_dim: int = 32) -> None:
        """输入词表大小和嵌入维度，返回初始化模型。"""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入 [batch, context] token，返回 [batch, vocab] logits。"""
        context = self.embedding(token_ids).mean(dim=1)
        return self.head(context)


if __name__ == "__main__":
    print(TokenPredictor(8)(torch.zeros(2, 3, dtype=torch.long)).shape)
