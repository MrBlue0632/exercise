"""Top-k 稀疏专家语言模型。"""

import torch
from torch import Tensor, nn


class Expert(nn.Module):
    """一个小型前馈专家。"""

    def __init__(self, width: int) -> None:
        """输入：隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[N,D]。
        返回: output[N,D]。
        """
        return self.net(hidden)


class SparseMoE(nn.Module):
    """按 Top-k 路由激活专家。"""

    def __init__(self, width: int = 48, experts: int = 4, top_k: int = 2) -> None:
        """输入：隐藏维度、专家数和 top-k；返回：初始化模块。"""
        super().__init__()
        self.expert_count, self.top_k = experts, top_k
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList(Expert(width) for _ in range(experts))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D], probs[B,T,E], ids[B,T,K]。
        """
        batch, steps, width = hidden.shape
        flat = hidden.reshape(-1, width)
        probs = torch.softmax(self.router(flat), dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        output = torch.zeros_like(flat)

        # 仅计算被选中的专家
        for expert_id, expert in enumerate(self.experts):
            selected = (indices == expert_id).any(dim=-1)
            if selected.any():
                gate = (weights * (indices == expert_id)).sum(dim=-1, keepdim=True)
                output[selected] = output[selected] + expert(flat[selected]) * gate[selected]
        return output.view(batch, steps, width), probs.view(batch, steps, -1), indices.view(batch, steps, -1)


class MoELM(nn.Module):
    """在词元层上使用稀疏专家。"""

    def __init__(self, vocab_size: int = 32, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(32, width)
        self.norm = nn.LayerNorm(width)
        self.moe = SparseMoE(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输入: ids[B,T]。
        返回: logits[B,T,V], probs[B,T,E], ids[B,T,K]。
        """
        position = torch.arange(ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(position)[None]
        update, probs, expert_ids = self.moe(self.norm(hidden))
        return self.head(hidden + update), probs, expert_ids


def build_model() -> MoELM:
    """输入: 无。
    返回: MoELM。
    """
    return MoELM()
