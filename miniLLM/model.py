import math
import struct
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 4096  # Token 向量及模型主干的维度
    n_layers: int = 32  # Transformer Block 的数量
    n_heads: int = 32  # Query 的注意力头数量
    n_kv_heads: Optional[int] = None  # Key、Value 的注意力头数量；None 表示与 n_heads 相同
    vocab_size: int = 32000  # Tokenizer 词表大小
    hidden_dim: Optional[int] = None  # SwiGLU 前馈网络的中间层维度；None 表示后续根据 dim 自动计算
    multiple_of: int = 256  # 将 hidden_dim 向上对齐为该值的整数倍
    norm_eps: float = 1e-5  # RMSNorm 使用的数值稳定项
    max_seq_len: int = 2048  # 模型支持的最大上下文长度
    dropout: float = 0.0  # Dropout 概率；0.0 表示不使用


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()                            # 初始化父类 nn.Module
        self.eps = eps                                # 防止除零
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习的缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)   # 计算最后一维的平方均值
        scale = torch.rsqrt(mean_square + self.eps)         # 计算平方根的倒数
        return self.weight * x * scale                      # 归一化并进行可学习缩放

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回形状均为 [end, dim // 2] 的 cos、sin 频率表。"""
    assert dim % 2 == 0

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    positions = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)

    return torch.cos(angles), torch.sin(angles)


def reshape_for_broadcast(
    freqs: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """将 [T, D/2] 变为 [1, T, 1, D/2]，用于和 Q/K 广播。"""
    assert freqs.shape == (x.shape[1], x.shape[-1])

    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq/xk: [B, T, H, D]
    freqs: [T, D/2]
    """
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(dim=-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(dim=-1)

    cos = reshape_for_broadcast(freqs_cos, xq_r)
    sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out = torch.stack(
        [xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos],
        dim=-1,
    ).flatten(-2)

    xk_out = torch.stack(
        [xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos],
        dim=-1,
    ).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (batch_size, seq_len, n_kv_heads, head_dim)
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]                         # (B, T, H_kv, 1, D)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # 若未启用 GQA，则 KV 头数等于 Q 头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.dim % args.n_heads == 0
        assert args.n_heads % self.n_kv_heads == 0

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # 输入 x: (B, T, dim)
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. 得到 Q、K、V
        xq = self.wq(x)  # (B, T, n_heads * head_dim)
        xk = self.wk(x)  # (B, T, n_kv_heads * head_dim)
        xv = self.wv(x)

        # 2. 拆成多个头
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 3. 只旋转 Q、K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 4. GQA：将较少的 KV 头扩展到与 Q 头数一致
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 5. 转为 (B, H, T, D)，便于逐头计算注意力
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. QK^T / sqrt(D)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 7. causal mask：当前位置不能看未来 token
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        # 8. softmax 后加权求和 V
        weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = weights @ xv  # (B, H, T, D)

        # 9. 拼接所有头，再投影回 dim
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int],
        multiple_of: int,
        dropout: float,
    ):
        super().__init__()

        # Llama 2 默认的中间层维度计算
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of
            )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU
        hidden = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(hidden))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()

        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-Norm Attention + residual
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
        )

        # Pre-Norm FFN + residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.params = args
        self.vocab_size = args.vocab_size

        # token id: (B, T) → token embedding: (B, T, dim)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        # 堆叠多个 Transformer Block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, args)
                for layer_id in range(args.n_layers)
            ]
        )

        # 最终归一化与词表输出层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 输入词嵌入与输出词表投影共享权重
        self.output.weight = self.tok_embeddings.weight

        # 预计算 RoPE 表；head_dim 必须是偶数
        head_dim = args.dim // args.n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            end=args.max_seq_len,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w3.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * args.n_layers),
                )

        self.last_loss = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, seq_len = tokens.shape

        if seq_len > self.params.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.params.max_seq_len}"
            )

        # (B, T) → (B, T, dim)
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 取当前序列所需长度的 RoPE 频率
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        # 依次经过所有 Transformer Block
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            # 训练：每个位置都输出 logits，并计算交叉熵
            logits = self.output(h)  # (B, T, vocab_size)

            self.last_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            # 推理：只需要最后一个位置预测下一个 token
            logits = self.output(h[:, [-1], :])  # (B, 1, vocab_size)
            self.last_loss = None

        return logits

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    top_values, _ = torch.topk(
                        logits,
                        k=min(top_k, logits.size(-1)),
                        dim=-1,
                    )
                    logits = logits.masked_fill(
                        logits < top_values[:, [-1]],
                        float("-inf"),
                    )

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
