"""LoRA 的冻结与计数工具。"""

import torch
from torch import nn


def freeze_for_lora(module: nn.Module) -> None:
    """输入: 含 LoRA 的 module。
    返回: 无。
    """
    for name, parameter in module.named_parameters():
        parameter.requires_grad = "lora_" in name


def trainable_parameters(module: nn.Module) -> int:
    """输入: module。
    返回: 可训练参数数量。
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """输入: batch 与 device。
    返回: 设备上的 batch。
    """
    values, labels = batch
    return values.to(device), labels.to(device)
