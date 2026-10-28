from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """将最后一维按相邻两项配对，施加旋转位置编码。"""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]

        #三角函数表由位置和频率确定，可重新生成；随模型迁移设备，但不存入权重。
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        _positions_are_validated: bool = False,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected last dimension {self.d_k}, got {x.shape[-1]}")

        if token_positions is None:
            if x.shape[-2] > self.max_seq_len:
                raise IndexError(f"sequence length must not exceed {self.max_seq_len}, got {x.shape[-2]}")
            token_positions = torch.arange(x.shape[-2], device=self.cos_cache.device)
            _positions_are_validated = True
        else:
            token_positions = token_positions.to(device=self.cos_cache.device)

        if (
            not _positions_are_validated
            and token_positions.numel()
            and (token_positions.min() < 0 or token_positions.max() >= self.max_seq_len)
        ):
            raise IndexError(f"token positions must be in [0, {self.max_seq_len - 1}]")

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        #只补一个头维度；缺少的前导批次维度由 PyTorch 自动广播。
        if cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        #每对特征做二维旋转，Q 与 K 的点积由此包含相对位置差。
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


#早期学习脚本仍使用这个名字。
RoPE = RotaryPositionalEmbedding
