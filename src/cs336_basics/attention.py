from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right

from .kv_cache import KVCache
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .softmax import softmax

AttentionBackend = Literal["reference", "sdpa"]
ATTENTION_BACKENDS: tuple[AttentionBackend, ...] = ("reference", "sdpa")


def causal_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """查询位于连续 KV 前缀的末尾，因果掩码要按右下角对齐。

    Q 个新 token 对应 K 个缓存与新 token，查询 i 只能看见 j <= K - Q + i。
    非方阵的 is_causal=True 按左上角对齐，会错误遮住缓存前缀。
    符号偏置省去稠密 Q×K 掩码；具体内核由 PyTorch 或外层 sdpa_kernel 选择。
    """
    query_length, key_length = query.shape[-2], key.shape[-2]
    if not 0 < query_length <= key_length:
        raise ValueError("causal SDPA requires 0 < query length <= key length")
    if query_length == key_length:
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=True)
    if query_length == 1:
        #单个查询就是最新 token，已有的全部 key 都在它的可见范围内。
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=causal_lower_right(query_length, key_length), dropout_p=0.0
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """沿 key 维度计算 softmax(QK^T / sqrt(d_k))V。"""
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same final dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must contain the same number of tokens")

    d_k = query.shape[-1]
    #除以 sqrt(d_k)，避免点积随维度增大而把 softmax 推向饱和。
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
        #这里 True 表示可见；被遮住的位置填 -inf，softmax 后权重为零。
        logits = logits.masked_fill(~mask, float("-inf"))

    probabilities = softmax(logits, dim=-1)
    return torch.matmul(probabilities, value)


class CausalMultiHeadSelfAttention(nn.Module):
    """使用合并 QKV 投影的因果多头自注意力。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
        attention_backend: AttentionBackend = "reference",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unsupported attention_backend: {attention_backend!r}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend

        #把 Q、K、V 合成一次矩阵乘法，减少三次独立调用的调度开销。
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if rope_theta is None:
            self.rope: RotaryPositionalEmbedding | None = None
        else:
            if max_seq_len is None:
                raise ValueError("max_seq_len is required when RoPE is enabled")
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )

        if max_seq_len is None or attention_backend == "sdpa":
            causal_mask = torch.empty(0, 0, dtype=torch.bool, device=device)
        else:
            causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device).tril_()
        self.causal_mask: torch.Tensor
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 [..., S, d_model] 拆为 [..., num_heads, S, head_dim]。"""
        sequence_length = x.shape[-2]
        batch_shape = x.shape[:-2]
        return x.reshape(*batch_shape, sequence_length, self.num_heads, self.head_dim).transpose(-3, -2)

    def _get_causal_mask(
        self,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        #缓存前缀也占据位置，查询对应的掩码行要从 past_length 开始。
        if self.causal_mask.shape[-1] >= key_length:
            return self.causal_mask[
                past_length : past_length + query_length,
                :key_length,
            ]
        return torch.ones(query_length, key_length, dtype=torch.bool, device=device).tril_(diagonal=past_length)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        _positions_are_validated: bool = False,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {x.shape[-1]}")

        query_length = x.shape[-2]
        batch_shape = x.shape[:-2]
        if kv_cache is None:
            if layer_idx is not None:
                raise ValueError("layer_idx requires kv_cache")
            past_length = 0
        else:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            past_length = kv_cache.length(layer_idx)
            if token_positions is None:
                token_positions = torch.arange(
                    past_length,
                    past_length + query_length,
                    device=x.device,
                )

        #1.只投影新 token；RoPE 使用接在缓存后面的绝对位置。
        query, key, value = (self._split_heads(part) for part in self.qkv_proj(x).chunk(3, dim=-1))

        if self.rope is not None:
            query = self.rope(
                query,
                token_positions,
                _positions_are_validated=_positions_are_validated,
            )
            key = self.rope(
                key,
                token_positions,
                _positions_are_validated=_positions_are_validated,
            )

        if kv_cache is not None:
            #2.先写入旋转后的 key，后续解码就不必重复投影和旋转旧 token。
            assert layer_idx is not None
            kv_cache.append(layer_idx, key, value)
            key, value = kv_cache.get(layer_idx)

        key_length = key.shape[-2]

        if self.attention_backend == "sdpa":
            attended = causal_sdpa(query, key, value)
        else:
            attended = scaled_dot_product_attention(
                query,
                key,
                value,
                mask=self._get_causal_mask(
                    query_length,
                    key_length,
                    past_length,
                    x.device,
                ),
            )
        #3.先把同一 token 的各个头放到一起，再合并回 d_model。
        attended = attended.transpose(-3, -2).contiguous()
        attended = attended.reshape(*batch_shape, query_length, self.d_model)
        return self.output_proj(attended)
