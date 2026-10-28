from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from .attention import ATTENTION_BACKENDS, AttentionBackend, CausalMultiHeadSelfAttention
from .embedding import Embedding
from .kv_cache import KVCache
from .linear import Linear
from .rmsnorm import RMSNorm
from .swiglu import Swiglu


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        attention_backend: AttentionBackend = "reference",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            attention_backend=attention_backend,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        _positions_are_validated: bool = False,
    ) -> torch.Tensor:
        #归一化放在分支内部，残差路径保留原输入，梯度也能直接沿这条路径传播。
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            _positions_are_validated=_positions_are_validated,
        )
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context_length = context_length
        self.attention_backend = attention_backend
        position_ids = torch.arange(context_length, device=device)
        self.position_ids: torch.Tensor
        self.register_buffer("position_ids", position_ids, persistent=False)
        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    attention_backend=attention_backend,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def new_kv_cache(
        self,
        batch_size: int,
        *,
        max_seq_len: int | None = None,
    ) -> KVCache:
        """为每个请求独立分配连续的 KV 缓存。"""
        capacity = self.context_length if max_seq_len is None else max_seq_len
        if capacity > self.context_length:
            raise ValueError(f"max_seq_len cannot exceed model context_length={self.context_length}, got {capacity}")
        return KVCache(
            num_layers=self.num_layers,
            batch_size=batch_size,
            num_heads=self.num_heads,
            max_seq_len=capacity,
            head_dim=self.head_dim,
            device=self.token_embeddings.weight.device,
            dtype=self._inference_dtype(),
        )

    def _inference_dtype(self) -> torch.dtype:
        """按当前推理上下文确定 QKV 激活的 dtype，包含自动混合精度。"""
        weight = self.token_embeddings.weight
        if torch.is_autocast_enabled(weight.device.type):
            return torch.get_autocast_dtype(weight.device.type)
        return weight.dtype

    def validate_kv_cache(self, token_ids: torch.Tensor, kv_cache: KVCache) -> int:
        """在模型计算前检查整次追加，避免只写入部分层后才发现输入不合法。"""
        if token_ids.ndim != 2:
            raise ValueError(
                f"cached forward requires token_ids with shape [batch, tokens], got {tuple(token_ids.shape)}"
            )
        if token_ids.shape[-1] == 0:
            raise ValueError("cached forward requires at least one token")
        if torch.is_grad_enabled():
            raise RuntimeError("KV cache is inference-only; use torch.inference_mode() or torch.no_grad()")
        expected = {
            "num_layers": self.num_layers,
            "batch_size": token_ids.shape[0],
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }
        for name, value in expected.items():
            if getattr(kv_cache, name) != value:
                raise ValueError(f"cache {name} must be {value}, got {getattr(kv_cache, name)}")
        weight = self.token_embeddings.weight
        if kv_cache.device != weight.device:
            raise ValueError(f"cache must be on {weight.device}, got {kv_cache.device}")
        expected_dtype = self._inference_dtype()
        if kv_cache.dtype != expected_dtype:
            raise TypeError(f"cache must have dtype {expected_dtype}, got {kv_cache.dtype}")

        past_length = kv_cache.uniform_length()
        end = past_length + token_ids.shape[-1]
        limit = min(self.context_length, kv_cache.max_seq_len)
        if end > limit:
            raise OverflowError(
                f"cached sequence would exceed capacity: {past_length} + {token_ids.shape[-1]} > {limit}"
            )
        return past_length

    def _resolve_token_positions(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None,
        *,
        past_length: int,
        cached: bool,
    ) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        model_device = self.token_embeddings.weight.device
        expected_positions = self.position_ids[past_length : past_length + sequence_length]
        if past_length + sequence_length > self.context_length:
            raise IndexError(
                f"token positions must be in [0, {self.context_length - 1}], "
                f"got end position {past_length + sequence_length - 1}"
            )
        if token_positions is None:
            return expected_positions

        positions = token_positions.to(device=model_device)
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_positions must have dtype torch.int32 or torch.int64, got {positions.dtype}")
        if cached:
            #缓存按连续前缀存储，RoPE 的位置必须与缓存槽位对应。
            if positions.ndim != 1 or not torch.equal(positions, expected_positions):
                raise ValueError(
                    "cached token_positions must be the contiguous range "
                    f"[{past_length}, {past_length + sequence_length})"
                )
            return expected_positions

        if positions.ndim == 0 or positions.shape[-1] != sequence_length:
            raise ValueError(
                f"token_positions must end with sequence length {sequence_length}, got shape {tuple(positions.shape)}"
            )
        try:
            broadcast_shape = torch.broadcast_shapes(positions.shape[:-1], token_ids.shape[:-1])
        except RuntimeError as error:
            raise ValueError(
                f"token_positions batch shape {tuple(positions.shape[:-1])} cannot broadcast "
                f"to token_ids batch shape {tuple(token_ids.shape[:-1])}"
            ) from error
        if broadcast_shape != token_ids.shape[:-1]:
            raise ValueError(
                f"token_positions batch shape {tuple(positions.shape[:-1])} would expand "
                f"token_ids batch shape {tuple(token_ids.shape[:-1])} to {tuple(broadcast_shape)}"
            )
        if positions.numel() and (positions.min() < 0 or positions.max() >= self.context_length):
            raise IndexError(f"token positions must be in [0, {self.context_length - 1}]")
        return positions

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim == 0:
            raise ValueError("token_ids must have at least one dimension")
        if logits_to_keep is not None:
            if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
                raise TypeError(f"logits_to_keep must be an integer, got {type(logits_to_keep).__name__}")
            if not 0 <= logits_to_keep <= token_ids.shape[-1]:
                raise ValueError(f"logits_to_keep must be in [0, {token_ids.shape[-1]}], got {logits_to_keep}")
        past_length = 0
        if kv_cache is not None:
            past_length = self.validate_kv_cache(token_ids, kv_cache)
        token_positions = self._resolve_token_positions(
            token_ids,
            token_positions,
            past_length=past_length,
            cached=kv_cache is not None,
        )
        x = self.token_embeddings(token_ids)

        #任一层失败时统一回退缓存长度，避免各层保留不同长度的前缀。
        transaction = kv_cache.transaction() if kv_cache is not None else nullcontext()
        with transaction:
            for layer_idx, layer in enumerate(self.layers):
                layer_output = layer(
                    x,
                    token_positions=token_positions,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx if kv_cache is not None else None,
                    _positions_are_validated=True,
                )
                if not isinstance(layer_output, torch.Tensor):
                    raise RuntimeError("standard TransformerBlock must return a Tensor")
                x = layer_output

            #只裁剪输出投影的输入；整段 token 仍需经过所有层并写入缓存。
            if logits_to_keep == 0:
                return x.new_empty(*x.shape[:-2], 0, self.lm_head.out_features)
            if logits_to_keep is not None:
                x = x[..., -logits_to_keep:, :]
            return self.lm_head(self.ln_final(x))
