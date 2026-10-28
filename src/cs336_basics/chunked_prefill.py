from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .kv_cache import KVCache
from .transformer import TransformerLM


@dataclass(frozen=True, slots=True)
class ChunkedPrefillResult:
    """提示词最后一个位置的 logits，以及当前请求已填充的缓存。"""

    last_logits: torch.Tensor
    kv_cache: KVCache
    input_tokens: int
    past_tokens: int
    total_tokens: int
    chunk_size: int
    num_chunks: int


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")


def _prepare_tokens(model: TransformerLM, token_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(f"token_ids must be a torch.Tensor, got {type(token_ids).__name__}")
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, tokens], got {tuple(token_ids.shape)}")
    if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must have a non-empty batch and sequence dimension")
    if token_ids.shape[1] > model.context_length:
        raise ValueError(f"prompt length cannot exceed context_length={model.context_length}, got {token_ids.shape[1]}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must have dtype torch.int32 or torch.int64, got {token_ids.dtype}")
    model_device = model.token_embeddings.weight.device
    if token_ids.device != model_device:
        raise ValueError(f"token_ids must be on {model_device}, got {token_ids.device}")
    return token_ids.to(dtype=torch.long)


def _prepare_cache(
    model: TransformerLM,
    tokens: torch.Tensor,
    kv_cache: KVCache | None,
) -> tuple[KVCache, int]:
    cache = model.new_kv_cache(batch_size=tokens.shape[0]) if kv_cache is None else kv_cache
    past_tokens = model.validate_kv_cache(tokens, cache)
    return cache, past_tokens


@torch.inference_mode()
def chunked_prefill(
    model: TransformerLM,
    token_ids: torch.Tensor,
    *,
    chunk_size: int,
    kv_cache: KVCache | None = None,
) -> ChunkedPrefillResult:
    """按固定块大小，对等长批次做预填充。

    中间块只更新 KV 缓存，只有提示词最后一个位置经过最终归一化和词表投影，
    省去不会用于采样的 logits 计算。任一块失败都会回滚本次预填充的可见长度。
    """
    _validate_chunk_size(chunk_size)
    tokens = _prepare_tokens(model, token_ids)
    cache, past_tokens = _prepare_cache(model, tokens, kv_cache)
    input_tokens = tokens.shape[1]
    num_chunks = math.ceil(input_tokens / chunk_size)

    was_training = model.training
    model.eval()
    try:
        with cache.transaction():
            final_logits: torch.Tensor | None = None
            #复用同一缓存，RoPE 位置接着已有前缀继续，不能在每块开头重新从 0 计算。
            for start in range(0, input_tokens, chunk_size):
                end = min(start + chunk_size, input_tokens)
                is_final_chunk = end == input_tokens
                chunk_logits = model(
                    tokens[:, start:end],
                    kv_cache=cache,
                    logits_to_keep=1 if is_final_chunk else 0,
                )
                if is_final_chunk:
                    final_logits = chunk_logits[:, -1]
            if final_logits is None:
                raise RuntimeError("chunked prefill produced no final logits")
    finally:
        model.train(was_training)

    return ChunkedPrefillResult(
        last_logits=final_logits,
        kv_cache=cache,
        input_tokens=input_tokens,
        past_tokens=past_tokens,
        total_tokens=past_tokens + input_tokens,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
    )


@torch.inference_mode()
def chunked_forward(
    model: TransformerLM,
    token_ids: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """按固定大小分块，在给定真实前缀的条件下返回每个位置的 logits。

    用于第 2 阶段的数值等价性对照；实际生成使用 chunked_prefill，
    只保留提示词最后一个位置的 logits。
    """
    _validate_chunk_size(chunk_size)
    tokens = _prepare_tokens(model, token_ids)
    cache = model.new_kv_cache(batch_size=tokens.shape[0])
    prompt_tokens = tokens.shape[1]

    was_training = model.training
    model.eval()
    try:
        with cache.transaction():
            output: torch.Tensor | None = None
            for start in range(0, prompt_tokens, chunk_size):
                end = min(start + chunk_size, prompt_tokens)
                chunk_logits = model(tokens[:, start:end], kv_cache=cache)
                if output is None:
                    output = chunk_logits.new_empty(
                        tokens.shape[0],
                        prompt_tokens,
                        chunk_logits.shape[-1],
                    )
                output[:, start:end].copy_(chunk_logits)
            if output is None:
                raise RuntimeError("chunked forward produced no logits")
    finally:
        model.train(was_training)

    return output
