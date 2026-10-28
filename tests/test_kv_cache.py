from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from cs336_basics.generation import generate, generate_cached, incremental_forward
from cs336_basics.kv_cache import KVCache
from cs336_basics.transformer import TransformerBlock, TransformerLM


def _tiny_model(
    *,
    context_length: int = 32,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TransformerLM:
    torch.manual_seed(2026)
    return TransformerLM(
        vocab_size=97,
        context_length=context_length,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
        device=device,
        dtype=dtype,
    ).eval()


def _cache(*, max_seq_len: int = 8, dtype: torch.dtype = torch.float32) -> KVCache:
    return KVCache(
        num_layers=2,
        batch_size=2,
        num_heads=3,
        max_seq_len=max_seq_len,
        head_dim=4,
        device="cpu",
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("d_model", "num_heads", "message"),
    [
        (0, 4, "d_model and num_heads must be positive"),
        (32, 0, "d_model and num_heads must be positive"),
        (30, 4, "d_model must be divisible by num_heads"),
    ],
)
def test_transformer_validates_attention_dimensions(d_model: int, num_heads: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TransformerLM(
            vocab_size=97,
            context_length=32,
            d_model=d_model,
            num_layers=2,
            num_heads=num_heads,
            d_ff=64,
            rope_theta=10_000.0,
        )


def test_kv_cache_append_get_reset_and_layer_isolation() -> None:
    cache = _cache()
    key = torch.randn(2, 3, 4, 5).transpose(-1, -2)
    value = torch.randn(2, 3, 4, 5).transpose(-1, -2)
    expected_key = key.clone()
    expected_value = value.clone()

    cache.append(1, key, value)
    key.add_(100)
    value.mul_(0)

    cached_key, cached_value = cache.get(1)
    torch.testing.assert_close(cached_key, expected_key)
    torch.testing.assert_close(cached_value, expected_value)
    assert cached_key.shape == (2, 3, 5, 4)
    assert cache.lengths == (0, 5)
    assert cache.get(0)[0].shape == (2, 3, 0, 4)

    pointers = cache.storage_data_ptrs
    cache.reset()
    replacement_key = torch.randn(2, 3, 2, 4)
    replacement_value = torch.randn_like(replacement_key)
    cache.append(1, replacement_key, replacement_value)

    assert cache.storage_data_ptrs == pointers
    assert cache.lengths == (0, 2)
    torch.testing.assert_close(cache.get(1)[0], replacement_key)


def test_kv_cache_append_never_reallocates() -> None:
    cache = _cache()
    pointers = cache.storage_data_ptrs

    for _ in range(4):
        key = torch.randn(2, 3, 2, 4)
        value = torch.randn_like(key)
        cache.append(0, key, value)
        assert cache.storage_data_ptrs == pointers

    assert cache.length(0) == cache.max_seq_len


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        (torch.randn(2, 3, 2, 5), torch.randn(2, 3, 2, 5), ValueError),
        (torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4), ValueError),
        (torch.randn(2, 3, 2, 4), torch.randn(2, 3, 3, 4), ValueError),
        (torch.randn(2, 3, 2, 4, dtype=torch.float64), torch.randn(2, 3, 2, 4), TypeError),
    ],
)
def test_kv_cache_rejects_invalid_appends_atomically(
    key: torch.Tensor,
    value: torch.Tensor,
    error: type[Exception],
) -> None:
    cache = _cache()
    before = cache.lengths

    with pytest.raises(error):
        cache.append(0, key, value)

    assert cache.lengths == before


def test_kv_cache_rejects_invalid_layer_and_capacity_overflow_atomically() -> None:
    cache = _cache(max_seq_len=2)
    key = torch.randn(2, 3, 2, 4)
    value = torch.randn_like(key)

    with pytest.raises(IndexError):
        cache.append(2, key, value)
    cache.append(0, key, value)
    before = cache.lengths
    with pytest.raises(OverflowError):
        cache.append(0, key[:, :, :1], value[:, :, :1])
    assert cache.lengths == before


def test_incremental_forward_matches_full_forward() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 97, (2, 17))

    with torch.inference_mode():
        full_logits = model(tokens)
        cached_logits = incremental_forward(model, tokens)

    torch.testing.assert_close(full_logits, cached_logits, atol=1e-5, rtol=1e-5)


def test_irregular_cached_chunks_match_full_forward_and_use_rope_offsets() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 97, (2, 17))

    with torch.inference_mode():
        expected = model(tokens)
        cache = model.new_kv_cache(batch_size=2)
        pieces = []
        start = 0
        for chunk_size in (3, 5, 2, 7):
            end = start + chunk_size
            pieces.append(model(tokens[:, start:end], kv_cache=cache))
            start = end
        actual = torch.cat(pieces, dim=1)

    assert cache.lengths == (17, 17)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_cached_forward_accepts_exact_context_boundary_and_rejects_overflow() -> None:
    model = _tiny_model(context_length=8)
    tokens = torch.randint(0, 97, (2, 9))
    cache = model.new_kv_cache(batch_size=2)

    with torch.inference_mode():
        model(tokens[:, :7], kv_cache=cache)
        model(tokens[:, 7:8], kv_cache=cache)
        assert cache.lengths == (8, 8)
        before = cache.lengths
        with pytest.raises(OverflowError):
            model(tokens[:, 8:9], kv_cache=cache)

    assert cache.lengths == before


def test_cached_forward_rolls_back_all_layer_lengths_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    cache = model.new_kv_cache(batch_size=1)

    def fail(_x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model.layers[1].ffn, "forward", fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected failure"):
        model(torch.randint(0, 97, (1, 3)), kv_cache=cache)

    assert cache.lengths == (0, 0)


def test_cached_forward_rolls_back_if_output_head_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    cache = model.new_kv_cache(batch_size=1)
    tokens = torch.randint(0, 97, (1, 5))

    with torch.inference_mode():
        model(tokens[:, :2], kv_cache=cache)
        expected = model(tokens)[:, 2:]
    prefix = tuple(tensor.clone() for layer in range(model.num_layers) for tensor in cache.get(layer))
    original_forward = model.lm_head.forward

    def fail(_x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected output failure")

    monkeypatch.setattr(model.lm_head, "forward", fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected output failure"):
        model(tokens[:, 2:], kv_cache=cache)

    assert cache.lengths == (2, 2)
    after_failure = tuple(tensor for layer in range(model.num_layers) for tensor in cache.get(layer))
    for actual_tensor, expected_tensor in zip(after_failure, prefix, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)

    monkeypatch.setattr(model.lm_head, "forward", original_forward)
    with torch.inference_mode():
        retried = model(tokens[:, 2:], kv_cache=cache)
    torch.testing.assert_close(retried, expected, atol=1e-5, rtol=1e-5)
    assert cache.lengths == (5, 5)


def test_cached_forward_reuses_preallocated_position_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    cache = model.new_kv_cache(batch_size=1)
    original_arange = cast(Any, torch.arange)
    calls = 0

    def counting_arange(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", counting_arange)
    with torch.inference_mode():
        model(torch.randint(0, 97, (1, 3)), kv_cache=cache)

    assert calls == 0


def test_cached_forward_rejects_noncontiguous_explicit_positions() -> None:
    model = _tiny_model()
    cache = model.new_kv_cache(batch_size=1)

    with torch.inference_mode(), pytest.raises(ValueError, match="contiguous range"):
        model(
            torch.randint(0, 97, (1, 3)),
            token_positions=torch.tensor([4, 5, 6]),
            kv_cache=cache,
        )

    assert cache.lengths == (0, 0)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([False, True, True]),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
    ],
)
def test_cached_forward_rejects_invalid_position_dtype_or_batch_shape(
    positions: torch.Tensor,
) -> None:
    model = _tiny_model()
    cache = model.new_kv_cache(batch_size=1)

    with torch.inference_mode(), pytest.raises((TypeError, ValueError)):
        model(torch.randint(0, 97, (1, 3)), token_positions=positions, kv_cache=cache)

    assert cache.lengths == (0, 0)


def test_uncached_forward_rejects_positions_that_expand_the_token_batch() -> None:
    model = _tiny_model()

    with pytest.raises(ValueError, match="would expand"):
        model(
            torch.randint(0, 97, (1, 3)),
            token_positions=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        )


def test_cache_is_external_to_state_dict_and_request_local() -> None:
    model = _tiny_model()
    state_keys = tuple(model.state_dict())
    first = model.new_kv_cache(batch_size=1)
    second = model.new_kv_cache(batch_size=1)

    with torch.inference_mode():
        model(torch.randint(0, 97, (1, 2)), kv_cache=first)

    assert tuple(model.state_dict()) == state_keys
    assert first.lengths == (2, 2)
    assert second.lengths == (0, 0)
    assert first.storage_data_ptrs != second.storage_data_ptrs
    assert first.allocated_bytes == 2 * model.num_layers * model.context_length * model.d_model * 4


def test_cached_forward_is_inference_only_but_uncached_backward_is_unchanged() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 97, (1, 4))
    cache = model.new_kv_cache(batch_size=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        model(tokens, kv_cache=cache)

    model(tokens).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_cached_generation_matches_uncached_generation() -> None:
    model = _tiny_model(context_length=16)
    model.train()
    prompt = [4, 8, 15, 16]

    uncached = generate(
        model,
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(11),
    )
    cached = generate_cached(
        model,
        prompt,
        max_new_tokens=6,
        eos_token_id=None,
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(11),
    )

    torch.testing.assert_close(cached, uncached)
    assert model.training


def test_cached_generation_rejects_a_request_that_needs_eviction() -> None:
    model = _tiny_model(context_length=8)

    with pytest.raises(ValueError, match="cannot exceed context_length"):
        generate_cached(model, [1, 2, 3, 4, 5, 6], max_new_tokens=4, eos_token_id=None)


def test_cached_generation_allows_one_sample_at_full_context() -> None:
    model = _tiny_model(context_length=8)
    generated = generate_cached(
        model,
        list(range(8)),
        max_new_tokens=1,
        eos_token_id=None,
        generator=torch.Generator().manual_seed(7),
    )

    assert generated.shape == (9,)


def test_cached_forward_uses_autocast_activation_dtype() -> None:
    model = _tiny_model(dtype=torch.float32)
    tokens = torch.randint(0, 97, (1, 9))

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(tokens)
        cache = model.new_kv_cache(batch_size=1)
        first = model(tokens[:, :5], kv_cache=cache)
        second = model(tokens[:, 5:], kv_cache=cache)

    assert cache.dtype == torch.bfloat16
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, atol=2e-2, rtol=2e-2)


def test_cache_reduces_repeated_qkv_projection_work() -> None:
    model = _tiny_model()
    prompt_length = 8
    output_tokens = 5
    tokens = torch.randint(0, 97, (1, prompt_length + output_tokens - 1))
    projected_tokens = 0

    def count_tokens(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal projected_tokens
        projected_tokens += inputs[0].shape[-2]

    hooks = [
        cast(TransformerBlock, layer).attn.qkv_proj.register_forward_pre_hook(count_tokens) for layer in model.layers
    ]
    try:
        with torch.inference_mode():
            for index in range(output_tokens):
                model(tokens[:, : prompt_length + index])
        uncached_tokens = projected_tokens

        projected_tokens = 0
        cache = model.new_kv_cache(batch_size=1)
        with torch.inference_mode():
            model(tokens[:, :prompt_length], kv_cache=cache)
            for index in range(output_tokens - 1):
                start = prompt_length + index
                model(tokens[:, start : start + 1], kv_cache=cache)
        cached_tokens = projected_tokens
    finally:
        for hook in hooks:
            hook.remove()

    expected_uncached = model.num_layers * sum(prompt_length + index for index in range(output_tokens))
    expected_cached = model.num_layers * (prompt_length + output_tokens - 1)
    assert uncached_tokens == expected_uncached
    assert cached_tokens == expected_cached
    assert cached_tokens < uncached_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_cached_forward_matches_full_forward(dtype: torch.dtype) -> None:
    model = _tiny_model(device="cuda", dtype=dtype)
    tokens = torch.randint(0, 97, (2, 17), device="cuda")
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2

    with torch.inference_mode():
        expected = model(tokens)
        actual = incremental_forward(model, tokens)

    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
