from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from cs336_basics.chunked_prefill import chunked_forward, chunked_prefill
from cs336_basics.generation import generate, generate_cached
from cs336_basics.transformer import TransformerBlock, TransformerLM

CHUNK_SIZES = (16, 32, 64, 128, 256)


def _model(
    *,
    context_length: int = 320,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TransformerLM:
    torch.manual_seed(2026)
    return TransformerLM(
        vocab_size=67,
        context_length=context_length,
        d_model=24,
        num_layers=3,
        num_heads=4,
        d_ff=48,
        rope_theta=10_000.0,
        device=device,
        dtype=dtype,
    ).eval()


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_fixed_chunk_sizes_match_full_forward(chunk_size: int) -> None:
    model = _model()
    tokens = torch.randint(0, 67, (2, 257))

    with torch.inference_mode():
        expected = model(tokens)
        actual = chunked_forward(model, tokens, chunk_size=chunk_size)
        prefill = chunked_prefill(model, tokens, chunk_size=chunk_size)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prefill.last_logits, expected[:, -1], atol=1e-5, rtol=1e-5)
    assert prefill.kv_cache.lengths == (257,) * model.num_layers
    assert prefill.num_chunks == (257 + chunk_size - 1) // chunk_size


@pytest.mark.parametrize(
    ("prompt_length", "chunk_size", "expected_chunks"),
    [
        (15, 16, 1),
        (16, 16, 1),
        (17, 16, 2),
        (255, 256, 1),
        (256, 256, 1),
        (257, 256, 2),
        (320, 64, 5),
    ],
)
def test_chunk_boundaries_and_remainder(
    prompt_length: int,
    chunk_size: int,
    expected_chunks: int,
) -> None:
    model = _model()
    tokens = torch.randint(0, 67, (1, prompt_length))

    with torch.inference_mode():
        expected = model(tokens)
        prefill = chunked_prefill(model, tokens, chunk_size=chunk_size)

    assert prefill.num_chunks == expected_chunks
    assert prefill.kv_cache.lengths == (prompt_length,) * model.num_layers
    torch.testing.assert_close(prefill.last_logits, expected[:, -1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_chunked_prefill_batch_and_decode_handoff(batch_size: int) -> None:
    model = _model(context_length=64)
    tokens = torch.randint(0, 67, (batch_size, 39))

    with torch.inference_mode():
        expected = model(tokens)
        prefill = chunked_prefill(model, tokens[:, :33], chunk_size=16)
        pieces = [prefill.last_logits[:, None]]
        for position in range(33, tokens.shape[1]):
            pieces.append(model(tokens[:, position : position + 1], kv_cache=prefill.kv_cache))
        actual = torch.cat(pieces, dim=1)

    torch.testing.assert_close(actual, expected[:, 32:], atol=1e-5, rtol=1e-5)


def test_chunked_prefill_only_projects_the_final_prompt_token() -> None:
    model = _model()
    tokens = torch.randint(0, 67, (2, 257))
    cache = model.new_kv_cache(batch_size=2)
    storage_pointers = cache.storage_data_ptrs
    qkv_token_positions = 0
    lm_head_token_positions = 0

    def count_qkv(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal qkv_token_positions
        qkv_token_positions += inputs[0].shape[0] * inputs[0].shape[-2]

    def count_head(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal lm_head_token_positions
        lm_head_token_positions += inputs[0].shape[0] * inputs[0].shape[-2]

    qkv_hooks = [
        cast(TransformerBlock, layer).attn.qkv_proj.register_forward_pre_hook(count_qkv) for layer in model.layers
    ]
    head_hook = model.lm_head.register_forward_pre_hook(count_head)
    try:
        prefill = chunked_prefill(model, tokens, chunk_size=16, kv_cache=cache)
    finally:
        for hook in qkv_hooks:
            hook.remove()
        head_hook.remove()

    assert qkv_token_positions == model.num_layers * tokens.numel()
    assert lm_head_token_positions == tokens.shape[0]
    assert prefill.kv_cache.storage_data_ptrs == storage_pointers


def test_logits_to_keep_contract_and_skips_output_modules() -> None:
    model = _model(context_length=32)
    tokens = torch.randint(0, 67, (2, 9))

    with torch.inference_mode():
        expected = model(tokens)
        kept = model(tokens, logits_to_keep=1)

        cache = model.new_kv_cache(batch_size=2)
        norm_mock = Mock(wraps=model.ln_final.forward)
        head_mock = Mock(wraps=model.lm_head.forward)
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
        ):
            monkeypatch.setattr(model.ln_final, "forward", norm_mock)
            monkeypatch.setattr(model.lm_head, "forward", head_mock)
            skipped = model(tokens, kv_cache=cache, logits_to_keep=0)

    torch.testing.assert_close(kept, expected[:, -1:])
    assert skipped.shape == (2, 0, 67)
    assert skipped.numel() == 0
    assert cache.lengths == (9,) * model.num_layers
    norm_mock.assert_not_called()
    head_mock.assert_not_called()


@pytest.mark.parametrize(
    ("invalid", "error"),
    [(True, TypeError), (-1, ValueError), (10, ValueError), (1.5, TypeError)],
)
def test_logits_to_keep_rejects_invalid_values(
    invalid: Any,
    error: type[Exception],
) -> None:
    model = _model(context_length=16)
    tokens = torch.randint(0, 67, (1, 9))

    with pytest.raises(error):
        model(tokens, logits_to_keep=invalid)


@pytest.mark.parametrize(("invalid", "error"), [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_chunked_prefill_rejects_invalid_chunk_sizes(
    invalid: Any,
    error: type[Exception],
) -> None:
    model = _model()
    tokens = torch.randint(0, 67, (1, 9))

    with pytest.raises(error):
        chunked_prefill(model, tokens, chunk_size=invalid)


def test_chunked_prefill_preflights_small_cache_before_compute() -> None:
    model = _model(context_length=32)
    tokens = torch.randint(0, 67, (1, 17))
    projected_tokens = 0

    def count(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal projected_tokens
        projected_tokens += inputs[0].shape[-2]

    first_block = cast(TransformerBlock, model.layers[0])
    hook = first_block.attn.qkv_proj.register_forward_pre_hook(count)
    try:
        small = model.new_kv_cache(batch_size=1, max_seq_len=16)
        with pytest.raises(OverflowError, match="capacity"):
            chunked_prefill(model, tokens, chunk_size=8, kv_cache=small)
    finally:
        hook.remove()

    assert projected_tokens == 0


def test_chunked_prefill_appends_to_a_nonempty_prefix() -> None:
    model = _model(context_length=64)
    tokens = torch.randint(0, 67, (2, 41))
    cache = model.new_kv_cache(batch_size=2)

    with torch.inference_mode():
        expected = model(tokens)
        model(tokens[:, :7], kv_cache=cache, logits_to_keep=0)
        prefix_pointers = cache.storage_data_ptrs
        prefill = chunked_prefill(model, tokens[:, 7:], chunk_size=16, kv_cache=cache)

    assert prefill.past_tokens == 7
    assert prefill.input_tokens == 34
    assert prefill.total_tokens == 41
    assert cache.lengths == (41,) * model.num_layers
    assert cache.storage_data_ptrs == prefix_pointers
    torch.testing.assert_close(prefill.last_logits, expected[:, -1], atol=1e-5, rtol=1e-5)


def test_chunked_prefill_rolls_back_the_entire_request_on_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    tokens = torch.randint(0, 67, (1, 45))
    cache = model.new_kv_cache(batch_size=1)
    with torch.inference_mode():
        model(tokens[:, :5], kv_cache=cache, logits_to_keep=0)
    prefix = tuple(tensor.clone() for layer in range(model.num_layers) for tensor in cache.get(layer))
    failing_block = cast(TransformerBlock, model.layers[1])
    original_forward = failing_block.ffn.forward
    calls = 0

    def fail_on_third_chunk(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("late prefill failure")
        return original_forward(x)

    monkeypatch.setattr(failing_block.ffn, "forward", fail_on_third_chunk)
    with pytest.raises(RuntimeError, match="late prefill failure"):
        chunked_prefill(model, tokens[:, 5:], chunk_size=16, kv_cache=cache)

    assert cache.lengths == (5,) * model.num_layers
    after_failure = tuple(tensor for layer in range(model.num_layers) for tensor in cache.get(layer))
    for actual, expected in zip(after_failure, prefix, strict=True):
        torch.testing.assert_close(actual, expected)


def test_chunked_cached_generation_matches_unchunked() -> None:
    model = _model(context_length=32)
    model.train()
    prompt = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9]

    uncached = generate(
        model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(11),
    )
    unchunked = generate_cached(
        model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(11),
    )
    chunked = generate_cached(
        model,
        prompt,
        max_new_tokens=8,
        eos_token_id=None,
        temperature=0.8,
        top_p=0.9,
        prefill_chunk_size=4,
        generator=torch.Generator().manual_seed(11),
    )

    assert torch.equal(chunked, unchunked)
    assert torch.equal(chunked, uncached)
    assert model.training


@pytest.mark.parametrize(
    ("invalid", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
@pytest.mark.parametrize(
    ("max_new_tokens", "eos_token_id"),
    [(0, None), (4, 9)],
)
def test_cached_generation_validates_chunk_size_before_early_return(
    invalid: Any,
    error: type[Exception],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> None:
    model = _model(context_length=16)

    with pytest.raises(error, match="prefill_chunk_size"):
        generate_cached(
            model,
            [2, 9],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            prefill_chunk_size=invalid,
        )


@pytest.mark.parametrize(
    ("max_new_tokens", "eos_token_id"),
    [(0, None), (4, 9)],
)
def test_cached_generation_early_return_does_not_run_model(
    max_new_tokens: int,
    eos_token_id: int | None,
) -> None:
    model = _model(context_length=16)
    model.train()
    cache_factory = Mock(side_effect=AssertionError("cache must not be allocated on an early return"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(model, "new_kv_cache", cache_factory)
        actual = generate_cached(
            model,
            [2, 9],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            prefill_chunk_size=4,
        )

    torch.testing.assert_close(actual, torch.tensor([2, 9]))
    cache_factory.assert_not_called()
    assert model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_fixed_chunks_match_full_forward(dtype: torch.dtype) -> None:
    model = _model(device="cuda", dtype=dtype)
    tokens = torch.randint(0, 67, (2, 257), device="cuda")
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2

    with torch.inference_mode():
        expected = model(tokens)
        actual = chunked_forward(model, tokens, chunk_size=64)

    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
