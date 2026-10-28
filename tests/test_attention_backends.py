from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from cs336_basics.attention import AttentionBackend, causal_sdpa
from cs336_basics.transformer import TransformerLM


def tiny_model(backend: AttentionBackend) -> TransformerLM:
    torch.manual_seed(71)
    return TransformerLM(
        vocab_size=47,
        context_length=17,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000,
        attention_backend=backend,
        dtype=torch.float64,
    )


def test_cached_causal_alignment_has_an_analytic_counterexample():
    #把Q和K都设成0，可见位置的分数就一样，softmax会把权重平均分给它们。
    #前面已经缓存了3个位置，两个新查询分别能看到前4个、前5个值。
    #V是[1,2,3,4,5]，取平均就是10/4=2.5和15/5=3.0。
    #如果掩码错误地从左上角对齐，就只能看到前1个、前2个值，得到1.0和1.5。
    key = torch.zeros(1, 1, 5, 4, dtype=torch.float64)
    value = torch.arange(1, 6, dtype=torch.float64).reshape(1, 1, 5, 1)
    query = torch.zeros(1, 1, 2, 4, dtype=torch.float64)
    torch.testing.assert_close(causal_sdpa(query, key, value).flatten(), torch.tensor([2.5, 3.0]).double())
    wrong = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.testing.assert_close(wrong.flatten(), torch.tensor([1.0, 1.5]).double())
    assert causal_sdpa(query[..., :1, :], key, value).item() == pytest.approx(3.0)
    assert F.scaled_dot_product_attention(query[..., :1, :], key, value, is_causal=True).item() == pytest.approx(1.0)


@pytest.mark.parametrize("query_length", [0, 6])
def test_causal_sdpa_rejects_queries_outside_the_contiguous_cache(query_length):
    query = torch.zeros(1, 1, query_length, 4)
    key = value = torch.zeros(1, 1, 5, 4)
    with pytest.raises(ValueError, match="query length"):
        causal_sdpa(query, key, value)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_backend_switch_preserves_weights_full_logits_and_gradients(batch_shape):
    reference, sdpa = tiny_model("reference"), tiny_model("sdpa")
    assert reference.state_dict().keys() == sdpa.state_dict().keys()
    for name, parameter in reference.state_dict().items():
        assert torch.equal(parameter, sdpa.state_dict()[name])
    masks = [buffer for name, buffer in sdpa.named_buffers() if name.endswith("causal_mask")]
    assert len(masks) == sdpa.num_layers and all(mask.numel() == 0 for mask in masks)
    tokens = torch.randint(0, 47, (*batch_shape, 11))
    expected, actual = reference(tokens), sdpa(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, a), (other, b) in zip(reference.named_parameters(), sdpa.named_parameters(), strict=True):
        assert name == other
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(b.grad, a.grad, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("chunks", [(3, 5, 2, 7), (16, 1), (1,) * 17])
@torch.inference_mode()
def test_sdpa_irregular_prefill_decode_positions_and_context_boundary(batch_size, chunks):
    reference, sdpa = tiny_model("reference").eval(), tiny_model("sdpa").eval()
    tokens = torch.randint(0, 47, (batch_size, 17))
    expected = reference(tokens)
    cache = sdpa.new_kv_cache(batch_size)
    outputs = []
    offset = 0
    for length in chunks:
        positions = torch.arange(offset, offset + length)
        outputs.append(sdpa(tokens[:, offset : offset + length], token_positions=positions, kv_cache=cache))
        offset += length
        assert cache.lengths == (offset, offset)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-8, rtol=1e-7)
    with pytest.raises(OverflowError):
        sdpa(tokens[:, :1], kv_cache=cache)
    assert cache.lengths == (17, 17)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("batch_shape", [(2, 2), (2, 3), (2, 1, 3)])
@torch.inference_mode()
def test_broadcast_positions_match_individual_sequences(backend, batch_shape):
    model = tiny_model(backend).eval()
    tokens = torch.randint(0, 47, (*batch_shape, 4))
    positions = torch.arange(1, batch_shape[-1] + 1)[:, None] * torch.arange(4)
    expanded = positions.expand_as(tokens)
    expected = torch.stack([
        model(sequence, token_positions=sequence_positions)
        for sequence, sequence_positions in zip(tokens.reshape(-1, 4), expanded.reshape(-1, 4), strict=True)
    ]).reshape(*tokens.shape, 47)

    torch.testing.assert_close(model(tokens, token_positions=positions), expected, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(model(tokens, token_positions=expanded), expected, atol=1e-8, rtol=1e-7)


@torch.inference_mode()
def test_sdpa_preserves_selective_logits_and_rejects_wrong_cached_positions():
    model = tiny_model("sdpa").eval()
    tokens = torch.randint(0, 47, (1, 9))
    expected = model(tokens)
    cache = model.new_kv_cache(1)
    with patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
        output = model(tokens[:, :4], kv_cache=cache, logits_to_keep=0)
        assert output.shape == (1, 0, 47)
        head.assert_not_called()
        actual = model(tokens[:, 4:], kv_cache=cache, logits_to_keep=1)
        assert head.call_count == 1
    torch.testing.assert_close(actual, expected[:, -1:], atol=1e-8, rtol=1e-7)
    with pytest.raises(ValueError, match="token_positions"):
        model(tokens[:, :1], token_positions=torch.tensor([0]), kv_cache=cache)
    assert cache.lengths == (9, 9)


@torch.inference_mode()
def test_sdpa_failure_rolls_back_all_cache_layers_then_allows_retry():
    model = tiny_model("sdpa").eval()
    tokens = torch.randint(0, 47, (1, 8))
    cache = model.new_kv_cache(1)
    model(tokens[:, :3], kv_cache=cache)
    original = F.scaled_dot_product_attention
    calls = 0

    def fail_second_layer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected SDPA failure after one layer appended")
        return original(*args, **kwargs)

    with patch("cs336_basics.attention.F.scaled_dot_product_attention", side_effect=fail_second_layer):
        with pytest.raises(RuntimeError, match="injected"):
            model(tokens[:, 3:4], kv_cache=cache)
    assert cache.lengths == (3, 3)
    actual = model(tokens[:, 3:], kv_cache=cache)
    torch.testing.assert_close(actual, model(tokens)[:, 3:], atol=1e-8, rtol=1e-7)


def test_unknown_attention_backend_is_rejected():
    with pytest.raises(ValueError, match="attention_backend"):
        tiny_model(cast(AttentionBackend, "unknown"))
