from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .chunked_prefill import chunked_prefill
from .tokenizer import Tokenizer
from .transformer import TransformerLM

END_OF_TEXT = "<|endoftext|>"


def sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """对 logits 做温度缩放和核采样过滤，再采样 token ID。

    最后一维是词表维度，返回形状为 ``logits.shape[:-1]``。
    """
    temperature = float(temperature)
    top_p = float(top_p)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    output_shape = logits.shape[:-1]
    vocabulary_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary_size).to(dtype=torch.float32)
    sorted_logits, sorted_indices = torch.sort(
        flat_logits / temperature,
        dim=-1,
        descending=True,
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)

    #保留让累计概率首次达到 top_p 的那个 token；只去掉后面的项，至少能留下一项。
    remove = cumulative_probabilities - sorted_probabilities >= top_p
    sorted_logits.masked_fill_(remove, float("-inf"))
    filtered_probabilities = torch.softmax(sorted_logits, dim=-1)

    sampled_sorted_positions = torch.multinomial(
        filtered_probabilities,
        num_samples=1,
        generator=generator,
    )
    sampled_token_ids = sorted_indices.gather(-1, sampled_sorted_positions)
    return sampled_token_ids.reshape(output_shape)


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int] | torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """生成一条续写，返回提示词与生成内容拼接后的 token ID。"""
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    prompt = torch.as_tensor(prompt_token_ids, dtype=torch.long, device=device)
    if prompt.ndim != 1:
        raise ValueError(f"prompt_token_ids must be one-dimensional, got shape {tuple(prompt.shape)}")
    if prompt.numel() == 0:
        raise ValueError("prompt_token_ids must contain at least one token")

    context_length = getattr(model, "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("model must expose a positive integer context_length")
    if eos_token_id is not None and int(prompt[-1]) == eos_token_id:
        return prompt.clone()

    sequence = torch.empty(
        prompt.numel() + max_new_tokens,
        dtype=torch.long,
        device=device,
    )
    sequence[: prompt.numel()] = prompt
    current_length = prompt.numel()
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context_start = max(0, current_length - context_length)
            model_input = sequence[context_start:current_length].unsqueeze(0)
            next_token_logits = model(model_input)[0, -1]
            next_token = sample_top_p(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            sequence[current_length] = next_token
            current_length += 1
            if eos_token_id is not None and int(next_token) == eos_token_id:
                break
    finally:
        model.train(was_training)

    #复制有效部分，避免返回的视图一直占用未用完的预分配存储。
    return sequence[:current_length].clone()


@torch.inference_mode()
def incremental_forward(
    model: TransformerLM,
    token_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """用新建的 KV 缓存逐个输入真实 token，计算各位置的 logits。

    主要用于核对缓存推理的正确性。一维输入返回 ``[tokens, vocab]``，
    批量输入返回 ``[batch, tokens, vocab]``。
    """
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    tokens = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    unbatched = tokens.ndim == 1
    if unbatched:
        tokens = tokens.unsqueeze(0)
    elif tokens.ndim != 2:
        raise ValueError(f"token_ids must have shape [tokens] or [batch, tokens], got {tuple(tokens.shape)}")
    if tokens.shape[-1] == 0:
        raise ValueError("token_ids must contain at least one token")
    if tokens.shape[-1] > model.context_length:
        raise ValueError(f"token count cannot exceed context_length={model.context_length}, got {tokens.shape[-1]}")

    was_training = model.training
    model.eval()
    try:
        cache = model.new_kv_cache(batch_size=tokens.shape[0])
        pieces = [model(tokens[:, index : index + 1], kv_cache=cache) for index in range(tokens.shape[-1])]
        logits = torch.cat(pieces, dim=1)
    finally:
        model.train(was_training)
    return logits[0] if unbatched else logits


@torch.inference_mode()
def generate_cached(
    model: TransformerLM,
    prompt_token_ids: Sequence[int] | torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
    prefill_chunk_size: int | None = None,
) -> torch.Tensor:
    """为单次请求使用连续 KV 缓存，生成一条续写。

    默认一次预填充提示词；指定 prefill_chunk_size 后改为分块预填充。
    实际送入模型的前缀必须在上下文长度内；淘汰旧缓存会改变隐藏状态的语义，
    留给后续缓存管理阶段处理。
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if prefill_chunk_size is not None:
        if isinstance(prefill_chunk_size, bool) or not isinstance(prefill_chunk_size, int):
            raise TypeError(f"prefill_chunk_size must be an integer, got {type(prefill_chunk_size).__name__}")
        if prefill_chunk_size <= 0:
            raise ValueError(f"prefill_chunk_size must be positive, got {prefill_chunk_size}")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    prompt = torch.as_tensor(prompt_token_ids, dtype=torch.long, device=device)
    if prompt.ndim != 1:
        raise ValueError(f"prompt_token_ids must be one-dimensional, got shape {tuple(prompt.shape)}")
    if prompt.numel() == 0:
        raise ValueError("prompt_token_ids must contain at least one token")
    requested_length = prompt.numel() + max_new_tokens
    #最后采样出的 token 无须再送入模型，所以最多只缓存前 max_new_tokens - 1 个新 token。
    evaluated_length = prompt.numel() + max(0, max_new_tokens - 1)
    if evaluated_length > model.context_length:
        raise ValueError(
            f"tokens evaluated by the model cannot exceed context_length={model.context_length}, got {evaluated_length}"
        )
    if max_new_tokens == 0 or (eos_token_id is not None and int(prompt[-1]) == eos_token_id):
        return prompt.clone()

    sequence = torch.empty(requested_length, dtype=torch.long, device=device)
    sequence[: prompt.numel()] = prompt
    current_length = prompt.numel()
    was_training = model.training
    model.eval()
    try:
        cache_capacity = max(prompt.numel(), evaluated_length)
        cache = model.new_kv_cache(batch_size=1, max_seq_len=cache_capacity)
        prefill = chunked_prefill(
            model,
            prompt.unsqueeze(0),
            chunk_size=prompt.numel() if prefill_chunk_size is None else prefill_chunk_size,
            kv_cache=cache,
        )
        next_token_logits = prefill.last_logits[0]
        for output_index in range(max_new_tokens):
            next_token = sample_top_p(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            sequence[current_length] = next_token
            current_length += 1
            if eos_token_id is not None and int(next_token) == eos_token_id:
                break
            if output_index + 1 < max_new_tokens:
                next_token_logits = model(next_token.reshape(1, 1), kv_cache=cache)[0, -1]
    finally:
        model.train(was_training)

    return sequence[:current_length].clone()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 Transformer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--merges-file", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=_nonnegative_int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="use contiguous cached decoding (the evaluated prefix must fit the context window)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=_positive_int,
        help="fixed prefill chunk size; requires --use-kv-cache",
    )

    parser.add_argument("--context-length", type=_positive_int, default=256)
    parser.add_argument("--d-model", type=_positive_int, default=512)
    parser.add_argument("--num-layers", type=_positive_int, default=4)
    parser.add_argument("--num-heads", type=_positive_int, default=16)
    parser.add_argument("--d-ff", type=_positive_int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    args = parser.parse_args(argv)
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("the attention head dimension must be even for RoPE")
    if args.rope_theta <= 0:
        parser.error("--rope-theta must be positive")
    if args.prefill_chunk_size is not None and not args.use_kv_cache:
        parser.error("--prefill-chunk-size requires --use-kv-cache")
    return args


def _load_model_state(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    model_state = checkpoint.get("model", checkpoint)
    if not isinstance(model_state, dict):
        raise TypeError("checkpoint['model'] must contain a state dictionary")
    return model_state


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_files(
        args.vocab_file,
        args.merges_file,
        special_tokens=[END_OF_TEXT],
    )
    token_ids = sorted(tokenizer.vocab)
    if token_ids != list(range(len(token_ids))):
        raise ValueError("tokenizer vocabulary IDs must be contiguous and start at zero")
    eos_token_id = tokenizer.bytes_to_id[END_OF_TEXT.encode("utf-8")]

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    model.load_state_dict(_load_model_state(args.checkpoint, device))

    prompt_token_ids = tokenizer.encode(args.prompt)
    generation_function = generate_cached if args.use_kv_cache else generate
    generated_token_ids = generation_function(
        model,
        prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        **({"prefill_chunk_size": args.prefill_chunk_size} if args.use_kv_cache else {}),
    )
    print(tokenizer.decode(generated_token_ids.tolist()))


if __name__ == "__main__":
    main()
