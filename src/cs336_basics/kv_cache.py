from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch


class KVCache:
    """为每个请求预分配注意力 K/V 存储。

    K 写入前应已应用 RoPE，V 保持融合 QKV 投影的结果。
    追加时复制数据到缓存自己的存储中，避免依赖调用方张量的后续修改。
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        storage_shape = (
            num_layers,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
        )
        self._keys = torch.empty(storage_shape, device=self.device, dtype=dtype)
        self._values = torch.empty_like(self._keys)
        self._lengths = [0] * num_layers

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(self._lengths)

    @property
    def allocated_bytes(self) -> int:
        return (self._keys.numel() + self._values.numel()) * self._keys.element_size()

    @property
    def storage_data_ptrs(self) -> tuple[int, int]:
        """返回底层存储地址，用于检查是否发生了重新分配。"""
        return self._keys.data_ptr(), self._values.data_ptr()

    def _validate_layer(self, layer_idx: int) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError(f"layer_idx must be an integer, got {type(layer_idx).__name__}")
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx must be in [0, {self.num_layers - 1}], got {layer_idx}")

    def _validate_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_prefix = (self.batch_size, self.num_heads)
        if tensor.ndim != 4 or tensor.shape[:2] != expected_prefix or tensor.shape[-1] != self.head_dim:
            expected = (self.batch_size, self.num_heads, "tokens", self.head_dim)
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if tensor.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}, got {tensor.dtype}")

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """把一块 token 的 K/V 复制到指定层，不重新分配存储。"""
        self._validate_layer(layer_idx)
        self._validate_tensor("key", key)
        self._validate_tensor("value", value)
        if key.shape != value.shape:
            raise ValueError(f"key and value must have the same shape, got {tuple(key.shape)} and {tuple(value.shape)}")

        token_count = key.shape[-2]
        if token_count == 0:
            raise ValueError("cannot append an empty key/value chunk")
        start = self._lengths[layer_idx]
        end = start + token_count
        if end > self.max_seq_len:
            raise OverflowError(
                f"layer {layer_idx} cache capacity exceeded: {start} + {token_count} > {self.max_seq_len}"
            )

        #先校验再写入，避免输入错误导致 K/V 只更新一边。
        self._keys[layer_idx, :, :, start:end].copy_(key)
        self._values[layer_idx, :, :, start:end].copy_(value)
        self._lengths[layer_idx] = end

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回指定层有效前缀的可读写视图。"""
        self._validate_layer(layer_idx)
        length = self._lengths[layer_idx]
        return (
            self._keys[layer_idx, :, :, :length],
            self._values[layer_idx, :, :, :length],
        )

    def length(self, layer_idx: int) -> int:
        self._validate_layer(layer_idx)
        return self._lengths[layer_idx]

    def uniform_length(self) -> int:
        """返回各层共有的长度；层间长度不一致说明缓存未完整更新。"""
        first = self._lengths[0]
        if any(length != first for length in self._lengths[1:]):
            raise RuntimeError(f"all cache layers must have equal lengths, got {self.lengths}")
        return first

    def reset(self) -> None:
        """清空有效长度，保留已分配的存储。"""
        self._lengths[:] = [0] * self.num_layers

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """多层缓存前向计算失败时，回滚各层的可见长度。"""
        starting_lengths = self._lengths.copy()
        try:
            yield
        except BaseException:
            #前向计算只在旧前缀后追加；回退长度后，未完成的写入就不会被 get 读到。
            self._lengths[:] = starting_lengths
            raise
