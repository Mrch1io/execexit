from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import torch


def load_token_dataset(
    path: str | Path,
    *,
    dtype: npt.DTypeLike | None = None,
    vocab_size: int | None = None,
) -> npt.NDArray[np.integer]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"token dataset does not exist: {dataset_path}")

    expected_dtype = np.dtype(dtype) if dtype is not None else None
    if dataset_path.suffix.lower() == ".npy":
        dataset = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
        if expected_dtype is not None and dataset.dtype != expected_dtype:
            raise TypeError(f"dataset dtype is {dataset.dtype}, but {expected_dtype} was expected")
    else:
        if expected_dtype is None:
            raise ValueError("dtype is required when loading a raw binary token file")
        if dataset_path.stat().st_size % expected_dtype.itemsize != 0:
            raise ValueError(f"file size is not divisible by dtype item size {expected_dtype.itemsize}")
        dataset = np.memmap(dataset_path, dtype=expected_dtype, mode="r")

    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"dataset must contain integer token IDs, got {dataset.dtype}")
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")

    metadata_path = dataset_path.with_name(f"{dataset_path.name}.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("token dataset metadata must contain an object")
        if "token_count" in metadata and metadata["token_count"] != dataset.size:
            raise ValueError("token dataset metadata token_count does not match the file")
        if "dtype" in metadata and np.dtype(metadata["dtype"]) != dataset.dtype:
            raise ValueError("token dataset metadata dtype does not match the file")

    if vocab_size is not None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        min_token = int(dataset.min())
        max_token = int(dataset.max())
        if min_token < 0 or max_token >= vocab_size:
            raise ValueError(
                "dataset contains token IDs outside the vocabulary range "
                f"[0, {vocab_size}): observed [{min_token}, {max_token}]"
            )

    return dataset


def token_dataset_identity(path: str | Path, dataset: npt.NDArray[np.integer]) -> dict[str, Any]:
    """以有界内存计算文件内容指纹，运行之间允许路径变化。

    memmap 使用期间，输入文件必须保持不变。
    读取完整文件是为了识别等长修改，文件大小和修改时间不足以确认内容身份。
    """
    dataset_path = Path(path)
    before = dataset_path.stat()
    with dataset_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = dataset_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("token dataset changed while computing its identity")
    metadata_path = dataset_path.with_name(f"{dataset_path.name}.json")
    return {
        "sha256": digest,
        "size_bytes": after.st_size,
        "dtype": dataset.dtype.str,
        "token_count": int(dataset.size),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.is_file() else None,
    }


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"dataset must contain integer token IDs, got {dataset.dtype}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    num_starting_positions = len(dataset) - context_length
    if num_starting_positions <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens "
            f"(got {len(dataset)} tokens and context_length={context_length})"
        )

    #未传入独立随机流时，保留作业接口使用全局随机数的行为。
    starts = (
        np.random.randint(0, num_starting_positions, size=batch_size)
        if rng is None
        else rng.integers(0, num_starting_positions, size=batch_size)
    )
    offsets = np.arange(context_length)  #[0 1 2 3 4]
    indices = starts[:, None] + offsets[None, :]
    #广播后的索引示例：
    #[[ 2,  3,  4,  5],
    #[ 7,  8,  9, 10],
    #[10, 11, 12, 13]]

    inputs = torch.as_tensor(dataset[indices], dtype=torch.long, device=device)
    #每个位置预测下一个 token，因此标签索引比输入右移一位。
    targets = torch.as_tensor(dataset[indices + 1], dtype=torch.long, device=device)
    return inputs, targets
