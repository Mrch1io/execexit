from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, IO, Any, cast

import numpy as np
import torch

CheckpointFile = str | os.PathLike[str] | BinaryIO | IO[bytes]


def _encode_numpy_state(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _encode_numpy_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode_numpy_state(item) for item in value)
    return value


def _decode_numpy_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: _decode_numpy_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_decode_numpy_state(item) for item in value)
    return value


def capture_rng_state(generators: Mapping[str, np.random.Generator] | None = None) -> dict[str, Any]:
    """用张量和 weights_only 可加载的 Python 值保存随机数状态。

    纯 CPU 运行不为保存未使用的随机流而初始化 CUDA。
    """
    return {
        "python": random.getstate(),
        "numpy": _encode_numpy_state(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "generators": {
            name: _encode_numpy_state(generator.bit_generator.state) for name, generator in (generators or {}).items()
        },
    }


def restore_rng_state(state: Mapping[str, Any], generators: Mapping[str, np.random.Generator] | None = None) -> None:
    generators = generators or {}
    saved_generators = state["generators"]
    if saved_generators.keys() != generators.keys():
        raise ValueError("checkpoint RNG stream names do not match the supplied generators")
    for name, generator in generators.items():
        if saved_generators[name]["bit_generator"] != generator.bit_generator.state["bit_generator"]:
            raise ValueError(f"checkpoint bit generator does not match stream {name!r}")
    cuda_states = state["torch_cuda"]
    if cuda_states is not None and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA RNG device count differs from the current runtime")
    #恢复状态才能接着原来的随机序列采样；重新设种子会从头开始。
    random.setstate(state["python"])
    np.random.set_state(_decode_numpy_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    for name, generator in generators.items():
        generator.bit_generator.state = _decode_numpy_state(saved_generators[name])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointFile,
    *,
    rng_generators: Mapping[str, np.random.Generator] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """保存模型、优化器和随机数状态，保留作业要求的位置参数接口。

    路径目标先写入临时文件并 fsync，再原子替换，避免覆盖成不完整的检查点。
    传入文件对象时，持久化由调用方负责。
    """
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint iteration must be a non-negative int")
    checkpoint = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "rng": capture_rng_state(rng_generators),
        "training_state": dict(training_state) if training_state is not None else None,
    }
    if not isinstance(out, (str, os.PathLike)):
        torch.save(checkpoint, out)
        return
    destination = Path(cast(str | os.PathLike[str], out))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    src: CheckpointFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    restore_rng: bool = False,
    rng_generators: Mapping[str, np.random.Generator] | None = None,
    expected_training_state: Mapping[str, Any] | None = None,
) -> int:
    """恢复权重、优化器并返回迭代步数，也支持旧格式文件。

    精确续训需要显式启用；训练入口同时提供随机流和预期的配置、数据标识。
    """
    parameter = next(model.parameters(), None)
    map_location = parameter.device if parameter is not None else torch.device("cpu")
    checkpoint: Any = torch.load(src, map_location=map_location, weights_only=True)

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a dictionary")
    required_keys = {"model", "optimizer", "iteration"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise KeyError(f"checkpoint is missing required keys: {sorted(missing_keys)}")

    iteration = checkpoint["iteration"]
    if type(iteration) is not int:
        raise TypeError(f"checkpoint iteration must be an int, got {type(iteration).__name__}")
    if iteration < 0:
        raise ValueError("checkpoint iteration must be non-negative")
    version = checkpoint.get("format_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported checkpoint format version: {version}")
    if expected_training_state is not None and checkpoint.get("training_state") != dict(expected_training_state):
        raise ValueError("checkpoint training/data identity mismatch or missing; exact resume is not safe")
    if restore_rng and (version != 2 or "rng" not in checkpoint):
        raise ValueError("checkpoint has no restorable RNG state; legacy weights are not an exact resume")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        restore_rng_state(checkpoint["rng"], rng_generators)
    return iteration
