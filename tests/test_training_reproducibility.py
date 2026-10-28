"""CPU trajectory tests: real Transformer, AdamW, accumulation, validation, resume."""

from __future__ import annotations

import copy
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import cs336_basics.train as training
from cs336_basics.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from cs336_basics.data import get_batch, load_token_dataset, token_dataset_identity


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous_threads = torch.get_num_threads()
    state = capture_rng_state()
    torch.set_num_threads(1)
    yield
    restore_rng_state(state)
    torch.set_num_threads(previous_threads)


def _assert_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right, strict=True):
            _assert_state_equal(item_left, item_right)
    else:
        assert left == right


def _arguments(tmp_path: Path, name: str, *, steps: int = 6):
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    if not train_path.exists():
        np.save(train_path, np.random.default_rng(17).integers(0, 32, 131, dtype=np.int64))
        np.save(val_path, np.random.default_rng(29).integers(0, 32, 73, dtype=np.int64))
    return training.parse_args(
        [
            "--train-data",
            str(train_path),
            "--val-data",
            str(val_path),
            "--device",
            "cpu",
            "--vocab-size",
            "32",
            "--context-length",
            "4",
            "--d-model",
            "16",
            "--num-layers",
            "2",
            "--num-heads",
            "2",
            "--d-ff",
            "32",
            "--batch-size",
            "4",
            "--micro-batch-size",
            "2",
            "--max-steps",
            str(steps),
            "--warmup-steps",
            "1",
            "--cosine-end-step",
            "6",
            "--max-lr",
            "0.003",
            "--eval-interval",
            "2",
            "--eval-batches",
            "2",
            "--log-interval",
            "1",
            "--checkpoint-path",
            str(tmp_path / name / "checkpoint.pt"),
            "--checkpoint-interval",
            "1",
            "--experiment-log",
            str(tmp_path / name / "metrics.jsonl"),
        ]
    )


def _run(args, monkeypatch):
    batches: list[torch.Tensor] = []
    eval_batches: list[torch.Tensor] = []
    trajectories: dict[int, Any] = {}
    original_batch = training.get_batch
    original_save = training.save_checkpoint

    def observe_batch(dataset, *arguments, **kwargs):
        result = original_batch(dataset, *arguments, **kwargs)
        if Path(dataset.filename) == args.train_data:
            batches.append(torch.stack(result).clone())
        else:
            eval_batches.append(torch.stack(result).clone())
        return result

    def observe_checkpoint(model, optimizer, iteration, destination, **kwargs):
        trajectories[iteration] = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
        original_save(model, optimizer, iteration, destination, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(training, "get_batch", observe_batch)
        patch.setattr(training, "save_checkpoint", observe_checkpoint)
        training.train(args)
    events = [json.loads(line) for line in args.experiment_log.read_text(encoding="utf-8").splitlines()]
    losses = [(event["step"], event["loss"]) for event in events if event.get("split") == "train"]
    return batches, trajectories, losses, events, eval_batches


def test_real_training_resume_replays_every_sample_loss_and_parameter(tmp_path, monkeypatch):
    uninterrupted = _run(_arguments(tmp_path, "full"), monkeypatch)
    first_args = _arguments(tmp_path, "first", steps=3)
    first = _run(first_args, monkeypatch)
    resumed_args = _arguments(tmp_path, "resumed")
    resumed_args.resume_from = first_args.checkpoint_path
    resumed = _run(resumed_args, monkeypatch)

    _assert_state_equal(uninterrupted[0], first[0] + resumed[0])
    _assert_state_equal(uninterrupted[1], first[1] | resumed[1])
    assert uninterrupted[2] == first[2] + resumed[2]
    assert resumed[3][0]["config"]["initial_tokens_seen"] == 3 * 4 * 4
    assert resumed[3][-1]["status"] == "completed"


def test_evaluation_frequency_does_not_change_training_trajectory(tmp_path, monkeypatch):
    frequent_args = _arguments(tmp_path, "frequent")
    frequent_args.eval_interval = 1
    frequent = _run(frequent_args, monkeypatch)
    rare_args = _arguments(tmp_path, "rare")
    rare_args.eval_interval = 5
    rare = _run(rare_args, monkeypatch)
    _assert_state_equal(frequent[:3], rare[:3])
    assert frequent[3][0]["config"]["training_identity"] == rare[3][0]["config"]["training_identity"]
    for batch_index in range(0, len(frequent[4]), frequent_args.eval_batches):
        _assert_state_equal(
            frequent[4][batch_index : batch_index + frequent_args.eval_batches], rare[4][: rare_args.eval_batches]
        )


@pytest.mark.parametrize("change", ["same_size_data", "learning_rate", "implicit_horizon", "eval_batches"])
def test_resume_rejects_changed_data_or_update_contract(tmp_path, monkeypatch, change):
    args = _arguments(tmp_path, "first", steps=2)
    _run(args, monkeypatch)
    resume = _arguments(tmp_path, "resumed")
    resume.resume_from = args.checkpoint_path
    if change == "same_size_data":
        tokens = np.load(resume.train_data)
        tokens[1] = (tokens[1] + 1) % 32
        np.save(resume.train_data, tokens)
    elif change == "learning_rate":
        resume.max_lr *= 2
    elif change == "implicit_horizon":
        resume.cosine_end_step = None
        resume.max_steps = 7
    else:
        resume.eval_batches += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        training.train(resume)
    assert not resume.checkpoint_path.exists()


def test_checkpoint_preserves_global_and_named_rng_streams_and_filelike_api():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    streams = {"train": np.random.default_rng(7), "eval": np.random.Generator(np.random.MT19937(8))}
    stream = io.BytesIO()
    save_checkpoint(model, optimizer, 3, stream, rng_generators=streams, training_state={"horizon": 6})

    def draw():
        return (
            random.random(),
            np.random.randn(4),
            torch.rand(4),
            {key: value.integers(0, 100, 4) for key, value in streams.items()},
        )

    expected = draw()
    draw()
    stream.seek(0)
    assert (
        load_checkpoint(
            stream, model, optimizer, restore_rng=True, rng_generators=streams, expected_training_state={"horizon": 6}
        )
        == 3
    )
    _assert_state_equal(expected, draw())


def test_legacy_checkpoint_is_loadable_but_cannot_claim_exact_resume():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": 4}, stream)
    stream.seek(0)
    assert load_checkpoint(stream, model, optimizer) == 4
    stream.seek(0)
    with pytest.raises(ValueError, match="legacy weights"):
        load_checkpoint(stream, model, optimizer, restore_rng=True)


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, 1, destination)
    original_bytes = destination.read_bytes()

    def interrupted_save(payload, output):
        output.write(b"partial checkpoint")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(model, optimizer, 2, destination)
    assert destination.read_bytes() == original_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint.pt"]


def test_explicit_sampling_stream_is_independent_of_global_numpy():
    tokens = np.arange(1000)
    stream = np.random.default_rng(43)
    first = get_batch(tokens, 256, 16, "cpu", rng=stream)
    np.random.random(10000)
    second = get_batch(tokens, 256, 16, "cpu", rng=np.random.default_rng(43))
    _assert_state_equal(first, second)
    assert torch.equal(first[0] + 1, first[1])


@pytest.mark.parametrize("metadata", [{"token_count": 99}, {"dtype": "uint32"}])
def test_dataset_rejects_stale_sidecar(tmp_path, metadata):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    path.with_name("tokens.bin.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_token_dataset(path, dtype="uint16", vocab_size=32)


def test_dataset_identity_is_content_based_and_checks_vocabulary(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(32, dtype=np.uint16))
    data = load_token_dataset(path, vocab_size=32)
    original = token_dataset_identity(path, data)
    moved = tmp_path / "moved.npy"
    moved.write_bytes(path.read_bytes())
    assert token_dataset_identity(moved, load_token_dataset(moved)) == original
    with pytest.raises(ValueError, match="outside the vocabulary"):
        load_token_dataset(path, vocab_size=31)


def test_evaluation_restores_state_on_failure():
    class FailingModel(torch.nn.Module):
        def forward(self, inputs):
            raise RuntimeError("evaluation failed")

    rng = np.random.default_rng(10)
    state = copy.deepcopy(rng.bit_generator.state)
    model = FailingModel().eval()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        training.evaluate(model, np.arange(50), 2, 4, torch.device("cpu"), 2, rng=rng)
    assert not model.training
    assert rng.bit_generator.state == state


def test_training_failure_has_terminal_log_and_no_new_checkpoint(tmp_path, monkeypatch):
    args = _arguments(tmp_path, "failed")

    def nonfinite_loss(logits, targets):
        return logits.sum() * float("nan")

    monkeypatch.setattr(training, "cross_entropy", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        training.train(args)
    events = [json.loads(line) for line in args.experiment_log.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "failed"
    assert events[-1]["step"] == 0
    assert not args.checkpoint_path.exists()
