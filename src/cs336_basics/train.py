"""CS336 作业 1 第 5.3 节要求的最小训练脚本。"""

import argparse
import copy
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch, load_token_dataset, token_dataset_identity
from .training_log import ExperimentLogger
from .losses import cross_entropy
from .optimizer import AdamW, clip_gradients, get_lr_cosine_schedule
from .transformer import TransformerLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer language model")

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--data-dtype", choices=("uint16", "uint32", "int32", "int64"))

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=32,
        help="Samples processed at once; gradients are accumulated to reach --batch-size.",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-end-step", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use bfloat16 autocast while keeping parameters and optimizer state in float32.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--experiment-log", type=Path)
    parser.add_argument("--experiment-name", default="training-run")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
    use_bfloat16: bool = False,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    #每次评估后恢复独立的随机流，固定评估样本，也不影响训练采样。
    rng = np.random.default_rng(0) if rng is None else rng
    rng_state = copy.deepcopy(rng.bit_generator.state)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    try:
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device, rng=rng)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                total_loss += cross_entropy(model(inputs), targets).item()
    finally:
        rng.bit_generator.state = rng_state
        model.train(was_training)
    return total_loss / num_batches


def _training_identity(
    args: argparse.Namespace,
    train_dataset: np.ndarray,
    val_dataset: np.ndarray,
    eval_rng: np.random.Generator,
) -> dict[str, Any]:
    #这些选项只改变记录方式或停止位置；学习率日程的终点在下面单独保存。
    operational_keys = {
        "train_data",
        "val_data",
        "max_steps",
        "log_interval",
        "eval_interval",
        "checkpoint_path",
        "checkpoint_interval",
        "resume_from",
        "experiment_log",
        "experiment_name",
    }
    config = {key: value for key, value in vars(args).items() if key not in operational_keys}
    config["cosine_end_step"] = args.cosine_end_step if args.cosine_end_step is not None else args.max_steps
    config["device"] = str(torch.device(args.device))
    config["attention_backend"] = "reference"
    plan_rng = copy.deepcopy(eval_rng)
    starts = plan_rng.integers(
        0,
        len(val_dataset) - args.context_length,
        size=(args.eval_batches, args.micro_batch_size),
        dtype=np.int64,
    )
    return {
        "protocol_version": 1,
        "config": config,
        "datasets": {
            "train": token_dataset_identity(args.train_data, train_dataset),
            "validation": token_dataset_identity(args.val_data, val_dataset),
        },
        "eval_sample_plan_sha256": hashlib.sha256(starts.astype("<i8", copy=False).tobytes()).hexdigest(),
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "cpu_threads": torch.get_num_threads()},
    }


def train(args: argparse.Namespace):
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    for name in ("max_steps", "context_length", "log_interval", "eval_interval", "eval_batches", "checkpoint_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    device = torch.device(args.device)
    if args.bfloat16 and device.type != "cuda":
        raise ValueError("--bfloat16 currently requires a CUDA device")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_seed, eval_seed = np.random.SeedSequence(args.seed).spawn(2)
    rng_generators = {"train": np.random.default_rng(train_seed), "eval": np.random.default_rng(eval_seed)}
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    disk_dtype = np.dtype(args.data_dtype) if args.data_dtype is not None else None
    train_dataset = load_token_dataset(args.train_data, dtype=disk_dtype, vocab_size=args.vocab_size)
    val_dataset = load_token_dataset(args.val_data, dtype=disk_dtype, vocab_size=args.vocab_size)
    if min(len(train_dataset), len(val_dataset)) <= args.context_length:
        raise ValueError("each dataset must contain at least context_length + 1 tokens")
    training_identity = _training_identity(args, train_dataset, val_dataset, rng_generators["eval"])

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    iteration = 0
    if args.resume_from is not None:
        iteration = load_checkpoint(
            args.resume_from,
            model,
            optimizer,
            restore_rng=True,
            rng_generators=rng_generators,
            expected_training_state=training_identity,
        )
        if iteration > args.max_steps:
            raise ValueError("checkpoint iteration exceeds --max-steps")
        print(f"resumed from step {iteration}")
    initial_iteration = iteration

    cosine_end_step = training_identity["config"]["cosine_end_step"]
    started_at = time.perf_counter()
    logger = None
    if args.experiment_log is not None:
        logger = ExperimentLogger(
            args.experiment_log,
            experiment_name=args.experiment_name,
            config={
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "training_identity": training_identity,
                "initial_iteration": initial_iteration,
                "initial_tokens_seen": initial_iteration * args.batch_size * args.context_length,
            },
        )
    accumulation_steps = args.batch_size // args.micro_batch_size
    model.train()

    progress = tqdm(
        total=args.max_steps,
        initial=iteration,
        desc="training",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    status = "failed"
    try:
        while iteration < args.max_steps:
            learning_rate = get_lr_cosine_schedule(
                it=iteration,
                max_learning_rate=args.max_lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_steps,
                cosine_cycle_iters=cosine_end_step,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                inputs, targets = get_batch(
                    train_dataset,
                    args.micro_batch_size,
                    args.context_length,
                    device,
                    rng=rng_generators["train"],
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.bfloat16,
                ):
                    loss = cross_entropy(model(inputs), targets)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"non-finite training loss before optimizer step {iteration + 1}")
                #1.每个小批次先除以累积次数，累加后才对应整个大批次的平均梯度。
                (loss / accumulation_steps).backward()
                accumulated_loss += loss_value / accumulation_steps
            #2.先累积完整梯度再裁剪；逐个小批次裁剪可能改变累积梯度的方向。
            if args.max_grad_norm > 0:
                clip_gradients(model.parameters(), args.max_grad_norm)
            optimizer.step()
            iteration += 1
            elapsed = time.perf_counter() - started_at
            progress.update(1)
            progress.set_postfix(
                loss=f"{accumulated_loss:.4f}",
                lr=f"{learning_rate:.2e}",
                tokens_s=f"{(iteration - initial_iteration) * args.batch_size * args.context_length / elapsed:,.0f}",
            )

            if iteration % args.log_interval == 0 or iteration == args.max_steps:
                tqdm.write(
                    f"step={iteration} train_loss={accumulated_loss:.6f} lr={learning_rate:.3e} time={elapsed:.1f}s"
                )
                if logger is not None:
                    logger.log_metrics(
                        step=iteration,
                        wall_clock_seconds=elapsed,
                        split="train",
                        loss=accumulated_loss,
                        metrics={
                            "perplexity": math.exp(accumulated_loss),
                            "learning_rate": learning_rate,
                            "tokens_per_second": (iteration - initial_iteration)
                            * args.batch_size
                            * args.context_length
                            / elapsed,
                        },
                    )

            if iteration % args.eval_interval == 0 or iteration == args.max_steps:
                val_loss = evaluate(
                    model,
                    val_dataset,
                    args.micro_batch_size,
                    args.context_length,
                    device,
                    args.eval_batches,
                    args.bfloat16,
                    rng=rng_generators["eval"],
                )
                if not math.isfinite(val_loss):
                    raise FloatingPointError(f"non-finite validation loss at step {iteration}")
                progress.set_postfix(
                    loss=f"{accumulated_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    lr=f"{learning_rate:.2e}",
                )
                tqdm.write(f"step={iteration} val_loss={val_loss:.6f}")
                if logger is not None:
                    logger.log_metrics(
                        step=iteration,
                        wall_clock_seconds=time.perf_counter() - started_at,
                        split="validation",
                        loss=val_loss,
                        metrics={"perplexity": math.exp(val_loss)},
                    )

            if args.checkpoint_path is not None and iteration % args.checkpoint_interval == 0:
                args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model,
                    optimizer,
                    iteration,
                    args.checkpoint_path,
                    rng_generators=rng_generators,
                    training_state=training_identity,
                )
                print(f"saved checkpoint: {args.checkpoint_path}")

        if args.checkpoint_path is not None and iteration % args.checkpoint_interval != 0:
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                optimizer,
                iteration,
                args.checkpoint_path,
                rng_generators=rng_generators,
                training_state=training_identity,
            )
            print(f"saved final checkpoint: {args.checkpoint_path}")
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        progress.close()
        if logger is not None:
            logger.close(
                status=status,
                step=iteration,
                wall_clock_seconds=time.perf_counter() - started_at,
            )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
