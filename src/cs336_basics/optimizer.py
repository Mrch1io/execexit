from collections.abc import Callable, Iterable
import math
from typing import Any

import torch


class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
    ) :
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or lr < 0.0
        ):
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": float(lr)})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                step = state.get("step", 0)
                #这里按更新次数的平方根衰减步长；没有梯度的参数不会推进计数。
                step_size = lr / math.sqrt(step + 1)
                parameter.add_(gradient, alpha=-step_size)
                state["step"] = step + 1
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        lr = float(lr)
        beta1, beta2 = (float(beta) for beta in betas)
        eps = float(eps)
        weight_decay = float(weight_decay)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["second_moment"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                #1.权重衰减单独作用于参数，不混入梯度的矩估计，这就是 AdamW 的解耦。
                parameter.mul_(1.0 - lr * weight_decay)

                #2.一阶矩平滑更新方向，二阶矩记录梯度尺度，用来调节各参数的步长。
                first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )

                #3.零初始化会让早期矩估计偏向 0；把偏差修正并入学习率，再用二阶矩缩放更新。
                adjusted_lr = (
                    lr * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
                )
                denominator = second_moment.sqrt().add_(eps)
                parameter.addcdiv_(first_moment, denominator, value=-adjusted_lr)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    max_learning_rate = float(max_learning_rate)
    min_learning_rate = float(min_learning_rate)

    if it < 0:
        raise ValueError(f"Iteration must be non-negative, got {it}")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError(
            "cosine_cycle_iters must be greater than warmup_iters, "
            f"got {cosine_cycle_iters} and {warmup_iters}"
        )
    if min_learning_rate < 0.0:
        raise ValueError(
            f"min_learning_rate must be non-negative, got {min_learning_rate}"
        )
    if max_learning_rate < min_learning_rate:
        raise ValueError(
            "max_learning_rate must be greater than or equal to "
            f"min_learning_rate, got {max_learning_rate} and {min_learning_rate}"
        )

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_multiplier * (
        max_learning_rate - min_learning_rate
    )


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """按所有参数梯度的整体 L2 范数做原地裁剪。"""
    max_l2_norm = float(max_l2_norm)
    if max_l2_norm < 0.0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}")

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return

    norm_device = gradients[0].device
    per_gradient_norms = torch.stack(
        [
            torch.linalg.vector_norm(gradient.detach(), ord=2).to(norm_device)
            for gradient in gradients
        ]
    )
    
    #所有梯度共用一个缩放系数，才能限制整体范数而不改变拼接后的梯度方向。
    total_norm = torch.linalg.vector_norm(per_gradient_norms, ord=2)
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)

    for gradient in gradients:
        gradient.mul_(scale.to(gradient.device))
