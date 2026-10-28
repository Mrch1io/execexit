import torch
from torch import nn

from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """用 x·sigmoid(x) 直接实现逐元素 SiLU。"""
    return x * torch.sigmoid(x)


class Swiglu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #SiLU 分支充当门控，逐元素调节另一分支的特征，再投影回模型维度。
        return self.w2(silu(self.w1(x)) * self.w3(x))
