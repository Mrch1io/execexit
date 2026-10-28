import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        #平方与均值用 float32 计算，减少半精度下的溢出和舍入误差。
        x = x.to(dtype=torch.float32)
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * inverse_rms * self.weight).to(input_dtype)
