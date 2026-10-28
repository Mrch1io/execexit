from __future__ import annotations
import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #1.半精度输入先转 float32；减去最大值，避免后面求指数时上溢。
    compute_dtype = (
        torch.float32
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits.dtype
    )
    stable_logits = logits.to(dtype=compute_dtype)
    stable_logits = stable_logits - stable_logits.amax(dim=-1, keepdim=True)
    #2.用 logsumexp 减目标 logit 求负对数似然，避免先算概率再取 log。
    log_normalizer = stable_logits.exp().sum(dim=-1).log()
    target_logits = stable_logits.gather(
        dim=-1,
        index=targets.to(device=logits.device, dtype=torch.long).unsqueeze(-1),
    ).squeeze(-1)
    return (log_normalizer - target_logits).mean()
