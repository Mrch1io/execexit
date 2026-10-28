import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿 dim 计算 softmax；减去最大值不改变比例，又能避免指数上溢。"""
    max_values = x.amax(dim=dim, keepdim=True)
    exponentials = torch.exp(x - max_values)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)

