"""§4.5 Gradient clipping by global L2 norm."""

from collections.abc import Iterable

import torch

_EPS = 1e-6  # PyTorch default for numerical stability


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = _EPS,
) -> None:
    """Clip gradients of parameters in place so their global L2 norm is at most max_norm.

    Global norm = sqrt(sum over all parameters of ||param.grad||_2^2).
    If global norm > max_norm, scale all gradients by max_norm / (global_norm + eps).

    Args:
        parameters: Iterable of parameters whose gradients to clip.
        max_norm: Maximum allowed L2 norm of the combined gradients.
        eps: Small constant added for numerical stability (default 1e-6).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm_sq = sum(g.pow(2).sum().item() for g in grads)
    total_norm = total_norm_sq**0.5
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)