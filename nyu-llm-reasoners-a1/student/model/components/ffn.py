import torch
from torch import nn

from student.model.components.linear import Linear


def swiglu_d_ff(d_model: int) -> int:
    """Recommended d_ff: ~(8/3)*d_model, rounded to a multiple of 64."""
    raw = (8 / 3) * d_model
    return max(64, int(round(raw / 64) * 64))


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish): x * sigmoid(x). Uses torch.sigmoid for numerical stability."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: out = W2(silu(W1(x)) * W3(x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (..., d_model), output (..., d_model)."""
        gate = silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)
