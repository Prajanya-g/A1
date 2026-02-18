"""§3.5.1 RMSNorm: root mean square layer normalization."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm: normalize over last dim (d_model), then scale by learnable weight."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input (..., d_model); return same shape. Upcast to float32 for stability."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        out = x_norm * self.weight.to(torch.float32)
        return out.to(in_dtype)
