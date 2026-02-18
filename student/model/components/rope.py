import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """Apply RoPE to (..., seq_len, d_k) using precomputed cos/sin and token positions."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # inv_freq: (d_k/2,) with theta^(-2i/d_k) for i in 0..d_k/2-1
        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        # positions: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # angles: (max_seq_len, d_k/2)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        """x: (..., seq_len, d_k), token_positions: (..., seq_len). Out: same shape as x."""
        # Index cos/sin by token positions; cast to x dtype for the multiply
        cos = self.cos_cached[token_positions].to(x.dtype)
        sin = self.sin_cached[token_positions].to(x.dtype)
        # Split x into even and odd dims: (..., seq_len, d_k/2) each
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Rotation: (a*cos - b*sin, a*sin + b*cos)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        # Interleave back to (..., seq_len, d_k)
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return out
