"""§3.5 Pre-norm Transformer block: x = x + attn(ln1(x)); x = x + ffn(ln2(x)). Uses RoPE in attention."""

import torch
from torch import nn

from student.model.components.attention import MultiheadSelfAttention
from student.model.components.ffn import SwiGLU
from student.model.components.rmsnorm import RMSNorm
from student.model.components.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: residual + attention(ln1(x)), residual + ffn(ln2(x))."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model). Returns (batch, seq_len, d_model)."""
        seq_len = x.size(-2)
        token_positions = torch.arange(
            seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        # Pre-norm + residual: x = x + attn(ln1(x)) with RoPE
        h = self.ln1(x)
        x = x + self.attn(h, token_positions=token_positions, rope=self.rope)
        # Pre-norm + residual: x = x + ffn(ln2(x))
        x = x + self.ffn(self.ln2(x))
        return x
