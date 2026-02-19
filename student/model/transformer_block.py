"""§3.5 Pre-norm Transformer block: x = x + attn(ln1(x)); x = x + ffn(ln2(x)). Uses RoPE in attention."""

import torch
from torch import nn

from student.model.components.attention import MultiheadSelfAttention
from student.model.components.ffn import SiLUFFN, SwiGLU
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
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rmsnorm = use_rmsnorm
        self.pre_norm = pre_norm
        self.use_rope = use_rope
        d_k = d_model // num_heads
        self.ln1 = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else nn.Identity()
        )
        self.rope = (
            RotaryPositionalEmbedding(
                theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device, dtype=dtype
            )
            if use_rope
            else None
        )
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else nn.Identity()
        )
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if use_swiglu
            else SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model). Returns (batch, seq_len, d_model)."""
        seq_len = x.size(-2)
        token_positions = torch.arange(
            seq_len, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        rope_arg = self.rope if self.use_rope else None

        if self.pre_norm:
            # Pre-norm: x = x + attn(ln1(x)); x = x + ffn(ln2(x))
            h = self.ln1(x)
            x = x + self.attn(h, token_positions=token_positions, rope=rope_arg)
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-norm: x = ln1(x + attn(x)); x = ln2(x + ffn(x))
            x = self.ln1(x + self.attn(x, token_positions=token_positions, rope=rope_arg))
            x = self.ln2(x + self.ffn(x))
        return x
