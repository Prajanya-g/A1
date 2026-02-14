import math

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply softmax along dimension dim. Same shape as x; dim is a probability distribution.

    Uses the max-subtraction trick for numerical stability:
    subtract max along dim before exp to avoid overflow.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention. Handles (batch_size, ..., seq_len, d_k) Q/K and (..., seq_len, d_v) V.

    Optional boolean mask (..., queries, keys): True = attend, False = zero probability.
    """
    d_k = Q.size(-1)
    scale = math.sqrt(d_k)
    scores = (Q @ K.transpose(-2, -1)) / scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V


class MultiheadSelfAttention(torch.nn.Module):
    """Causal multi-head self-attention. d_k = d_v = d_model / num_heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k * num_heads == d_model
        from student.model.components.linear import Linear

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        rope: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """x: (..., seq_len, d_model). Returns (..., seq_len, d_model).
        If rope and token_positions are given, apply RoPE to Q and K before attention."""
        # Project to Q, K, V: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        seq_len = x.size(-2)
        # Reshape to (..., num_heads, seq_len, d_k)
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*V.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        if rope is not None and token_positions is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        # Causal mask: (seq_len, seq_len), True where j <= i
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous()
        out = out.reshape(*out.shape[:-2], self.d_model)
        return self.output_proj(out)
