"""§3.1 Transformer language model: embed -> stack of blocks -> ln_final -> lm_head -> logits."""

import torch
from torch import nn

from student.model.components.embedding import Embedding
from student.model.components.linear import Linear
from student.model.components.rmsnorm import RMSNorm
from student.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Transformer language model with token embedding, num_layers blocks, final norm, and output projection."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_model, vocab_size, device=device, dtype=dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch_size, sequence_length). Returns (batch_size, sequence_length, vocab_size) logits."""
        x = self.token_embeddings(token_ids)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
