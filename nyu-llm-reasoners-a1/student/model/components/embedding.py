"""§3.4.3 Embedding module: lookup table for token embeddings."""
import torch
from torch import nn


class Embedding(nn.Module):
    """Embedding lookup. Matrix shape (num_embeddings, embedding_dim), no nn.Embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Store with embedding_dim (d_model) as the final dimension
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weight[token_ids]
