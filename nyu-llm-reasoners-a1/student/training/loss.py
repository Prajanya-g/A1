"""Cross-entropy loss for language modeling: ℓ_i = -log softmax(o_i)[x_{i+1}], averaged over batch."""

import torch


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute average cross-entropy over batch with numerical stability.

    Per-example loss: ℓ_i = -log softmax(logits_i)[targets_i].
    Uses max-subtraction for stability and avoids forming softmax explicitly
    (log and exp cancel: -log_softmax(o)[k] = -o_k + log_sum_exp(o)).

    Args:
        logits: Unnormalized predictions; shape (..., vocab_size). Batch dimensions first.
        targets: Correct class index per example; shape (...). Must match logits.shape[:-1].

    Returns:
        Scalar tensor: mean of per-example NLL across all batch dimensions.
    """
    # Subtract max along vocab for numerical stability
    o_max = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = (logits - o_max).exp().sum(dim=-1).log() + o_max.squeeze(-1)
    # -log softmax(o)[target] = -o[target] + log_sum_exp(o)
    logits_at_target = logits.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)
    nll = -logits_at_target + log_sum_exp
    return nll.mean()
