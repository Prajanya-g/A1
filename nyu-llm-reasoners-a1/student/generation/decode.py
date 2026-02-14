"""§6 Decode from the language model.

This script lives at: student/generation/decode.py

Supports: prompt completion, max generated tokens, temperature scaling, top-p (nucleus) sampling,
and stopping at an end-of-text token (e.g. <|endoftext|>).
"""

import torch


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature for sampling. temperature > 0."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def _apply_top_p(
    probs: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Nucleus (top-p): keep smallest set of tokens whose cumulative prob >= top_p (Holtzman et al., 2020)."""
    if top_p >= 1.0:
        return probs
    if top_p <= 0:
        raise ValueError("top_p must be in (0, 1]")
    probs_sorted, indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(probs_sorted, dim=-1)
    prev_cumsum = cumsum - probs_sorted
    mask = (cumsum < top_p) | ((cumsum >= top_p) & (prev_cumsum < top_p))
    probs_sorted = probs_sorted.where(mask, torch.zeros_like(probs_sorted))
    probs_sorted = probs_sorted / probs_sorted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    probs = torch.zeros_like(probs).scatter_(-1, indices, probs_sorted)
    return probs


def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate a completion from the model given a prompt.

    Autoregressively samples from the model until an end-of-text token is produced
    or max_new_tokens is reached. Applies temperature scaling and optional top-p
    (nucleus) sampling to the next-token distribution at each step.

    This function lives in: student/generation/decode.py

    Args:
        model: Transformer LM with .forward(token_ids) -> logits (batch, seq, vocab).
        prompt_ids: Token IDs of the prompt; shape (1, prompt_len) or (prompt_len,).
        max_new_tokens: Maximum number of tokens to generate after the prompt.
        temperature: Softmax temperature (> 0). Higher = more random.
        top_p: Nucleus sampling threshold in (0, 1]. 1.0 = no filtering.
        end_of_text_token_id: If set, stop when this token is sampled (e.g. <|endoftext|>).
        eos_token_id: Alias for end_of_text_token_id; used if end_of_text_token_id is None.

    Returns:
        Tensor of shape (1, prompt_len + num_generated) containing prompt + completion.
    """
    eos_id = end_of_text_token_id if end_of_text_token_id is not None else eos_token_id
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    device = next(model.parameters()).device
    context_length = getattr(model, "context_length", prompt_ids.size(1))
    # Keep only the last context_length tokens of the prompt as context
    if prompt_ids.size(1) > context_length:
        context = prompt_ids[:, -context_length:].to(device)
    else:
        context = prompt_ids.to(device)
    generated = list(prompt_ids[0].tolist())
    prompt_len = len(generated)

    for _ in range(max_new_tokens):
        logits = model(context)
        next_logits = logits[:, -1, :]
        next_logits = _apply_temperature(next_logits, temperature)
        probs = torch.softmax(next_logits, dim=-1)
        if top_p < 1.0:
            probs = _apply_top_p(probs, top_p)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        next_tensor = torch.tensor([[next_id]], device=device, dtype=context.dtype)
        context = torch.cat([context, next_tensor], dim=1)[:, -context_length:]

    return torch.tensor([generated], device=prompt_ids.device, dtype=prompt_ids.dtype)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one next token from logits with optional temperature and top-p.

    Args:
        logits: (batch, vocab) or (vocab,).
        temperature: Scale logits by 1/temperature before softmax.
        top_p: Nucleus sampling threshold; 1.0 = no filtering.

    Returns:
        Sampled token IDs, shape (batch,) or scalar.
    """
    logits = _apply_temperature(logits, temperature)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
