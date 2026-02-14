"""§4.4 Cosine annealing LR schedule (LLaMA-style) with linear warmup."""

import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """Learning rate at step t for cosine annealing with warmup.

    Warm-up (t < T_w): α_t = (t / T_w) * α_max.
    Cosine (T_w ≤ t ≤ T_c): α_t = α_min + (1/2)(1 + cos(π*(t-T_w)/(T_c-T_w)))(α_max - α_min).
    Post (t > T_c): α_t = α_min.

    Args:
        t: Current iteration (step) index.
        alpha_max: Maximum learning rate (reached at end of warmup).
        alpha_min: Minimum / final learning rate (reached at end of cosine).
        T_w: Number of warm-up iterations (warmup for t = 0, 1, ..., T_w - 1).
        T_c: Step index at which cosine ends (cosine runs t = T_w .. T_c inclusive).

    Returns:
        Learning rate to use at step t.
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    if t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        coeff = 0.5 * (1 + math.cos(progress * math.pi))
        return alpha_min + coeff * (alpha_max - alpha_min)
    return alpha_min
