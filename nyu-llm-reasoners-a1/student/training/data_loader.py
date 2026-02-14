"""§5.1 Batch loading: sample input sequences and next-token targets."""

import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size input sequences and next-token targets from a 1D token array.

    Each sample is a contiguous slice; targets are the next token at each position.
    Valid start indices: 0 to len(x) - context_length - 1.

    Args:
        x: 1D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each input/target sequence.
        device: PyTorch device string (e.g. 'cpu', 'cuda:0', 'mps').

    Returns:
        (inputs, targets): LongTensors (batch_size, context_length) on device.
    """
    n = len(x)
    max_start = n - context_length
    if max_start <= 0:
        raise ValueError(
            f"Dataset length {n} too small for context_length {context_length}"
        )
    starts = np.random.randint(0, max_start, size=batch_size)
    inputs = np.stack([x[s:s + context_length] for s in starts])
    targets = np.stack([x[s + 1:s + context_length + 1] for s in starts])
    inputs_t = torch.from_numpy(inputs).long().to(device)
    targets_t = torch.from_numpy(targets).long().to(device)
    return inputs_t, targets_t
