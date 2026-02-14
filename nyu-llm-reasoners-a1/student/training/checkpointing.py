"""§5.2 Save and load training checkpoints (model, optimizer, iteration)."""

import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model state, optimizer state, and iteration to a path or file-like object.

    Args:
        model: Module whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        iteration: Training iteration count to save.
        out: File path or binary file-like object to write to.
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint from path or file-like object; restore model and optimizer; return iteration.

    Args:
        src: File path or binary file-like object to read from.
        model: Module to restore (load_state_dict called).
        optimizer: Optimizer to restore (load_state_dict called).

    Returns:
        The iteration number that was saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
