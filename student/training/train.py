"""§5.3 Training loop: configurable hyperparameters, memmap data, checkpoints, logging.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from student.model.transformer_lm import TransformerLM
from student.training.checkpointing import load_checkpoint, save_checkpoint
from student.training.data_loader import get_batch
from student.training.gradient_clipping import clip_grad_norm_
from student.training.loss import cross_entropy_loss
from student.training.optimizer import AdamW
from student.training.scheduler import get_lr_cosine_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_tokens_memmap(path: str | Path) -> np.ndarray:
    """Load a 1D token array with memory mapping for large files."""
    path = Path(path)
    if not path.suffix.lower() == ".npy":
        return np.load(path)
    return np.load(path, mmap_mode="r")


def _eval_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int,
    context_length: int,
    num_batches: int,
) -> float:
    """Compute average loss over num_batches validation batches."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, context_length, str(device))
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches if num_batches else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM")
    # Data
    p.add_argument("--train_tokens", type=str, required=True, help="Train token .npy path (memmap)")
    p.add_argument("--valid_tokens", type=str, default="", help="Path to valid token .npy (optional)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=256)
    # Model
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    p.add_argument("--eps", type=float, default=1e-8)
    # Schedule
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--max_iters", type=int, default=10_000)
    # Training
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    # Checkpointing & logging
    p.add_argument("--checkpoint_dir", type=str, default="", help="Dir for checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    # W&B (optional)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="llm-reasoners")
    p.add_argument("--wandb_run_name", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Memory-efficient data loading
    logger.info("Loading train tokens (memmap)...")
    train_data = _load_tokens_memmap(args.train_tokens)
    valid_data = None
    if args.valid_tokens and os.path.isfile(args.valid_tokens):
        logger.info("Loading valid tokens (memmap)...")
        valid_data = _load_tokens_memmap(args.valid_tokens)

    # Model and optimizer
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        start_iter = load_checkpoint(args.resume, model, optimizer)
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name or None)
            wandb.config.update(vars(args))
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for it in range(start_iter, args.max_iters):
        # Learning rate
        lr = get_lr_cosine_schedule(
            it, args.lr_max, args.lr_min,
            args.warmup_iters, args.max_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (it + 1) % args.log_interval == 0:
            logger.info("iter %d loss %.4f lr %.2e", it + 1, loss.item(), lr)
            if args.wandb:
                try:
                    wandb.log(
                        {"train/loss": loss.item(), "train/lr": lr},
                        step=it + 1,
                    )
                except Exception:
                    pass

        do_eval = valid_data is not None and (it + 1) % args.eval_interval == 0
        if do_eval:
            val_loss = _eval_loss(
                model, valid_data, device,
                args.batch_size, args.context_length,
                args.eval_batches,
            )
            logger.info("iter %d val_loss %.4f", it + 1, val_loss)
            if args.wandb:
                try:
                    wandb.log({"val/loss": val_loss}, step=it + 1)
                except Exception:
                    pass

        do_save = checkpoint_dir and (it + 1) % args.save_every == 0
        if do_save:
            ckpt_path = checkpoint_dir / f"ckpt_iter_{it + 1}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            logger.info("Saved checkpoint %s", ckpt_path)

    logger.info("Training finished at iter %d", args.max_iters)


if __name__ == "__main__":
    main()
