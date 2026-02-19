"""§5.3 Training loop — extended with ablation flags for §7.3 experiments.

New flags vs. original:
  --no_rmsnorm   Remove all RMSNorms (layer_norm_ablation)
  --post_norm    Use post-norm instead of pre-norm (pre_norm_ablation)
  --no_rope      Remove RoPE positional embeddings (no_pos_emb / NoPE)
  --use_silu     Replace SwiGLU with plain SiLU FFN; d_ff should be set to
                 4*d_model=2048 to match parameter count (swiglu_ablation)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_tokens_memmap(path: str | Path) -> np.ndarray:
    path = Path(path)
    return np.load(path, mmap_mode="r") if path.suffix.lower() == ".npy" else np.load(path)


def _eval_loss(model, data, device, batch_size, context_length, num_batches) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, context_length, str(device))
            total += cross_entropy_loss(model(x), y).item()
    model.train()
    return total / num_batches if num_batches else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM")
    # Data
    p.add_argument("--train_tokens", required=True)
    p.add_argument("--valid_tokens", default="")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--context_length", type=int, default=256)
    # Model
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Ablation flags
    p.add_argument("--no_rmsnorm", action="store_true", help="Remove all RMSNorms (layer_norm_ablation)")
    p.add_argument("--post_norm", action="store_true", help="Use post-norm instead of pre-norm (pre_norm_ablation)")
    p.add_argument("--no_rope", action="store_true", help="Remove RoPE entirely / NoPE (no_pos_emb)")
    p.add_argument("--use_silu", action="store_true", help="Use plain SiLU FFN instead of SwiGLU (swiglu_ablation)")
    # Optimizer
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--eps", type=float, default=1e-8)
    # Schedule
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--max_iters", type=int, default=10000)
    # Training
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (speeds up on H100)")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 (cuda only, not mps)")
    # Checkpointing & logging
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=50)
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="a1-tinystories")
    p.add_argument("--wandb_run_name", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # TF32 speeds up matmuls on Ampere/Hopper GPUs with negligible precision loss
    if args.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info("Loading train tokens...")
    train_data = _load_tokens_memmap(args.train_tokens)
    valid_data = None
    if args.valid_tokens and os.path.isfile(args.valid_tokens):
        logger.info("Loading valid tokens...")
        valid_data = _load_tokens_memmap(args.valid_tokens)

    logger.info(
        "Config: d_model=%d layers=%d heads=%d d_ff=%d | "
        "no_rmsnorm=%s post_norm=%s no_rope=%s use_silu=%s",
        args.d_model, args.num_layers, args.num_heads, args.d_ff,
        args.no_rmsnorm, args.post_norm, args.no_rope, args.use_silu,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        # ablation flags — your TransformerLM must accept and forward these
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        use_swiglu=not args.use_silu,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (%.1fM)", n_params, n_params / 1e6)

    if args.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

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
        lr = get_lr_cosine_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
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
            logger.info("iter %d  train_loss %.4f  lr %.2e", it + 1, loss.item(), lr)
            if args.wandb:
                try:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=it + 1)
                except Exception:
                    pass

        if valid_data is not None and (it + 1) % args.eval_interval == 0:
            val_loss = _eval_loss(model, valid_data, device, args.batch_size,
                                  args.context_length, args.eval_batches)
            logger.info("iter %d  val_loss %.4f", it + 1, val_loss)
            if args.wandb:
                try:
                    wandb.log({"val/loss": val_loss}, step=it + 1)
                except Exception:
                    pass

        if checkpoint_dir and (it + 1) % args.save_every == 0:
            ckpt = checkpoint_dir / f"ckpt_iter_{it + 1}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt)
            logger.info("Saved %s", ckpt)

    logger.info("Training done at iter %d", args.max_iters)


if __name__ == "__main__":
    main()
