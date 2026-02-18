#!/usr/bin/env python3
"""Run TinyStories experiment with specified hyperparameters.

Hyperparameters (from assignment):
  vocab_size 10000
  context_length 256
  d_model 512
  d_ff 1344  (~8/3 * d_model, multiple of 64)
  rope_theta 10000
  num_layers 4, num_heads 16  (~17M non-embedding params)
  total tokens ~327,680,000  => batch_size * max_iters * context_length

With batch_size=128, context_length=256:
  max_iters = 327_680_000 / (128 * 256) = 10_000

Usage:
  uv run python scripts/run_tinystories_experiment.py [--wandb]
  Requires data/train_tokens.npy (run student.tinystories_tokenize_analysis first).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_TOKENS = DATA_DIR / "train_tokens.npy"
VALID_TOKENS = DATA_DIR / "valid_tokens.npy"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "tinystories"

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
ROPE_THETA = 10_000.0
NUM_LAYERS = 4
NUM_HEADS = 16

BATCH_SIZE = 128
MAX_ITERS = 10_000  # 128 * 10_000 * 256 = 327_680_000 total tokens
WARMUP_ITERS = 500
SAVE_EVERY = 1000
LOG_INTERVAL = 50
EVAL_INTERVAL = 500
EVAL_BATCHES = 50


def main() -> None:
    if not TRAIN_TOKENS.exists():
        print(
            f"Tokenized data not found at {TRAIN_TOKENS}. "
            "Run tokenization first:\n"
            "  uv run python -m student.tinystories_tokenize_analysis\n"
            "This trains a 10K BPE and writes data/train_tokens.npy and "
            "data/valid_tokens.npy.",
            file=sys.stderr,
        )
        sys.exit(1)

    use_wandb = "--wandb" in sys.argv
    device = "mps" if sys.platform == "darwin" else "cuda:0"
    argv = [
        "--train_tokens", str(TRAIN_TOKENS),
        "--vocab_size", str(VOCAB_SIZE),
        "--context_length", str(CONTEXT_LENGTH),
        "--d_model", str(D_MODEL),
        "--d_ff", str(D_FF),
        "--rope_theta", str(ROPE_THETA),
        "--num_layers", str(NUM_LAYERS),
        "--num_heads", str(NUM_HEADS),
        "--batch_size", str(BATCH_SIZE),
        "--max_iters", str(MAX_ITERS),
        "--warmup_iters", str(WARMUP_ITERS),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--save_every", str(SAVE_EVERY),
        "--log_interval", str(LOG_INTERVAL),
        "--eval_interval", str(EVAL_INTERVAL),
        "--eval_batches", str(EVAL_BATCHES),
        "--device", device,
    ]
    if VALID_TOKENS.exists():
        argv.extend(["--valid_tokens", str(VALID_TOKENS)])
    if use_wandb:
        argv.append("--wandb")

    sys.argv = [sys.argv[0]] + argv
    from student.training.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
