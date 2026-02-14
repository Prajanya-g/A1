#!/usr/bin/env python3
"""Submit TinyStories training job to HPC via submitit (SLURM).

Usage (from repo root on HPC):
  uv run python scripts/submit_tinystories_hpc.py [--partition PARTITION] [--wandb]

Before first run:
  1. Prepare tokenized data on a login node:
       uv run python -m student.tinystories_tokenize_analysis
  2. Edit SLURM defaults below if your cluster uses different partition names or limits.
     Check HPC_doc.pdf for partition name (e.g. gpu, gpu_short, greene), time limits, and GRES.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_training(wandb: bool = False) -> None:
    """Entry point for the submitted job: run training in repo root."""
    os.chdir(REPO_ROOT)
    cmd = [
        sys.executable, "-m", "student.training.train",
        "--train_tokens", str(REPO_ROOT / "data" / "train_tokens.npy"),
        "--valid_tokens", str(REPO_ROOT / "data" / "valid_tokens.npy"),
        "--vocab_size", "10000",
        "--context_length", "256",
        "--d_model", "512",
        "--d_ff", "1344",
        "--rope_theta", "10000",
        "--num_layers", "4",
        "--num_heads", "16",
        "--batch_size", "128",
        "--max_iters", "10000",
        "--warmup_iters", "500",
        "--checkpoint_dir", str(REPO_ROOT / "checkpoints" / "tinystories"),
        "--save_every", "1000",
        "--log_interval", "50",
        "--eval_interval", "500",
        "--eval_batches", "50",
        "--device", "cuda:0",
    ]
    if wandb:
        cmd.append("--wandb")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit TinyStories training to SLURM via submitit")
    parser.add_argument("--partition", type=str, default="gpu",
                        help="SLURM partition (check HPC_doc.pdf; e.g. gpu, greene)")
    parser.add_argument("--time-min", type=int, default=1440,
                        help="Job time limit in minutes (default 24h)")
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print job config and exit")
    args = parser.parse_args()

    train_tokens = REPO_ROOT / "data" / "train_tokens.npy"
    if not train_tokens.exists():
        print(
            f"Tokenized data not found at {train_tokens}. On the login node run:\n"
            "  uv run python -m student.tinystories_tokenize_analysis",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import submitit
    except ImportError:
        print("submitit not installed. Install with: uv sync", file=sys.stderr)
        sys.exit(1)

    log_dir = REPO_ROOT / "logs" / "submitit"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        timeout_min=args.time_min,
        slurm_partition=args.partition,
        mem_gb=args.mem_gb,
        cpus_per_task=args.cpus_per_task,
        name="tinystories",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    if args.dry_run:
        print("Would submit job with:", executor.parameters)
        print("To submit for real, run without --dry-run")
        return

    job = executor.submit(run_training, wandb=args.wandb)
    print(f"Submitted job {job.job_id}. Logs: {log_dir}")
    print("Check status: squeue -u $USER  or  submitit job state")


if __name__ == "__main__":
    main()
