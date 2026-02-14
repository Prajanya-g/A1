#!/bin/bash
#SBATCH --job-name=tinystories
#SBATCH --output=logs/tinystories_%j.out
#SBATCH --error=logs/tinystories_%j.err
#SBATCH --partition=gpu          # EDIT: set to your cluster's GPU partition (e.g. gpu, gpu_short, greene)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00          # EDIT: adjust to your cluster's max or expected runtime
#SBATCH --mail-type=END,FAIL     # optional: email when job ends or fails
#SBATCH --mail-user=YOUR_NETID@nyu.edu   # EDIT: your email

# Run TinyStories experiment on HPC. Prepare data on login node first:
#   uv run python -m student.tinystories_tokenize_analysis
# Then submit: sbatch scripts/run_tinystories_slurm.sh

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs checkpoints/tinystories

# Use uv if available (recommended); otherwise activate your venv/conda and use python.
if command -v uv &>/dev/null; then
  RUN_CMD="uv run python"
else
  RUN_CMD="python"
  # If you use conda: source activate your_env
  # If you use modules: module load python/3.11 cuda/11.8
fi

$RUN_CMD -m student.training.train \
  --train_tokens data/train_tokens.npy \
  --valid_tokens data/valid_tokens.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --num_layers 4 \
  --num_heads 16 \
  --batch_size 128 \
  --max_iters 10000 \
  --warmup_iters 500 \
  --checkpoint_dir checkpoints/tinystories \
  --save_every 1000 \
  --log_interval 50 \
  --eval_interval 500 \
  --eval_batches 50 \
  --device cuda:0

echo "Done."
