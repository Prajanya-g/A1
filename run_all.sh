#!/bin/bash
# =============================================================================
# run_all.sh — launch every A1 experiment in parallel across 8 H100s
#
# Assumes 8 GPUs (0-7). Each GPU runs one job at a time; jobs are queued
# per-GPU using &&. Total wall time ≈ max(per-GPU chain) ≈ 2-3 hours.
#
# GPU assignment:
#   GPU 0: lr_3e-4  (BASELINE — checkpoint used for generate)
#   GPU 1: lr_1e-3
#   GPU 2: lr_1e-4
#   GPU 3: lr_3e-3
#   GPU 4: ablation_no_norm_bestlr  → ablation_silu
#   GPU 5: ablation_no_norm_lowlr   → batch_64
#   GPU 6: ablation_postnorm        → batch_128
#   GPU 7: ablation_nope            → batch_256 → batch_512
#
# BEST_LR is hardcoded to 3e-4 (overwhelmingly likely winner for this config).
# If a different LR wins the sweep, rerun only the 5 ablation jobs.
#
# Usage:
#   chmod +x run_all.sh && bash run_all.sh
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN="data/train_tokens.npy"
VALID="data/valid_tokens.npy"
CKPT="checkpoints"
LOGS="logs"
mkdir -p "$LOGS"

# ── Assumed best LR (change if sweep says otherwise, then rerun ablations) ────
BEST_LR=3e-4
LOWER_LR=6e-5   # 1/5 of best, for no_rmsnorm stability test

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
COMMON="
  --train_tokens  $TRAIN
  --valid_tokens  $VALID
  --vocab_size    10000
  --context_length 256
  --d_model       512
  --num_layers    4
  --num_heads     16
  --d_ff          1344
  --batch_size    128
  --max_iters     10000
  --warmup_iters  1000
  --lr_min        1e-5
  --betas         0.9 0.95
  --weight_decay  0.1
  --grad_clip     1.0
  --eval_interval 250
  --eval_batches  50
  --save_every    1000
  --device        cuda
  --wandb
  --wandb_project a1-tinystories
"

# ── Helper ────────────────────────────────────────────────────────────────────
# job <gpu_id> <run_name> [extra args...]
job() {
    local gpu=$1 name=$2; shift 2
    mkdir -p "$CKPT/$name"
    echo "[GPU $gpu] starting: $name"
    CUDA_VISIBLE_DEVICES=$gpu python -m student.training.train \
        --checkpoint_dir "$CKPT/$name" \
        --wandb_run_name "$name" \
        --lr_max $BEST_LR \
        $COMMON "$@" \
        > "$LOGS/${name}.log" 2>&1
    echo "[GPU $gpu] done:     $name"
}

# ── GPU 0: LR sweep baseline  (also the checkpoint for `generate`) ─────────
(
  job 0 lr_3e-4  --lr_max 3e-4
) &

# ── GPU 1: LR sweep — likely diverges (good: shows instability boundary) ─────
(
  job 1 lr_1e-3  --lr_max 1e-3
) &

# ── GPU 2: LR sweep — too slow (good: shows underfitting) ────────────────────
(
  job 2 lr_1e-4  --lr_max 1e-4
) &

# ── GPU 3: LR sweep — almost certainly diverges ──────────────────────────────
(
  job 3 lr_3e-3  --lr_max 3e-3
) &

# ── GPU 4: no_rmsnorm @ best LR  →  then SwiGLU ablation ─────────────────────
(
  job 4 ablation_no_norm_bestlr --no_rmsnorm
  job 4 ablation_silu --d_ff 2048 --use_silu
) &

# ── GPU 5: no_rmsnorm @ lower LR  →  then batch_64 ──────────────────────────
(
  job 5 ablation_no_norm_lowlr  --lr_max $LOWER_LR --no_rmsnorm
  job 5 batch_64   --batch_size 64  --max_iters 20000
) &

# ── GPU 6: post_norm  →  then batch_128 ──────────────────────────────────────
(
  job 6 ablation_postnorm --post_norm
  job 6 batch_128  --batch_size 128 --max_iters 10000
) &

# ── GPU 7: NoPE  →  batch_256  →  batch_512 ──────────────────────────────────
(
  job 7 ablation_nope     --no_rope
  job 7 batch_256  --batch_size 256 --max_iters 5000
  job 7 batch_512  --batch_size 512 --max_iters 2500
) &

# ── Wait for all GPUs to finish ───────────────────────────────────────────────
wait
echo ""
echo "════════════════════════════════════════"
echo "  All experiments complete."
echo "  Logs in: $LOGS/"
echo "  Checkpoints in: $CKPT/"
echo "════════════════════════════════════════"

# ── Generate text from baseline checkpoint ────────────────────────────────────
echo "Generating text from baseline checkpoint..."
python -m student.generate \
    --checkpoint "$CKPT/lr_3e-4/ckpt_iter_10000.pt" \
    --train_corpus "data/TinyStoriesV2-GPT4-train.txt" \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.95 \
    --device cuda \
    --output generate_output.txt

echo "Done! Generated text saved to generate_output.txt"
