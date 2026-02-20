#!/bin/bash
#SBATCH --job-name=a1_all_experiments
#SBATCH --output=logs/a1_all_%j.out
#SBATCH --error=logs/a1_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pg2963@nyu.edu

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

if command -v uv &>/dev/null; then
  RUN_CMD="uv run python"
else
  RUN_CMD="python"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN="data/train_tokens.npy"
VALID="data/valid_tokens.npy"
CKPT="/scratch/pg2963/A1/checkpoints"
LOGS="logs"

BEST_LR=1e-3
LOWER_LR=2e-4   # 1/5 of best, for no_rmsnorm stability test

# ── Shared args ───────────────────────────────────────────────────────────────
COMMON="
  --train_tokens   $TRAIN
  --valid_tokens   $VALID
  --vocab_size     10000
  --context_length 256
  --d_model        512
  --num_layers     4
  --num_heads      16
  --d_ff           1344
  --batch_size     128
  --max_iters      10000
  --warmup_iters   500
  --lr_min         1e-5
  --betas          0.9 0.95
  --weight_decay   0.1
  --grad_clip      1.0
  --eval_interval  500
  --eval_batches   50
  --save_every     1000
"

# ── Helper ────────────────────────────────────────────────────────────────────
job() {
    local gpu=$1 name=$2; shift 2
    mkdir -p "$CKPT/$name"
    echo "[GPU $gpu] starting: $name"
    CUDA_VISIBLE_DEVICES=$gpu $RUN_CMD -m student.training.train \
        --checkpoint_dir "$CKPT/$name" \
        --wandb_run_name "$name" \
        --device cuda \
        $COMMON "$@" \
        > "$LOGS/${name}.log" 2>&1
    echo "[GPU $gpu] done: $name"
}

# ── GPU 0: LR sweep — 3e-4 (likely best) ─────────────────────────────────────
(
  job 0 lr_3e-4 --lr_max 3e-4
) &

# ── GPU 1: LR sweep — 1e-3 (baseline, known good) ────────────────────────────
(
  job 1 lr_1e-3 --lr_max 1e-3
) &

# ── GPU 2: LR sweep — 1e-4 (too slow) ────────────────────────────────────────
(
  job 2 lr_1e-4 --lr_max 1e-4
) &

# ── GPU 3: LR sweep — 3e-3 (likely diverges) ─────────────────────────────────
(
  job 3 lr_3e-3 --lr_max 3e-3
) &

# ── GPU 4: no_rmsnorm @ best LR → swiglu ablation ────────────────────────────
(
  job 4 ablation_no_norm_bestlr --lr_max $BEST_LR --no_rmsnorm
  job 4 ablation_silu           --lr_max $BEST_LR --d_ff 2048 --use_silu
) &

# ── GPU 5: no_rmsnorm @ lower LR → batch_64 ──────────────────────────────────
(
  job 5 ablation_no_norm_lowlr --lr_max $LOWER_LR --no_rmsnorm
  job 5 batch_64               --lr_max $BEST_LR  --batch_size 64  --max_iters 20000
) &

# ── GPU 6: post_norm → batch_128 ─────────────────────────────────────────────
(
  job 6 ablation_postnorm --lr_max $BEST_LR --post_norm
  job 6 batch_128         --lr_max $BEST_LR --batch_size 128 --max_iters 10000
) &

# ── GPU 7: NoPE → batch_256 → batch_512 ──────────────────────────────────────
(
  job 7 ablation_nope --lr_max $BEST_LR --no_rope
  job 7 batch_256     --lr_max $BEST_LR --batch_size 256 --max_iters 5000
  job 7 batch_512     --lr_max $BEST_LR --batch_size 512 --max_iters 2500
) &

# ── Wait for all GPUs ─────────────────────────────────────────────────────────
wait
echo "All experiments done."

# ── Generate text from best checkpoint ───────────────────────────────────────
echo "Generating text..."
CUDA_VISIBLE_DEVICES=0 $RUN_CMD -m student.generate \
    --checkpoint "$CKPT/lr_1e-3/ckpt_iter_10000.pt" \
    --train_corpus "data/TinyStoriesV2-GPT4-train.txt" \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_p 0.95 \
    --device cuda \
    --output generate_output.txt

echo "Done. Generated text in generate_output.txt"
