#!/usr/bin/env bash
# Reproduce the final curriculum stage: chunk_len=60, initialized from the
# previous (chunk_len=32) checkpoint. This is the last and longest-horizon stage
# of the rollout curriculum (see README "Training curriculum").
#
# The released model is NOT trained from scratch at chunk_len=60 -- it is the
# tail of a curriculum that climbs chunk_len 2 -> 4 -> 8 -> 12 -> 32 -> 60, each
# stage warm-started from the previous best_model.pt. Run the earlier stages the
# same way (only --chunk-len / --init-from / --out-dir change) to reproduce
# end-to-end.
#
# Usage:
#   NPROC=7 DATA_DIR=data/v4_repr_1M INIT_FROM=runs/prev/best_model.pt \
#     OUT_DIR=runs/main bash scripts/train.sh
set -euo pipefail

NPROC="${NPROC:-1}"                                  # GPUs for DDP (orig: 7xH100)
DATA_DIR="${DATA_DIR:-data/v4_repr_1M}"
INIT_FROM="${INIT_FROM:-runs/prev/best_model.pt}"    # previous (chunk_len=32) best
OUT_DIR="${OUT_DIR:-runs/main}"
MASTER_PORT="${MASTER_PORT:-29519}"

mkdir -p "$OUT_DIR"

# DDP correctness note: TomasMultiHeadMLP must run with broadcast_buffers=False
# (the trainer sets this). Rank-0-only validation + the default broadcast_buffers
# desyncs NCCL and surfaces as bogus CUDA "index out of bounds" asserts.
torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" \
    train.py \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR" \
        --init-from "$INIT_FROM" \
        --batch-size 256 \
        --chunk-len 60 --pushforward-steps 4 --rollout-every 2 \
        --epochs 20 --lr 2e-6 --grad-clip 1.0 \
        --hidden-dim 1024 --n-layers 5 \
        --head-hidden-dim 256 --nuc-head-hidden-dim 512 \
        --gc-weight 20.0 --edge-weight 3.0 --proj-n-iter 3 \
        --log-factor-clamp 2.302585 \
        --relerr-weight 0.4 --relerr-clip-max 5.0 \
        --skip-loss-thresh 2.0 --noise-std 0.0 \
        2>&1 | tee "$OUT_DIR/train.log"

echo "Done. Checkpoint at $OUT_DIR/best_model.pt"
echo "Evaluate with: python evaluate.py --run-dir $OUT_DIR --data-dir $DATA_DIR"
