#!/bin/bash
# Train the Δt-conditioned emulator with the shipped checkpoint's exact
# hyperparameters (7 GPUs; adjust --nproc-per-node to your machine).
#
# Note: the released checkpoint was additionally warm-started (--init-from)
# from an earlier Δt-rollout run of the same architecture; training from
# scratch with this recipe reaches comparable but not bit-identical results.
set -eu
cd "$(dirname "$0")/.."
OUT="${OUT:-runs/dt_champion_repro}"
DATA_DIR="${DATA_DIR:-data/v4_repr_1M}"
mkdir -p "$OUT"
torchrun --nproc-per-node="${NPROC:-7}" --master-port=29729 \
    train_dt_rollout.py --data-dir "$DATA_DIR" --out-dir "$OUT" \
    --chunk-len 12 --epochs 18 --dt-list "1,2,3,4,5,6,8,10,12,15,20,24,30" \
    --proj-n-iter 3 --log-factor-clamp 2.302585 \
    --loss-type logmae --huber-beta 0.2 --aux-mse-weight 0.1 \
    --gc-weight 4 --edge-weight 1.0 --nuc-weight 2.5 --coarse-weight 1.0 --mk-weight 1.2 \
    --lr 5e-5 --lr-min-frac 0.05 --samples-per-scenario 6 \
    --batch-size 256 --workers 4 --val-max-batches 60 \
    --noise-std 0.0 --grad-clip 1.0 --skip-loss-thresh inf \
    --resume auto
