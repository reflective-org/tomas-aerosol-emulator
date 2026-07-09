#!/usr/bin/env bash
# Reproduce the final OOD-augmentation fine-tune of the axial trajectory
# operator (the released checkpoints/axial_aug5 model).
#
# The released model is the tail of a lineage, each stage warm-started from the
# previous best_model.pt:
#   1. axial base model trained on CESM v4 data (train_axial_aug.py without
#      --extra-dirs; d_model 512, 8 blocks, MoE-2 FFN, ~29.5M params)
#   2. + OOD augmentation fine-tune (this script): idealized single-mode
#      lognormal ICs sweeping the nucleation-burst-triggering corner of forcing
#      space (high H2SO4, low condensation sink, cold T, fion in [0,5] incl. 0),
#      mixed into the TRAIN pool only -- validation stays pure CESM.
#
# TWO THINGS THE RESULT DEPENDS ON:
#   * Normalization statistics MUST be frozen from the base model. Recomputing
#     them on the mixed pool shifts state_mean past 1 sigma (the augmentation
#     floor values dominate) and the warm-started weights never recover their
#     in-distribution skill. That is why this script pre-seeds OUT_DIR with the
#     base model's norm_stats.pt BEFORE launching -- the trainer reuses a
#     norm_stats.pt it finds in OUT_DIR instead of recomputing.
#   * The 6th forcing channel (fion, ion-pair formation rate) was added at this
#     stage. The trainer zero-pads cond_in when warm-starting from a 5-channel
#     checkpoint, so training starts numerically identical to the base model.
#
# Usage:
#   NPROC=7 DATA_DIR=data/v4_repr_1M AUG_DIR=data/v4_aug \
#     WARM_START=runs/axial_base/best_model.pt \
#     BASE_NORM=runs/axial_base/norm_stats.pt \
#     OUT_DIR=runs/axial_aug5 bash scripts/train_axial_aug5.sh
set -euo pipefail

NPROC="${NPROC:-1}"                                # GPUs for DDP (orig: 7xH100)
DATA_DIR="${DATA_DIR:-data/v4_repr_1M}"            # CESM base data
AUG_DIR="${AUG_DIR:-data/v4_aug}"                  # idealized-IC augmentation data
WARM_START="${WARM_START:-runs/axial_base/best_model.pt}"
BASE_NORM="${BASE_NORM:-runs/axial_base/norm_stats.pt}"
OUT_DIR="${OUT_DIR:-runs/axial_aug5}"
MASTER_PORT="${MASTER_PORT:-29521}"

mkdir -p "$OUT_DIR"

# Freeze the base model's normalization (see header). Do NOT skip this.
cp -n "$BASE_NORM" "$OUT_DIR/norm_stats.pt"

torchrun --nproc-per-node="$NPROC" --master-port="$MASTER_PORT" \
    train_axial_aug.py \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR" \
        --warm-start "$WARM_START" \
        --extra-dirs "$AUG_DIR" --extra-cap 5000 \
        --per-rank-train 14000 --per-rank-val 1500 \
        --epochs 40 --lr 1e-4 --lr-min-frac 0.1 \
        --batch-size 128 \
        --d-model 512 --nhead 8 --num-blocks 8 --dim-ff 768 --ff-experts 2 \
        --gc-weight 4.0 --nuc-weight 2.5 --coarse-weight 1.0 --mk-weight 1.2 \
        --grad-clip 1.0 \
        2>&1 | tee "$OUT_DIR/train.log"

echo "Done. Checkpoint at $OUT_DIR/best_model.pt"
echo "Evaluate with: RUN_DIR=$OUT_DIR DATA_DIR=$DATA_DIR python evaluate_axial.py"
