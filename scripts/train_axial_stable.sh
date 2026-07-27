#!/usr/bin/env bash
# Long-range stability + physical-consistency fine-tune of the axial_aug5 model.
#
# Adds three things to the aug5 recipe: two-window PUSHFORWARD on 24 h
# trajectories (window 2 starts from the model's own detached output), INTERIOR
# window starts, and the MONOTONICITY + bin-geometry projection active in the
# forward pass. See docs/CONSERVATION_LONGRANGE.md.
#
# Requires the 24 h dataset (gen_longhorizon.py) and the aug5 checkpoint
# (scripts/download_axial_checkpoint.sh with MODEL=axial_aug5).
#
# --replay-weight trades 6 h accuracy against long-range accuracy, monotonically:
#   1.0 -> 6 h p90 +39%, 19 h chained 0.064
#   3.0 -> 6 h p90 +10%, 19 h chained 0.078   <- shipped checkpoint
set -euo pipefail
OUT="${OUT:-runs/axial_cons2}"

mkdir -p "$OUT"
# norm_stats MUST be inherited, never recomputed on the mixed pool
# (recomputing is the known catastrophic-forgetting trap -- see aug3/aug5).
cp -n checkpoints/axial_aug5/norm_stats.pt "$OUT/norm_stats.pt"

python train_axial_stable.py \
    --out-dir "$OUT" \
    --warm-start checkpoints/axial_aug5/best_model.pt \
    --long-dir data/v4_long24h --replay-dir data/v4_repr_1M \
    --epochs 16 --long-pairs 40000 --replay-max 30000 \
    --batch-size 48 --lr 3e-5 --replay-weight 3.0 \
    --pushforward-p 0.75 --proj-mode mono --budget-weight 0.05 \
    --resume auto
