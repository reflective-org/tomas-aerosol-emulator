#!/bin/bash
# Evaluate the shipped Δt-conditioned checkpoint on held-out scenarios.
# Reproduces the numbers in the README (500 scenarios, Δt = 1 / 5 / 30 min).
set -eu
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-data/v4_repr_1M}" \
RUN_DIR="${RUN_DIR:-checkpoints/dt_champion}" \
ROLL_DTS="${ROLL_DTS:-1,5,30}" \
python3 evaluate_dt.py
