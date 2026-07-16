# TOMAS Aerosol Microphysics Emulator — Δt-conditioned ("time-skip") model

> 🚧 **Status: active development.** This model is being actively improved —
> accuracy, APIs, the shipped checkpoints, and reported numbers may change.
> Treat current results as a strong baseline, not a final release.

A neural surrogate for the **TOMAS** sectional aerosol microphysics model. Given
the aerosol size distribution and the atmospheric state, it predicts how the
distribution evolves — nucleation, condensation, and coagulation — over a
6-hour horizon, at a tiny fraction of the cost of integrating the stiff
microphysics ODEs.

**This branch ships the Δt-conditioned emulator** (`checkpoints/dt_champion`):
the same 5.4M-parameter conservation-projected multi-head MLP as the 60-second
baseline on `main`, but **conditioned on the step size Δt**. One trained
network serves every step size from 1 to 30 minutes — you pick Δt at inference
and each forward pass advances the state that far. At Δt = 30 min it reaches
6 h in **12 steps instead of 360**, and is *more* accurate there than the
baseline (per-bin median 9.0 % vs 12.5 % at 6 h): fewer autoregressive steps
means less compounding of per-step error.

How Δt enters: the step size is encoded as **8 log-scale Fourier features**
(4 octaves × sin/cos, `dt_features.py`) appended to the 5 atmospheric forcing
features, giving 94 network inputs (81 log-state + 5 forcing + 8 Δt). The
per-step sulfur-mass conservation projection (`model_conserved_v2.py`,
3 iterations) is applied at every rollout step, in training and at inference.

---

## Performance

Autoregressive rollout to 6 h on 500 held-out scenarios, pooled per-bin
relative error in **physical space** (floor-masked; protocol of
`evaluate_dt.py`). Reproduced from the shipped checkpoint:

| Δt | steps to 6 h | horizon | Nk p50 | Nk p90 | Mk p50 | Mk p90 |
|---:|---:|---:|---:|---:|---:|---:|
| **30 min** | **12** | 1 h | 0.106 | 0.589 | 0.101 | 0.554 |
| | | 3 h | 0.088 | 0.431 | 0.079 | 0.395 |
| | | **6 h** | **0.090** | **0.432** | **0.086** | **0.407** |
| 5 min | 72 | 6 h | 0.131 | 0.614 | 0.144 | 0.646 |
| 1 min | 360 | 6 h | 0.413 | 1.728 | 0.410 | 1.861 |

**Use Δt = 30 min.** It is simultaneously the most accurate *and* the fastest
configuration — long strides both skip error compounding and let the network
average over sub-step stiffness. Any Δt in the training list
(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30 min) works; **do not exceed
30 min** — at Δt = 60 the total particle number collapses (p90 → 1.0), as that
step size is outside the training range.

### Speed

One 6-h trajectory = 12 forward passes of a 5.4M MLP. At large batch on a
single H100 this sustains roughly **160k scenario-trajectories per second**
(~6 µs per scenario per 6 h) — about 5,600× faster than the native TOMAS
integrator. A million grid cells is ~6 GPU-seconds, or under a second across
an 8-GPU node.

---

## Repository layout (this branch)

```
README.md                  This file
requirements.txt           Runtime dependencies (torch, numpy)
dt_features.py             Fourier-Δt encoding (the only new model input)
train_dt_rollout.py        Δt-rollout trainer (chunked curriculum, conservation projection)
evaluate_dt.py             Rollout evaluation at Δt = 1/5/30 -> the table above
metrics.py                 log-state -> physical conversion + train/val split helper
model_multihead.py         TomasMultiHeadMLP: shared backbone + 4 region heads
model_conserved_v2.py      ConservativeMultiHeadMLPV2: per-step sulfur-mass projection wrapper
dataset_v4.py              v4 data loader (per-step conditioning, log-state, normalization)
scripts/eval_dt_champion.sh    One-command evaluation of the shipped checkpoint
scripts/train_dt_champion.sh   Training reproduction (exact shipped hyperparameters)
checkpoints/dt_champion/   The trained Δt-conditioned model
  best_model.pt              weights + the exact training args (epoch 18, val_mse 0.034)
  norm_stats.pt              normalization statistics (required to run the model)
docs/DATA.md               How the CESM-seeded v4 training data was generated
checkpoints/{best_model,norm_stats}.pt   The 60-s baseline from main (unchanged)
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch>=2.2, numpy
```

The checkpoint was produced with **Python 3.10 / torch 2.5.1+cu121** on H100s.
A GPU is recommended but not required (the scripts fall back to CPU).

---

## Quickstart

### 1. Evaluate the shipped checkpoint

```bash
DATA_DIR=/path/to/v4_repr_1M scripts/eval_dt_champion.sh
```

Prints the Δt = 1 / 5 / 30 table above. The val split (`val_frac 0.05, seed 0`)
matches training, so evaluation runs on truly held-out scenarios.

### 2. Run inference in your own code

The model maps a **normalized** `[state(81) ‖ cond(5) ‖ fourier_dt(8)]` vector
to the normalized log-state **Δt minutes later**. State layout is
`[log Nk(40) ‖ log Mk_SO4(40) ‖ log Gc_H2SO4(1)]`; conditioning is
`[pres, nh3_conc, T, RH, H2SO4_vmr]` (log-transformed pres/nh3/H2SO4 — see
`dataset_v4.transform_cond`; slice to the first 5 features, this checkpoint
predates the 6th `fion` column).

```python
import torch
from dataset_v4 import NormStatsV4
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2
from dt_features import fourier_dt_feat, N_COND_TOTAL

device = "cuda"
norm = NormStatsV4.load("checkpoints/dt_champion/norm_stats.pt")
ckpt = torch.load("checkpoints/dt_champion/best_model.pt",
                  weights_only=False, map_location=device)
a = ckpt["args"]

base = TomasMultiHeadMLP(
    hidden_dim=a["hidden_dim"], n_backbone_layers=a["n_layers"],
    head_hidden_dim=a["head_hidden_dim"], nuc_head_hidden_dim=a["nuc_head_hidden_dim"],
    n_cond=N_COND_TOTAL, core_end_excl=a["core_end_excl"],
).to(device)
base.load_state_dict(ckpt["model_state_dict"]); base.eval()

model = ConservativeMultiHeadMLPV2(
    base, norm, log_factor_clamp=a["log_factor_clamp"], n_iter=a["proj_n_iter"],
).to(device).eval()

# 6-h rollout at Δt = 30 min: 12 steps.
# state_norm: (B, 81); cond_norm: (B, T, 5) at 1-min resolution -- both normalized.
DT = 30
feat = fourier_dt_feat(torch.tensor([float(DT)], device=device)).expand(state_norm.shape[0], -1)
cur, t = state_norm, 0
with torch.no_grad():
    while t < 360:
        x = torch.cat([cur, cond_norm[:, t], feat], dim=-1)
        cur = model(x).clamp(-20, 20)      # normalized log-state at t + DT
        t += DT
# physical values: metrics.log_state_to_physical(norm.denormalize_state(cur))
```

See `evaluate_dt.py` for the complete, runnable rollout + de-normalization path.

### 3. Retrain

```bash
DATA_DIR=/path/to/v4_repr_1M NPROC=7 scripts/train_dt_champion.sh
```

Uses the shipped checkpoint's exact hyperparameters (12-step Δt-rollout
curriculum, log-MAE hybrid loss, conservation projection in the training
loop). Note the released checkpoint was additionally warm-started from an
earlier Δt-rollout run of the same architecture, so a from-scratch run
reaches comparable but not bit-identical numbers.

---

## Data

Training data (`v4_repr_1M`: ~1M 6-hour scenarios, each a TOMAS-JAX box-model
simulation seeded from a real CESM2 grid cell) is not included in the repo —
`docs/DATA.md` describes how it was generated.

> **Protocol note.** Relative-error numbers are only comparable under the
> *same* masking protocol — always state which one you used. This README uses
> the floor-masked pooled per-bin protocol defined in `evaluate_dt.py`.
