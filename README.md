# TOMAS Aerosol Microphysics Emulator

> 🚧 **Status: active development.** This model is being actively improved —
> accuracy, APIs, the shipped checkpoints, and reported numbers may change.
> Treat current results as a strong baseline, not a final release.

A neural surrogate for the **TOMAS** sectional aerosol microphysics model. Given
the aerosol size distribution and the atmospheric state, it predicts how the
distribution evolves — nucleation, condensation, and coagulation — over a
6-hour horizon, at a tiny fraction of the cost of integrating the stiff
microphysics ODEs.

Two trained models are shipped:

| Model | Params | How it predicts | Use it when |
|---|---|---|---|
| **`checkpoints/axial_cons2` — long-range / physically-constrained axial (this branch)** | 29.5M | Same non-autoregressive operator, fine-tuned with two-window pushforward and a monotonicity constraint | You **chain past 6 h**, or need sulfur mass never to be destroyed. 4.4× better at 24 h; costs ~10% at 6 h |
| `checkpoints/axial_aug5` — axial trajectory operator | 29.5M | **Non-autoregressive**: one forward pass maps the initial state + 6 h forcing to all 12 × 30-min states | You only need a single 6 h window and want the best accuracy there |
| `checkpoints/` — conservation-projected multi-head MLP | 5.4M | Autoregressive 60-s steps with exact per-step sulfur-mass conservation | You want a small, fast per-step model |

---

## Long-range chaining and mass conservation (`axial_cons2`)

The 6 h operator reaches longer horizons by **chaining**: the 6 h output becomes
the next window's initial condition. Measured against 24 h TOMAS ground truth
(`gen_longhorizon.py`), the base model's *per-window* skill is flat out to 19 h,
but chaining it triples the error — the failure is exposure bias, not skill. It
also destroys sulfur: over a free-running 7-day chain one cell lost **92%** of
its mass, which the box model's physics forbids.

`axial_cons2` fine-tunes the base model with two-window pushforward, interior
window starts, and a hard **monotonicity** constraint (ΣMk can never decrease —
exact on ground truth to the float32 floor, unlike the sulfur *equality* budget,
which is wrong at Δt = 30 and is refuted in the doc below).

| | `axial_aug5` | `axial_cons2` + `mono` |
|---|---|---|
| 6 h per-bin Nk p90 | **0.236** | 0.259 |
| 24 h chained Nk p90 | 0.345 | **0.078** |
| 24 h chained, unseen season | 0.408 | **0.109** |
| 24 h chained total-mass p90 | 0.096 | **0.015** |
| chained monotonicity violations | 59% | **8%** |
| 7-day mass ratio M(7d)/M(0) | 0.075 … 2.64 (destroys mass) | **1.05 … 3.25 (never destroys)** |

```python
from model_axial_traj import AxialTrajOperator
from model_conserved_axial import ConservativeAxialTraj

net = ConservativeAxialTraj(base_model, norm_stats, mode="mono")   # drop-in wrapper
```

The constraint also works **post-hoc on any checkpoint** (on `axial_aug5` it costs
+0.004 Mk p90 and cuts violations from 25% to 8%), so you can enable it without
retraining.

Evaluate a chained rollout with `evaluate_chain.py`, which reports both
self-chained and GT-restarted error so compounding is separated from per-window
skill. **`docs/CONSERVATION_LONGRANGE.md` has the full study**, including what was
tried and refuted, and a known initial-condition bug that matters if you couple
this to CESM.

---

## The axial trajectory operator (`axial_aug5`) — current best

`AxialTrajOperator` (`model_axial_traj.py`) treats the state as a **(12 time
steps) × (41 tokens)** grid — 40 size-bin tokens (each carrying its number and
mass) plus a gas token — and alternates **bin-axis attention** (all-pairs bin
coupling, matching coagulation's N·N structure) with **bidirectional time-axis
attention** (non-AR temporal composition with the full forcing in view). Because
nothing is fed back autoregressively, rollout error does not compound: error is
roughly **flat in horizon**, which is the single biggest reason it beats the
AR baseline at 6 h.

**`docs/ARCHITECTURE.md` walks through the whole model piece by piece** —
tokenization, the axial block, the mixture-of-experts FFN, the
persistence-residual head, and a tensor-shape cheat sheet.

The released checkpoint is additionally **OOD-augmentation fine-tuned**: the
base model is trained on CESM-seeded scenarios, then fine-tuned with idealized
single-mode lognormal initial conditions sweeping the nucleation-burst-triggering
corner of forcing space (high H₂SO₄, low condensation sink, cold T, ion-pair
rate `fion` ∈ [0,5] including 0), mixed into the train pool only while
validation stays pure CESM. This removes the OOD cliff on clean/idealized
initial size distributions at ~2 pt in-distribution cost.

### Performance

Held-out CESM scenarios, pooled per-bin relative error in **physical space**
(floor-masked — see *Why the floor matters* below). The base axial model's full
profile at 6 h:

| quantity | p50 | p90 | p99 |
|---|---|---|---|
| per-bin Nk | 0.031 | 0.216 | 1.21 |
| per-bin Mk | 0.028 | 0.189 | 1.26 |
| total number | 0.029 | 0.110 | — |
| total sulfate mass | 0.006 | 0.028 | — |
| gas-phase H₂SO₄ | 0.006 | 0.058 | — |

The shipped `axial_aug5` fine-tune holds this at Nk p50/p90 **0.035 / 0.218**
while transforming out-of-distribution behaviour — on idealized lognormal-IC
scenarios the pre-augmentation model was off by up to ~50× on burst regimes;
`axial_aug5` brings most idealized regimes to in-distribution quality (p90
0.10–0.48). The remaining hard tail is pure-binary nucleation at high N and
high H₂SO₄. Inference: ~467 µs/scenario on an H100 (~76× faster than the
native box model).

### Get the checkpoint

The weights are ~113 MB, over GitHub's 100 MB in-repo limit, so they ship as a
**release asset**:

```bash
bash scripts/download_axial_checkpoint.sh
# -> checkpoints/axial_aug5/best_model.pt  (norm_stats.pt is already in the repo)
```

### Evaluate

```bash
RUN_DIR=checkpoints/axial_aug5 DATA_DIR=/path/to/v4_repr_1M python evaluate_axial.py
```

### Inference in your own code

State layout is `[log Nk(40) ‖ log Mk_SO4(40) ‖ log Gc_H2SO4(1)]`, normalized
with the shipped stats. Conditioning is `[pres, nh3_conc, T, RH, H2SO4_vmr,
fion]` at the 13 grid times (minutes 0, 30, …, 360).

```python
import torch
from dataset_v4 import NormStatsV4
from model_axial_traj import AxialTrajOperator

device = "cuda"
norm = NormStatsV4.load("checkpoints/axial_aug5/norm_stats.pt")
ck = torch.load("checkpoints/axial_aug5/best_model.pt", weights_only=False, map_location=device)
ma = ck["args"]
n_cond = ck["model_state_dict"]["cond_in.weight"].shape[1]      # 6 (with fion)

model = AxialTrajOperator(
    d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"],
    dim_ff=ma["dim_ff"], dropout=0.0, ff_experts=ma.get("ff_experts", 1),
    n_out=12, n_cond=n_cond,
).to(device)
model.load_state_dict(ck["model_state_dict"]); model.eval()

# state0_norm: (B, 81)    normalized log-state at t=0
# cond_norm:   (B, 13, 6) normalized conditioning at minutes [0, 30, ..., 360]
with torch.no_grad():
    traj = model(state0_norm, cond_norm)   # (B, 12, 81): minutes 30, 60, ..., 360
# physical: metrics.log_state_to_physical(norm.denormalize_state(traj))
```

See `evaluate_axial.py` for the complete, runnable path including
normalization.

### Retrain / fine-tune

```bash
# OOD-augmentation fine-tune (the released model's final stage):
NPROC=7 DATA_DIR=/path/to/v4_repr_1M AUG_DIR=/path/to/v4_aug \
    WARM_START=runs/axial_base/best_model.pt \
    BASE_NORM=runs/axial_base/norm_stats.pt \
    OUT_DIR=runs/axial_aug5 bash scripts/train_axial_aug5.sh
```

Two hard-won details are baked into that script — read its header before
changing it: **normalization statistics must be frozen from the base model**
(recomputing them on a mixed CESM+augmentation pool shifts the state mean past
1σ and warm-started weights never recover), and the 6th forcing channel
(`fion`) is zero-padded into a 5-channel warm-start so training begins
numerically identical to the base model.

---

## The MLP baseline (`checkpoints/`)

A **5.4M-parameter conservation-projected multi-head MLP**. It advances the
state one 60-second step at a time and is rolled out autoregressively. A hard
per-step sulfur-mass projection sits on top of the network so total sulfur is
conserved exactly (to float32) regardless of what the network predicts.

### Performance

Autoregressive rollout on 500 held-out scenarios, median per-bin relative error
in **physical space**. These are the numbers `evaluate.py` reproduces from the
shipped checkpoint:

| Horizon | Number (Nk) median rel-err | Mass (Mk) median rel-err |
|--------:|:--------------------------:|:------------------------:|
| 1 h     | 8.4 %                      | 5.9 %                    |
| 3 h     | 10.5 %                     | 7.8 %                    |
| **6 h** | **12.5 %**                 | **10.7 %**               |

Scenario-level (median across the 32 scored bins, per scenario) at 6 h:
**p50 = 9.4 %, p90 = 44.9 %.**

> **Protocol note.** The masking metric is the one defined in `evaluate.py`.
> A stricter alternative protocol reports ~11.2 % / 9.3 %. Numbers are only
> comparable under the *same* protocol — always state which one you used.

### Evaluate

```bash
python evaluate.py \
    --run-dir  checkpoints \
    --data-dir /path/to/v4_repr_1M \
    --n-scenarios 500
```

Prints the 1 h / 3 h / 6 h table above. `--val-frac 0.05 --seed 0` must match
training (they are the defaults) so evaluation runs on truly held-out scenarios.

### Inference in your own code

The model maps a **normalized** `[state(81) ‖ cond(5)]` vector to the next
normalized log-state.

```python
import torch
from dataset_v4 import NormStatsV4
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2

device = "cuda"
norm = NormStatsV4.load("checkpoints/norm_stats.pt")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=True, map_location=device)
a = ckpt["args"]

base = TomasMultiHeadMLP(
    hidden_dim=a["hidden_dim"], n_backbone_layers=a["n_layers"],
    head_hidden_dim=a["head_hidden_dim"], nuc_head_hidden_dim=a["nuc_head_hidden_dim"],
    n_cond=5, core_end_excl=a.get("core_end_excl", 26),
).to(device)
base.load_state_dict(ckpt["model_state_dict"]); base.eval()

model = ConservativeMultiHeadMLPV2(
    base, norm, log_factor_clamp=a["log_factor_clamp"], n_iter=a["proj_n_iter"],
).to(device).eval()

# Autoregressive rollout (state_norm: (B,81), cond_norm: (B, T, 5)) -- both normalized.
cur = state_norm
with torch.no_grad():
    for t in range(T):
        x = torch.cat([cur, cond_norm[:, t]], dim=-1)
        cur = model(x).clamp(-20, 20)          # next normalized log-state
# physical values: metrics.log_state_to_physical(norm.denormalize_state(cur))
```

This model predates the `fion` conditioning channel; its checkpoint uses the
first 5 cond channels (`evaluate.py` slices automatically).

### Architecture

```
input  [ state(81) ‖ cond(5) ]  = 86 normalized features
          │
   shared backbone : 5 × Linear(1024) + GELU
          ├── nucleation head (bins 0–5)   → 12 outputs   2-layer MLP, hidden 512
          ├── core head       (bins 6–25)  → 40 outputs   2-layer MLP, hidden 256
          ├── coarse head     (bins 26–39) → 28 outputs   2-layer MLP, hidden 256
          └── Gc head          (gas H2SO4) →  1 output
          │  (per-head residual over the matching state indices)
output [ next log-state(81) ]
          │
   ConservativeMultiHeadMLPV2 : per-step projection so
   Σ Mk[t+1] + Gc[t+1] = Σ Mk[t] + H2SO4_forced[t]   (sulfur mass conserved)
```

Why region heads: nucleation, accumulation, and coarse bins have very different
dynamics (the nucleation bins fire violently at `t=0`; coarse bins barely move).
Splitting them lets each head specialize. The boundary between the core and
coarse heads is configurable (`core_end_excl`).

### Conservation projection

The single quantity enforced each step is total box sulfur: the SO4 in the 40
aerosol mass bins plus gas-phase H2SO4 must equal the previous total plus the
H2SO4 mass injected from CESM that step (a closed-form function of the cond
vector). The network is free to *partition* that budget across bins and the gas
phase however it likes. The forcing term is per-step, **not** a running sum —
TOMAS hard-resets gas-phase H2SO4 to the CESM value at the start of every step,
so a summed-forcing constraint would be wrong. See `docs/DATA.md`.

### Training curriculum

The released model is the last of a sequence of stages that grow the rollout
horizon while warm-starting each stage from the previous best. Training a long
horizon from scratch is unstable (exposure bias / gradient blow-ups); ramping it
is not.

| Stage | chunk_len | init from |
|------:|:---------:|-----------|
| 1     | 2         | scratch   |
| 2     | 4         | stage 1   |
| 3     | 8         | stage 2   |
| 4     | 12        | stage 3   |
| 5     | 32        | stage 4   |
| **6** | **60**    | **stage 5** (released model) |

Each stage uses a **hybrid loss**: log-MSE (with an inverse-variance weighting
floored to avoid amplifying empty-bin noise) plus a clipped relative-error term
(`--relerr-weight 0.4`) to keep the metric the model is judged on in the
objective. Pushforward steps (`--pushforward-steps 4`) train multi-step
composition directly. Reproduce the final stage with:

```bash
NPROC=7 DATA_DIR=/path/to/v4_repr_1M \
    INIT_FROM=runs/prev/best_model.pt OUT_DIR=runs/main \
    bash scripts/train.sh
```

> **Multi-GPU caveat.** `TomasMultiHeadMLP` under DDP **must** set
> `broadcast_buffers=False` (the trainer does). With rank-0-only validation and
> the default `broadcast_buffers=True`, NCCL desyncs and surfaces as spurious
> CUDA "index out of bounds" asserts.

---

## Repository layout

```
README.md                  Project overview, performance, and instructions (this file)
requirements.txt           Runtime dependencies (torch, numpy)

model_axial_traj.py        AxialTrajOperator: non-AR bin×time axial-attention operator
train_axial_aug.py         Axial trainer (base training + OOD-augmentation fine-tune)
evaluate_axial.py          Held-out evaluation of an axial checkpoint
burst_feats.py             Analytic Dunne nucleation-rate feature (optional cond channel)
scripts/train_axial_aug5.sh          Reproduce the released augmentation fine-tune
scripts/download_axial_checkpoint.sh Fetch the ~113 MB axial weights (release asset)
checkpoints/axial_aug5/    The axial model
  best_model.pt              weights (NOT in git -- download via the script above)
  norm_stats.pt              normalization statistics (required; frozen from base model)

train.py                   MLP trainer (rollout curriculum, conservation projection)
evaluate.py                MLP rollout evaluation -> per-channel median relative error
model_multihead.py         TomasMultiHeadMLP: shared backbone + 4 region heads
model_conserved_v2.py      ConservativeMultiHeadMLPV2: per-step sulfur-mass projection
scripts/train.sh           One-command reproduction of the final MLP training stage
checkpoints/               The MLP model
  best_model.pt              weights + the exact training args
  norm_stats.pt              normalization statistics (required to run the model)

metrics.py                 log-state -> physical conversion + train/val split helper
dataset_v4.py              v4 data loader (log-state, normalization, 6 cond channels)
docs/ARCHITECTURE.md       Piece-by-piece walkthrough of the axial operator
docs/DATA.md               How the CESM-seeded v4 training data was generated
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch>=2.2, numpy
```

The checkpoints were produced with **Python 3.10 / torch 2.5.1+cu121** on
H100s. A GPU is recommended but not required.

---

## Data

Training data is **not** included (it is ~1 M scenarios). Each scenario is a
6-hour TOMAS-JAX box-model simulation seeded from a real CESM2 grid cell, with
the atmospheric environment (T, RH, OH, SO2 → H2SO4) **evolving hour-by-hour**
from CESM and interpolated to the 60-second step. Layout expected by the loader:

```
v4_repr_1M/
  t<cellid>/batch_s<scenario>_b<batch>.npz   # Nk, Mk, Gc, evolving cond
```

`docs/DATA.md` documents the generation pipeline and schema in full. The loader
also reads an optional per-scenario `fion` (ion-pair formation rate) key,
defaulting to 3.0 for older data that lacks it.

### Why the floor matters

Empty size bins sit at a numerical floor (`LOG_EPS = 1e-30`; coarse bins often
park at ~`1e-5` for >50 % of a rollout). A relative-error metric over those bins
divides by ~floor and explodes meaninglessly. Both evaluators therefore mask any
bin that is at the floor for more than half the horizon, so the reported median
reflects bins that actually carry signal. This is a metric-side convention, not
a clamp on the data.

---

## Status and roadmap

This emulator is under active development. Current focus areas:

- **The in-distribution per-bin p90 < 0.10 target** — limited by the
  sensitivity/ill-conditioning of the nucleation-burst flow map, not by model
  capacity or data volume; a physics-in-loop hybrid (exact burst source term)
  is the identified route.
- **The pure-binary high-N / high-H₂SO₄ OOD tail** — broader augmentation
  coverage.
- **Throughput** — distilling the axial model to a smaller axial student.

By design the MLP baseline conserves **sulfur mass** exactly; the axial model
learns conservation to ~0.6 % (total sulfate p50) without a hard constraint.
Neither conserves number — nucleation and coagulation legitimately change
particle number, which the networks predict.

---

## License

MIT — see `LICENSE`.
