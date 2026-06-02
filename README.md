# TOMAS Aerosol Microphysics Emulator (`p7`)

A neural surrogate for the **TOMAS** sectional aerosol microphysics model. Given
the aerosol size distribution and the atmospheric state, it predicts how the
distribution evolves — nucleation, condensation, and coagulation — over a
6-hour rollout, at a tiny fraction of the cost of integrating the stiff
microphysics ODEs.

`p7` is a **5.4M-parameter conservation-projected multi-head MLP**. It advances
the state one 60-second step at a time and is rolled out autoregressively. A
hard per-step sulfur-mass projection sits on top of the network so total sulfur
is conserved exactly (to float32) regardless of what the network predicts.

This repository packages the trained `p7` champion together with everything
needed to run, evaluate, and retrain it.

---

## Performance

Autoregressive rollout on 500 held-out scenarios, median per-bin relative error
in **physical space** (bins that sit at the empty-bin floor for most of the
horizon are masked — see *Why the floor matters* below). These are the numbers
`evaluate.py` reproduces from the shipped checkpoint:

| Horizon | Number (Nk) median rel-err | Mass (Mk) median rel-err |
|--------:|:--------------------------:|:------------------------:|
| 1 h     | 8.4 %                      | 5.9 %                    |
| 3 h     | 10.5 %                     | 7.8 %                    |
| **6 h** | **12.5 %**                 | **10.7 %**               |

Scenario-level (median across the 32 scored bins, per scenario) at 6 h:
**p50 = 9.4 %, p90 = 44.9 %.**

Per-bin breakdown at 6 h (selected):

| Bin region            | Behaviour                  | Median rel-err |
|-----------------------|----------------------------|:--------------:|
| Nucleation (0–5)      | violent particle formation | ~11–25 %       |
| Accumulation (7–24)   | smooth growth              | ~5–13 %        |
| Bin 25 (core/coarse boundary) | head-boundary artifact | ~60 %      |
| Coarse (28–39)        | nearly static              | ~2–4 %         |

> **Protocol note.** The masking metric is the one defined in `evaluate.py`. A
> stricter alternative protocol (used elsewhere in the research line) reports
> p7 at 11.2 % / 9.3 %. Numbers are only comparable under the *same* protocol —
> always state which one you used.

---

## Repository layout

```
README.md                Project overview, performance, and instructions (this file)
requirements.txt         Runtime dependencies (torch, numpy)
train.py                 Trainer (rollout curriculum, conservation projection, hybrid loss)
evaluate.py              Rollout evaluation -> per-channel median relative error
metrics.py               log-state -> physical conversion + the train/val split helper
model_multihead.py       TomasMultiHeadMLP: shared backbone + 4 region heads
model_conserved_v2.py    ConservativeMultiHeadMLPV2: per-step sulfur-mass projection wrapper
dataset_v4.py            v4 data loader (per-step conditioning, log-state, normalization)
scripts/train_p7.sh      One-command reproduction of the final (chunk_len=60) phase
checkpoints/p7/          The trained champion
  best_model.pt            weights + the exact training args
  norm_stats.pt            normalization statistics (required to run the model)
docs/DATA.md             How the CESM-seeded v4 training data was generated
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch>=2.2, numpy
```

The checkpoint was produced with **Python 3.10 / torch 2.5.1+cu121** on H100s.
A GPU is recommended for rollout but not required (`--device cpu`).

---

## Quickstart

### 1. Evaluate the shipped checkpoint

```bash
python evaluate.py \
    --run-dir  checkpoints/p7 \
    --data-dir /path/to/v4_repr_1M \
    --n-scenarios 500
```

Prints the 1 h / 3 h / 6 h table above. `--val-frac 0.05 --seed 0` must match
training (they are the defaults) so evaluation runs on truly held-out scenarios.

### 2. Run inference in your own code

The model maps a **normalized** `[state(81) ‖ cond(5)]` vector to the next
normalized log-state. State layout is `[log Nk(40) ‖ log Mk_SO4(40) ‖ log Gc_H2SO4(1)]`;
conditioning is `[pres, nh3_conc, T, RH, H2SO4_vmr]`.

```python
import torch
from dataset_v4 import NormStatsV4
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2

device = "cuda"
norm = NormStatsV4.load("checkpoints/p7/norm_stats.pt")
ckpt = torch.load("checkpoints/p7/best_model.pt", weights_only=True, map_location=device)
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

See `evaluate.py` for the complete, runnable rollout + de-normalization path.

### 3. Retrain

```bash
# Final phase only (needs a p6 / chunk_len=32 checkpoint to warm-start from):
NPROC=7 DATA_DIR=/path/to/v4_repr_1M \
    INIT_FROM=runs/p6/best_model.pt OUT_DIR=runs/p7 \
    bash scripts/train_p7.sh
```

`p7` is the tail of a curriculum, not a from-scratch run — see below.

---

## Model

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

`p7` is the last of a sequence of phases that grow the rollout horizon while
warm-starting each phase from the previous best. Training a long horizon from
scratch is unstable (exposure bias / gradient blow-ups); ramping it is not.

| Phase | chunk_len | init from |
|------:|:---------:|-----------|
| p1    | 2         | scratch   |
| p2    | 4         | p1        |
| p3    | 8         | p2        |
| p4    | 12        | p3        |
| p6    | 32        | p4        |
| **p7**| **60**    | **p6**    |

Each phase uses a **hybrid loss**: log-MSE (with an inverse-variance weighting
floored to avoid amplifying empty-bin noise) plus a clipped relative-error term
(`--relerr-weight 0.4`) to keep the metric the model is judged on in the
objective. Pushforward steps (`--pushforward-steps 4`) train multi-step
composition directly. The exact `p7` hyperparameters are baked into
`scripts/train_p7.sh` and into `best_model.pt`'s saved `args`.

> **Multi-GPU caveat.** `TomasMultiHeadMLP` under DDP **must** set
> `broadcast_buffers=False` (the trainer does). With rank-0-only validation and
> the default `broadcast_buffers=True`, NCCL desyncs and surfaces as spurious
> CUDA "index out of bounds" asserts.

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

`docs/DATA.md` documents the generation pipeline and schema in full.

### Why the floor matters

Empty size bins sit at a numerical floor (`LOG_EPS = 1e-30`; coarse bins often
park at ~`1e-5` for >50 % of a rollout). A relative-error metric over those bins
divides by ~floor and explodes meaninglessly. `evaluate.py` therefore masks any
bin that is at the floor for more than `--empty-frac` (default 50 %) of the
horizon, so the reported median reflects bins that actually carry signal. This
is a metric-side convention, not a clamp on the data.

---

## Known limitations

- **Median ~9 % at 6 h is a systematic floor** for this architecture/objective —
  ensembling, more capacity, EMA, and a from-scratch transformer did not break
  it. The remaining gap is dominated by the `t=0` nucleation transient.
- **Bin 25** sits on the core/coarse head boundary and is a structural
  artifact (~60 % rel-err); bin-weight up-weighting does not fix it.
- The model conserves **sulfur mass**, not number; nucleation/coagulation
  legitimately change number, and that is left to the network.

---

## License

MIT — see `LICENSE`.
