# TOMAS Aerosol Microphysics Emulator — bin-token transformers

> **Status: active development.** Accuracy, APIs and the shipped checkpoints may change.

A neural surrogate for the **TOMAS** sectional aerosol microphysics model (40 mass-doubling
bins, binary H₂SO₄–H₂O nucleation, coagulation, condensation, SO₂ + OH chemistry). Given a
box's size distribution, gases and the forcing over the next 6 hours, it returns the
state every 30 minutes through those 6 hours. Longer horizons are chained block by block.

**This branch ships two bin-token transformers** trained with the same recipe at two sizes:

| checkpoint | d × layers | params | cost (fp16, compiled, H200) | 6 h totals N · M (p50/p90) | day-14 injection N · M (p50/p90) |
|---|---|---|---|---|---|
| `checkpoints/bintf_d384_l6` (**production**) | 384 × 6 | 7.16 M | **2.9 µs/cell** | 3.1/16 · 2.6/13 % | 20/68 · 10/36 % |
| `checkpoints/bintf_d256_l4` (fast) | 256 × 4 | 2.15 M | **1.1 µs/cell** | 4.7/25 · 4.9/23 % | 17/71 · 10/46 % |

Error = |emulator − TOMAS| / TOMAS on held-out burst cells of the SAI injection population
(the hardest cells), chained self-rollout from the initial state. Quiet background cells are
at 0.1 % at 6 h. 1.5 M grid cells × one 6-hour step take 4.3 s (d384) or 1.7 s (d256) on one
H200, versus 24–74 s for the TOMAS-JAX fast solver.

Both checkpoints are the weight average of the three best epochs of their finishing run
(`avg_of` inside the file). Both use the same `checkpoints/norm_stats.npz`; never recompute it.

## Quick start

```bash
pip install -r requirements.txt
python example.py                      # 10 ppbv SO2 pulse into a dense layer, 48 h, d384
python example.py --ckpt bintf_d256_l4 --hours 336
```

```python
from normalization import state_to_log, cond_to_log, log_to_state
from inference import load_model, rollout

emu  = load_model("checkpoints/bintf_d384_l6")          # fp16 autocast + torch.compile
s0   = state_to_log(Nk, Mk, h2so4_kg, so2_kg)           # (B, 82)  per-box quantities, floors applied
cond = cond_to_log(pres_pa, temp_k, rh, oh_cm3)         # (B, 13, 6) for one block, (B, 12K+1, 6) for K blocks
traj = rollout(emu, s0, cond)                           # (B, 12K+1, 82) log-states every 30 min
state_6h = log_to_state(traj[:, 12])                    # dict: Nk, Mk, h2so4_kg, so2_kg
```

## What the model computes

**Input.** One box, one 6-hour step: the 82-value state at t = 0 (40 bin numbers, 40 bin
sulfate masses, H₂SO₄ gas, SO₂ gas, all as logs of per-box quantities, box = 1 m³) and the
forcing at 13 points, t = 0, 0.5, …, 6 h: pressure, temperature, relative humidity, OH, an
SO₂ source rate and a dilution rate. **Only the 6 hours of forcing are needed for a 6-hour
step.** The state and forcing layout, units and floors are documented in `normalization.py`.

**Output.** 12 states, one every 30 minutes, the last being the 6-hour state. The network is
a transformer over 42 tokens (40 bins + 2 gases); each token carries its bin's number and
mass plus the gases and the whole forcing window, attention runs across bins, and the head
emits all 12 time steps at once as a residual on the initial state. Nothing is stepped
through time inside a block, which is where the speed comes from (`bintf_model.py`).

**Chaining.** For a longer horizon the 6-hour output becomes the next block's initial state,
with the next 13 forcing points (`inference.rollout`). The checkpoints were trained on chains
up to 56 blocks (14 days), so chained rollouts stay bounded; the error numbers above are
chained self-rollouts.

## Floors (load-bearing)

The training corpus contains no zeros. A zero pushed through the log is many standard
deviations outside the training distribution and the network returns nonsense. Two floors
were found in a coupled climate run and are applied by `state_to_log` / `cond_to_log`:

- **SO₂ ≥ 1e-17 kg/box.** A zero SO₂ encoded at 1e-30 is 7σ out and produced 1660× the
  physics' aerosol mass in that cell.
- **OH ≥ 100 molec/cm³.** Night-time OH of exactly zero is 17σ out and produced spurious
  1e8 cm⁻³ nucleation bursts in every polar cell.

Also: Nk ≥ 1e-20 /box, Mk and gases ≥ 1e-30 kg/box, RH clipped to [1e-3, 1].

## Precision

fp16 autocast reproduces fp32 within rounding at every horizon (no non-finite outputs over
5 populations × 14-day chains). **bf16 is not lossless**: day-14 number p90 widens from 71 to
84 % on the d256 model, so `load_model` refuses it. `torch.compile(mode="reduce-overhead")`
halves the half-precision cost; it does nothing for fp32. Batch size must be constant for a
compiled graph; `rollout` pads the last batch.

## Known limits

- **Nucleation-burst number.** A pulse of SO₂ into clean, cold air nucleates within minutes;
  the emulator must infer the burst from the forcing and lands 20–60 % off in total number
  during the burst, while total mass stays within a few percent. Quiet and coagulating cells
  are at the percent level.
- **SO₂ source channel.** The continuous-source forcing channel (index 4) is unstable on the
  d384 checkpoint: SO₂ is taken up ~2× too fast in the first block and explodes in the second.
  Apply injections to the gas state between steps instead (which is what the coupled climate
  model does) and leave channel 4 at its zero value.
- **Initial conditions** must be bin-consistent (mass per particle inside each bin's mass
  range). States produced by TOMAS are; a modal-to-sectional mapping that spreads number and
  mass with the same pdf is not, and is out of distribution.

## Training recipe

Data: 6-hour windows and 14-day chains from a CESM/WACCM-CARMA-driven TOMAS-JAX box corpus
(global background, stratosphere, SAI pulse injection, continuous injection, idealized
populations; ~12 M scenarios). Not included here.

1. 6-hour pre-training from scratch: MSE on normalized log-states + total-number / total-mass
   terms + 4× weight on the 16 smallest bins (`training/frontier_transformers.py`, `ARCH=bintf`).
2. K-curriculum chain training, K = 2 → 48 blocks (12 h → 14 days), warm-started from step 1,
   burst-active chains oversampled ×3, data-parallel on 5 GPUs (`training/frontier_bintf_kchain.py`).
3. Rebalanced finishing (one K = 48 stage, 8 epochs): 6-hour term weight ×8, per-block chain loss
   normalized by the detached block error so early blocks contribute as much gradient as late
   ones, and a teacher-forced one-step loss from every true state along each chain.
4. Weight average of the three best epochs by day-14 totals (`training/frontier_bintf_ckpt_avg.py`).

The `training/` directory is the code as run, unmodified. It depends on the internal data
layout (`dataset_v8.py`, `frontier_common.py`) and on absolute paths of the training machine;
it is included for reference, not as a runnable pipeline.

## Files

| file | purpose |
|---|---|
| `bintf_model.py` | the architecture (identical to training) |
| `normalization.py` | state / forcing layout, units, floors, log transform, normalization stats |
| `inference.py` | `load_model` (fp16 + compile) and chained `rollout` |
| `example.py` | one box, SO₂ pulse, 48 h, prints the time course |
| `checkpoints/` | the two weight files and `norm_stats.npz` |
| `training/` | training, finishing and checkpoint-averaging scripts as run |

Other branches of this repository hold earlier model generations (`main`: 60-s multi-head
MLP; `dt-champion`: Δt-conditioned MLP; `axial-aug5`: axial attention).
