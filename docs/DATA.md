# TOMAS Emulator — v4 CESM Training Data

Generated: 2026-04-24.
Script: `/home/ubuntu/ali/github-repos/tomas-jax/generate_training_data_v4.py`.
Output: `/data/tomas-jax-output/v4/`.

## What this data is

Training data for the TOMAS aerosol microphysics emulator. Each scenario is a
6-hour TOMAS-JAX box-model simulation seeded from a real CESM2 grid cell, with
the atmospheric environment (T, RH, OH, SO2) **evolving hour-by-hour** from
CESM rather than held frozen at the initial snapshot.

## Why v4 exists (diagnostic summary)

The v3 generation (`/data/tomas-jax-output/v3_100k/`, 90k scenarios) had three
structural issues we fixed here:

1. **Frozen environment bug.** T, RH, OH were held constant at the initial
   snapshot for the full 6 h rollout. OH has a 10–100× diurnal swing, so a
   scenario starting near dawn or dusk saw wildly wrong chemistry throughout.
   A probe of one cell (`probe_evolving.py`) showed OH dropping 300× over 6 h
   in one afternoon-into-night trajectory — completely invisible to v3.
2. **Incomplete mass accounting.** v3 built TOMAS initial state from MAM4 modes
   using only SO4 (`so4_a{1,2,3}`), ignoring POM, BC, and primary-carbon mode
   a4. In polluted BL cells where carbonaceous aerosol dominates, v3
   undercounted the aerosol population.
3. **Sampling didn't cover the grid.** v3 used random uniform sampling biased
   to 30/50/20% across a coarse altitude stratification. No guarantee any
   specific location was hit; upper stratosphere (lev 0–29, where model top
   pressure is 0) was sampled anyway and mostly filtered out.

Minor: v3 saved `dNk_dt`/`dMk_dt` derivatives that the emulator never reads
(50% of disk footprint, zero training value).

## What changed

| | v3 | v4 |
|---|---|---|
| Environment during rollout | Frozen at t=0 | Evolving hourly from CESM, interpolated to 60 s |
| Aerosol mass in initial state | `so4_a{1,2,3}` only | `so4 + pom + bc` summed per mode; mode a4 (pom+bc) folded into mode a1 |
| Gas-phase H2SO4 forcing | Arbitrary `prod_rate` × DT injected per step | Read hourly from CESM `H2SO4` (7 snapshots), linearly interpolated to per-step grid; `Gc[SRTSO4]` reset every step |
| TOMAS processes | `nucleation + coagulation + condensation` | `nucleation + coagulation + condensation` (no in-rollout SO2+OH chemistry — H2SO4 is forced from CESM) |
| Nucleation scheme | ricco_dunne | ricco_dunne (unchanged) |
| Sampling strategy | 100k random uniform, 30/50/20 altitude strata | All 55,296 horizontal cells × K=3 stratified levels × 4 seasons × 1 day × 24 hours |
| Stratification bands | levs 30–50, 50–65, 65–70 | BL (61–69), free trop (48–60), UTLS+low strat (36–47) |
| Rollout loop | Python loop, 360 step_fn calls per batch | Single `lax.scan` compiled program |
| Per-scenario disk | 176 KB compressed | ~85 KB compressed (derivatives dropped) |
| Total scenarios | 90k | 15.9M |
| Total disk | 16 GB | ~1.3 TB |

## Sampling schedule

Four seasonal start dates (approximate; uses CESM hourly time-index offsets
from the 1970-01-01 00:00 start of the FWHIST simulation, no-leap
approximation):

| Season | CESM hourly index | Approximate date |
|---|---|---|
| Winter | 131,832 | mid-January, simulation year 15 (~1985) |
| Spring | 131,832 + 90×24 = 133,992 | mid-April |
| Summer | 131,832 + 180×24 = 136,152 | mid-July |
| Autumn | 131,832 + 270×24 = 138,312 | mid-October |

Each start date: **24 consecutive hourly snapshots** (one full day).

Per snapshot: **every (lat, lon) × K=3 stratified levels** = 55,296 × 3 = 165,888 scenarios.

Total: 4 seasons × 24 hours × 165,888 = **15,925,248 scenarios**.

### Stratified K=3 level sampling

For every (lat, lon) cell, three random levels are picked — one per band. The
bands are fixed; the level *within* each band is drawn at random per cell:

- **Boundary layer**: lev 61–69 (~700–1006 hPa at surface-like PS)
- **Free troposphere**: lev 48–60 (~100–700 hPa)
- **UTLS + lower stratosphere**: lev 36–47 (~10–100 hPa, includes Junge sulfate layer)

Higher levels (0–35, upper strat + mesosphere) are not sampled — pressure is
effectively 0 there, `compute_air_density` breaks, and aerosol microphysics
is dormant.

## CESM source and fields read

Base path: `/data/cesm2.1.5_output/histSST/f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST.001/postprocessed_output/atm/proc/tseries/`

**Hourly (hour_1) — 7 snapshots per scenario (t, t+1, ..., t+6):**
- `T`, `RELHUM`, `H2SO4` — evolving environment during the rollout (volume mixing ratio for H2SO4)
- `num_a{1,2,3}`, `so4_a{1,2,3}`, `pom_a{1,4}`, `bc_a{1,4}`, `dgnumwet{1,2,3}` — initial aerosol state (read at t only)

**Daily (day_1) — 1 snapshot:**
- `PS` — surface pressure (held constant during the 6 h rollout; PS varies only on daily timescales in this archive anyway)

**Missing from archive**: `num_a4`, `dgnumwet4`, `soa_a*`, `dst_a*`, `ncl_a*`.
Mode a4 (primary carbon) mass is folded into mode a1 since there's no a4
number/diameter to support a standalone discretization. SOA/DST/NCL mass
contributions are simply lost.

## Initial state construction (MAM → TOMAS)

For each scenario:

1. Compute air density `ρ_air = P / (R_air × T)` and kg of air in the box.
2. Per mode, sum all available species into a single "SO4-equivalent" mass:
   - Mode 1: `(so4_a1 + pom_a1 + bc_a1 + pom_a4 + bc_a4) × kg_air` — a4 mass added here
   - Mode 2: `so4_a2 × kg_air`
   - Mode 3: `so4_a3 × kg_air`
3. Convert `num_a{1,2,3}` number mixing ratios to per-box counts.
4. Discretize each mode onto TOMAS 40 bins via lognormal PDF at `dgnumwet{1,2,3}`
   with fixed σ (1.6 / 1.8 / 1.6). Mode masses and numbers are distributed
   proportionally to the PDF.
5. Sum three modes to get `Nk_init`, `Mk_init_so4` — all mass in component 0 (SO4).
6. Initial `Gc[SRTSO4]` set from CESM `H2SO4(t)` (vmr → kg/box using box T, P, volume).
7. During rollout, `Gc[SRTSO4]` is **reset every 60 s step** to the per-step
   linearly-interpolated CESM `H2SO4` value (between hourly snapshots t..t+6).
   No in-rollout SO2+OH chemistry — CESM's own H2SO4 budget is the prescribed
   forcing for nucleation + condensation.

NH3 is **not in the CESM archive** we're using (only `SFNH3` surface flux).
Per-scenario NH3 is drawn from v3's altitude-split climatology: BL cells get
log-uniform 1–100 molec/cm³, free-trop cells get 0.1–3.

Organics (`org_conc` for nucleation) is derived from `pom_a1 + pom_a4` mass
mixing ratios via MW_ORG=200 g/mol assumption — the same approximation v3
used. This is a crude proxy for biogenic ULVOC/SVOC that drives organic
nucleation; a proper fix would require GEOS-Chem fields we don't have.

## Output schema

Per batch file: `/data/tomas-jax-output/v4/t{tidx:08d}/batch_s{shard:02d}_b{batch:04d}.npz`.
Compressed `numpy.savez_compressed` archive, ~5000 scenarios per batch, ~425 MB per batch.

Fields:
- `lev_idx`, `lat_idx`, `lon_idx` — `int16`, `(B,)` — CESM cell indices
- `t_idx` — `int32` scalar — CESM hourly index at scenario start
- `pres` — `float32`, `(B,)` — Pa, constant over 6 h
- `nh3_conc` — `float32`, `(B,)` — molec/cm³, constant
- `T_hourly`, `RH_hourly`, `H2SO4_vmr_hourly` — `float32`, `(7, B)` — hourly CESM values spanning the rollout (H2SO4 in volume mixing ratio, mol/mol)
- `Nk` — `float32`, `(B, 361, 40)` — number per bin per scenario per timestep (t=0..360 min)
- `Mk_SO4` — `float32`, `(B, 361, 40)` — SO4 mass per bin per scenario per timestep (kg/box)
- `Gc_SO4` — `float32`, `(B, 361)` — gas-phase H2SO4 mass (kg/box)

Per-scenario derivatives (`dNk_dt`, `dMk_dt`) are **not** saved. Recover from
consecutive states if needed: `(Nk[t+1] - Nk[t]) / DT`.

Top-level: `/data/tomas-jax-output/v4/metadata.json` — full run config.

## How to run

Prerequisites: all 8 H100s idle, `/data` has at least 2 TB free.

```bash
# Full 8-way launch (writes to /data/tomas-jax-output/v4/, logs in logs/)
bash /home/ubuntu/ali/github-repos/tomas-jax/run_v4_generation.sh

# Single-shard test (one GPU):
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/ali/github-repos/tomas-jax/generate_training_data_v4.py \
    --shard-idx 0 --n-shards 8 --output-dir /data/tomas-jax-output/v4

# Smoke test (one timestep, 5000 scenarios — ~1 min):
CUDA_VISIBLE_DEVICES=7 python generate_training_data_v4.py \
    --shard-idx 0 --n-shards 1 --max-timesteps 1 \
    --max-scenarios-per-timestep 5000 \
    --output-dir /data/tomas-jax-output/v4_smoketest

# Dry run (no work, print scenario count):
python generate_training_data_v4.py --dry-run
```

Expected wall time for full run on 8 H100s: **~12 hours** (overnight).
Expected disk: **~0.97 TB** (measured 61 KB/scenario compressed in the smoke test).

## Knobs for scaling up or down

If you later want more or less data, these parameters are defined at the top
of `generate_training_data_v4.py`:

- `K_LEVELS` (default 3) — how many levels to sample per horizontal cell
- `SEASON_START_HOURS` — list of 4 starting hourly indices; add more or shift
- `DAYS_PER_SEASON` (default 1) — consecutive days per season. Raising to 2 doubles data.
- `HOURS_PER_DAY` (default 24) — snapshots per day. Halving to 12 halves data.
- `STRATA` — the three bands defining K=3 level stratification

Precomputed tradeoffs (K=3, 4 seasons):

| Days/season | Hours/day | Scenarios | Disk | GPU-h (8 H100s) |
|---|---|---|---|---|
| 1 | 24 | 15.9M | 1.35 TB | ~15 |
| 2 | 24 | 31.8M | 2.7 TB | ~31 |
| 1 | 12 (bi-hourly) | 7.9M | 673 GB | ~8 |
| 1 | 8 (3-hourly) | 5.3M | 450 GB | ~5 |
| 2 | 8 (3-hourly) | 10.6M | 900 GB | ~10 |

## Performance

Measured on one H100 at batch size 5000, 360 steps, ricco_dunne + so2_chemistry:

| | Value |
|---|---|
| Per-timestep extract (41 slab reads from CESM) | ~5.5 s (amortized across 33 batches) |
| Per-batch JIT compile (first call per shape) | ~5 s one-time |
| Per-batch trajectory (lax.scan, 360 steps) | ~108–137 s |
| Per-batch save (compressed NPZ, ~310 MB) | ~2.6 s |
| Per-timestep total (165,888 scenarios in 33 batches) | ~60 min |
| Throughput (steady, per H100) | **~46 scenarios/s** |
| Per-scenario disk (compressed) | **~61 KB** (measured in smoke test) |
| Full 15.9M run on 8 H100s | **~12 hours wall** |

## Downstream emulator refactor (not done yet)

The emulator (`/home/ubuntu/tomas-emulator/dataset.py`) currently assumes
`params: (B, 9)` — one constant parameter vector per scenario. v4 data has
**time-varying** T/RH/OH (stored as `(7, B)` hourly). Training on v4 requires:

1. Extend `params` to `(B, T, 9)` — interpolate hourly env to per-step.
2. Update `TomasMLP` input dim handling (still `81 + 9 = 90` per step, but
   params are now per-step not per-scenario).
3. Update `TomasTrajectoryDataset.__getitem__` to return `(state_chunk, params_chunk)`
   where both are indexed by timestep.
4. Compare v4-trained vs v3-trained emulator rollout RMSE to verify the evolving-env
   upgrade actually helps.

Step 4 is the actual validation that this whole change was worth doing.

## Gotchas

- **Year 15 approximation**: `SEASON_START_HOURS[0]` = 131,832 assumes a no-leap
  calendar. If CESM uses Gregorian, the actual calendar date shifts by ~4 days
  (4 leap years). We treat this as fine — the purpose is seasonal diversity,
  not precise calendar dates.
- **PS is daily**: constant within each 6-h rollout. Real PS varies on synoptic
  timescales (~hours for frontal passages), but CESM archive here is daily.
- **OH interpolation is linear between hours**: could under-capture the noon
  peak, but at 60 s sub-sampling with hourly CESM this is hard to improve
  without a photochemistry model.
- **Mode a4 mass fold**: dumps primary-carbon mass into the accumulation mode
  lognormal at mode 1's dgnumwet. This biases carbonaceous aerosol sizes upward
  (real a4 has dgnum ~50 nm vs mode 1 ~150 nm). Only matters for cells dominated
  by primary carbon (urban/BB plumes).
- **Gc[SO4] starts at 0**: always. The first few minutes of every rollout spin
  up the gas phase from SO2 oxidation. This is physically cleaner than v3's
  arbitrary `prod_rate × DT` injection.

## References

- v3 generator (prior art): `/home/ubuntu/ali/github-repos/tomas-jax/generate_training_data_v3.py`
- tomas-jax library CLAUDE.md: `/home/ubuntu/ali/github-repos/tomas-jax/CLAUDE.md`
- Probes used to validate the design:
  - `/home/ubuntu/tomas-emulator/probe_one_scenario.py` — single cell, constant env
  - `/home/ubuntu/tomas-emulator/probe_evolving.py` — single cell, evolving env (showed the OH-collapse motivation)
  - `/home/ubuntu/tomas-emulator/probe_batched.py` — N=5000 throughput measurement
