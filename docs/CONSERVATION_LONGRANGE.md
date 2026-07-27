# Mass conservation + long-range stability for the axial emulator

Work on `axial_aug5` (non-AR axial trajectory operator, 29.5M params).
Goal: enforce sulfur mass conservation, make the model stable when chained past
its 6h training horizon, without losing 6h accuracy.

Date: 2026-07-26. All numbers on 800 held-out scenarios unless stated.

---

## 1. The sulfur EQUALITY budget is the wrong constraint at Δt=30 (refuted)

The AR-MLP champion (`model_conserved_v2.ConservativeMultiHeadMLPV2`) enforced

```
sum(Mk)[t+dt] + Gc[t+dt] = sum(Mk)[t] + H2SO4_kg_forced[t]        (E)
```

by a uniform log-shift on (Mk, Gc), with start-of-interval forcing. Porting it to
the axial model and chaining it across the 12 horizons **holds the invariant to
1e-6 and wrecks the mass**:

| variant | Mk p50 | Mk p90 | totM p50 | totM p90 | budget resid p50 |
|---|---|---|---|---|---|
| raw aug5 | 0.032 | **0.209** | 0.010 | **0.034** | 4.7e-3 |
| chained (E), clamp ln10 | 0.061 | 0.347 | 0.036 | 0.437 | 2.3e-15 |
| chained (E), clamp ln1.5 | 0.061 | 0.346 | 0.036 | 0.436 | — |
| local anchor (no telescoping) | 0.035 | 0.217 | 0.013 | 0.054 | 1.6e-3 |
| gas-completion (Mk untouched) | 0.032 | 0.210 | 0.010 | 0.039 | 3.4e-15 |
| regime-gated, τ=0.1 | 0.056 | 0.328 | 0.020 | 0.173 | 3.3e-15 |
| regime-gated, τ=0.1, λ=0.5 | 0.043 | 0.245 | 0.010 | 0.093 | 1.8e-3 |

**Why it fails.** Three measured facts:

1. **Total sulfur is bimodal between reservoirs.** Gas share
   `Gc/(ΣMk+Gc)` has p50 = 1e-4 but p90 = 0.9985; 24.6% of (scenario, horizon)
   cells are gas-dominated. Where gas dominates, (E) constrains the gas and says
   nothing about the aerosol — and it is *wrong*, because the gas is
   re-prescribed every minute, so `Gc[t+30]` tracks `H[t+30]`, not `H[t]`.
2. **Where the aerosol dominates, (E) mis-sizes the increment.** It forces the
   30-minute mass gain to equal *one* minute of injected gas. Truth: median
   `dM/H_single` = 0.43, and only ~1.4% of the 30 minutes' injected gas actually
   condenses. (E) closes to p50 0.0016 only in the trivial sense that ΣMk barely
   moves; the *increment* is off by ~30×, and chaining accumulates that.
3. **The tail is unusable.** GT closure of (E) at Δt=30 is p50 0.0016 / p90 0.039
   but **p99 2405**.

Putting the whole correction in the gas keeps the aerosol intact but destroys Gc
by catastrophic cancellation (p90 1.2e7) whenever the aerosol dominates.

The AR champion could use (E) only because it *trained* with it. Applied post-hoc
to a model trained without it, (E) injects its own approximation error.

Anchor choice was checked and is not the issue: GT closure with start / mid /
mean-of-interval forcing is p50 0.0016 / 0.0012 / 0.0012 — all equivalent, all
with p99 = 1.0+.

## 2. What IS exact: monotonicity (adopted)

There is no deposition in this box model and coagulation conserves mass, so
aerosol sulfate is non-decreasing:

```
sum(Mk)[t+dt] >= sum(Mk)[t]                                        (M)
```

Verified on GT 30-min windows: holds for **99.24%** of pairs, and every violation
is float32 storage noise (max |dM|/M = **1.0e-6**). Unlike (E) it is one-sided,
needs no forcing reconstruction, is valid at any Δt, and is **inert when the
model is already physical**.

The upper bound (`dM <= gas injected`) was also tested and **rejected**: 1.4% of
GT windows violate it by large margins (near-empty boxes where ΣMk sits at the
floor), so it is not safe as a hard constraint. It is kept only as a soft
one-sided training penalty.

**aug5 violates (M) in 25% of 6h horizons and 59% of chained 24h steps** — real,
unphysical sulfur destruction, and exactly the mechanism that compounds when
chaining.

Post-hoc cost of enforcing (M) + the bin-geometry clamp on aug5 (`mode="mono"`):

| mode | Nk p90 | Mk p50 | Mk p90 | totM p50 | totM p90 | (M) violations |
|---|---|---|---|---|---|---|
| none | 0.236 | 0.032 | 0.209 | 0.010 | 0.034 | 25.20% |
| geom only | 0.236 | 0.032 | 0.209 | 0.010 | 0.034 | 25.20% |
| **mono** | **0.236** | 0.033 | 0.213 | 0.011 | **0.033** | **7.69%** |

Nk untouched, Mk p90 +0.004, totM slightly *better*. The residual 7.7% is a
genuine feasibility conflict: when every bin sits at its mass ceiling, ΣMk cannot
rise without moving particles between bins, which a projection cannot do — that
part has to be learned, which is why the constraint is also applied during
training.

## 3. Long-range: the error is exposure bias, not per-window skill

There was no ground truth past 6h anywhere in the project. Generated
`data/v4_long24h`: **8,004 CESM-forced 24h TOMAS trajectories**
(1441 stored minutes, 25 hourly anchors, BL/FT/UT stratified, v4 σ convention,
`gen_longhorizon.py`). Physics validated against v4 GT (one-step median
rel-err 1.1e-6) and the per-minute sulfur invariant closes to **p50 1.1e-8**,
matching v4's 1.9e-8. Unlike `gen_idealized_heldout.py` (which writes a constant
`Gc0`), this stores the **true post-step residual gas**, so the invariant is
actually usable on it.

Chained eval (`evaluate_chain.py`), aug5, chain started from a
TOMAS-consistent state at t=60min, 3 windows:

| horizon | self-chained Nk p50 / p90 | GT-restarted Nk p50 / p90 | self totM p90 |
|---|---|---|---|
| 4h | 0.026 / 0.105 | 0.026 / 0.105 | 0.032 |
| 7h | 0.027 / 0.127 | 0.027 / 0.127 | 0.033 |
| 10h | 0.050 / 0.217 | 0.026 / 0.108 | 0.064 |
| 13h | 0.052 / 0.240 | 0.028 / 0.136 | 0.065 |
| 16h | 0.073 / 0.323 | 0.026 / 0.107 | 0.095 |
| 19h | 0.075 / 0.345 | 0.028 / 0.135 | 0.096 |

**GT-restarted error is flat out to 19h** (p90 0.105 → 0.135): per-window skill
does not degrade at all on out-of-horizon states. **Self-chained error triples**
(0.105 → 0.345). So essentially *all* long-range error is exposure bias — the
model has never seen its own output as an input. This is directly fixable by
pushforward training, and it is what `train_axial_stable.py` does.

## 4. Separate finding: t=0 initial conditions are bin-mass-inconsistent

Chaining from t=0 blows up (Nk p90 5.7e2 at 3h, 6.9e4 at 6h) while chaining from
t=60 is clean. Cause: MAM4→sectional ICs distribute Nk and Mk with the *same*
lognormal pdf, so `Mk[k]/Nk[k]` is constant in k instead of tracking `XK[k]`.

| dataset | t=0 off-grid populated bins | median (Mk/Nk)/XK | t≥1 off-grid |
|---|---|---|---|
| v4_repr_1M (training set) | 91.99% | 0.345 | 0.00% |
| v4_long24h (new) | 88.50% | **1.45** | 0.00% |

TOMAS's rebinning fixes this at the first step, so every t≥1 state is on-grid.
Both datasets are inconsistent at t=0, but at *different* points — 1.45 vs 0.345
— so the new ICs are out-of-distribution for the model, which is why window 0 is
catastrophic while every later window is fine.

Note which side is unphysical: a bin-consistent state has `(Mk/Nk)/XK ∈ [1,2]`.
The new ICs sit at 1.45 (**consistent**); **v4's t=0 sits at 0.345, below the bin
floor**. The model was trained on the unphysical convention, so it is the
physically-correct IC that is out-of-distribution.

Two candidate fixes were tested on the 24h holdout (window 0 only) and **both
rejected** (`diagnostics/verify_constraints.py` and friends):

| IC treatment | 6h Nk p50 | 6h Nk p90 |
|---|---|---|
| raw MAM4 t=0 | 0.047 | 68970 |
| bin-geometry projection, total mass preserved | 2.090 | 189 |
| rescale Mk to match v4's 0.345 ratio | 0.815 | 49497 |
| rescale Mk ×0.1 | 0.940 | 25524 |
| *TOMAS t=60 state (reference)* | *0.027* | *0.127* |

Geometry sanitization trades the catastrophic tail for a broken median; mass
rescaling barely dents the tail. So the failure is **not** the mass-per-particle
convention alone — something else in the MAM4→sectional construction is
off-distribution. Since a bin-consistent state gives p90 0.127, the model itself
is fine; the IC path is the problem.

This is an **IC-construction problem, not a chaining problem**, but it matters
for deployment: coupling this emulator to CESM means handing it exactly this kind
of MAM4-derived IC. The remaining follow-up is to recover the v4 generator's
exact IC convention (not available in this repo) or regenerate training ICs bin-consistently and retrain. Chained
evaluation defaults to `--start-min 60` to keep the two questions separate.

## 4b. Free-running 7-day probe: conservation stops the sulfur bleed

`the 7-day free-running probe --conserve {none,mono}`, aug5, 28 chained windows, 6 CESM
cells. All finite in both cases (no blow-up), but the unconstrained chain
**destroys sulfur**:

| cell | steps where ΣMk decreases | M(7d)/M(0) |
|---|---|---|
| | none → mono | none → mono |
| BL lev61 | 54.2% → **13.4%** | 0.075 → 0.069 |
| BL lev64 | 55.4% → **9.8%** | 0.835 → 1.34 |
| FT lev59 | 33.6% → **5.1%** | 1.97 → 2.57 |
| BL lev66 | 58.9% → **8.0%** | 2.50 → 3.62 |
| UT lev43 | 80.7% → **10.1%** | 0.321 → 1.08 |
| UT lev37 | 55.7% → **6.2%** | 2.64 → 4.54 |

Unconstrained, one cell loses **92% of its sulfur over 7 days** and another 68% —
pure numerical destruction, since the box model has no removal process. The
projection cuts violating steps by 5–8× and moves 5 of 6 cells to physical growth
(sulfur accumulating from continued H2SO4 injection). The residual violations are
the geometry-ceiling infeasibility plus the `clamp(-20,20)` guardrail in the
7-day script, and are what training with the constraint is meant to remove.

## 5. The stability fine-tune: large chained gain, but confounded by domain shift

`axial_cons1` = aug5 + two-window pushforward (p=0.75) + interior starts
(t≥60) + monotonicity projection + soft budget penalty + 6h CESM replay
(weight 1.0). 20 epochs, ~7 min/epoch on one shared H100. Best = epoch 19
(val6h 0.188→0.079, valchain 0.0268→0.0106).

**Chained 24h (400 held-out scenarios, start t=60, self-chained):**

| horizon | aug5 Nk p50/p90 | cons1 Nk p50/p90 | aug5 totM p90 | cons1+mono totM p90 |
|---|---|---|---|---|
| 4h | 0.026 / 0.105 | **0.004 / 0.016** | 0.032 | **0.002** |
| 10h | 0.050 / 0.217 | **0.007 / 0.039** | 0.064 | **0.007** |
| 19h | 0.075 / 0.345 | **0.012 / 0.065** | 0.096 | **0.014** |

Monotonicity violations over the chain: aug5 49.4% → cons1 20.2% → cons1+mono
**8.8%**. 19h per-bin p90 improves **5.3×** and total-mass error **7×**.

**But two caveats make this not yet a clean win:**

1. **6h no-regression FAILS.** On the canonical v4_repr_1M held-out set, cons1 is
   *worse*: Nk p90 0.236 → **0.327**, Mk p90 0.209 → **0.310**. `diagnostics/domain_shift.py`
   settles the cause — the p90 deficit ratio is **flat at 1.38–1.43 across all 12
   horizons**, so it is uniform domain shift, not a t=0-transient artifact. cons1
   traded v4-distribution skill for v4_long24h-distribution skill.
2. **The chained gain is largely domain adaptation, not pushforward.** cons1 beats
   aug5 by 6.5× even at 4h under the *GT-restarted* protocol (0.016 vs 0.105) —
   a single window with no chaining at all, where pushforward cannot help. The
   unconfounded measure of pushforward's effect is the within-model amplification
   ratio self/gtstart at 19h: **aug5 2.56×, cons1 2.71× — unchanged.** So the
   fine-tune did not make the model more robust to its own errors; it made the
   errors smaller on this data distribution, and compounding scaled down with them.

## 6. `axial_cons2` — the recommended model

Same recipe warm-started from aug5 with **replay weight 3.0** (was 1.0) and
**lr 3e-5** (was 8e-5), 16 epochs. Best = epoch 14 (val6h 0.0634, valchain 0.0132).
Trained monotonically on both objectives with no oscillation, unlike cons1.

### Headline comparison — use `axial_cons2` with `mode="mono"`

| metric | aug5 | cons2 + mono | change |
|---|---|---|---|
| 6h canonical Nk p90 (v4_repr_1M) | **0.236** | 0.259 | 10% worse |
| 6h canonical Mk p90 | **0.209** | 0.234 | 12% worse |
| 6h monotonicity violations | 25.2% | **5.2%** | 4.8× better |
| 24h chained Nk p90, in-distribution | 0.345 | **0.078** | **4.4× better** |
| 24h chained Nk p90, out-of-sample season | 0.408 | **0.109** | **3.7× better** |
| 24h chained totM p90 | 0.096 | **0.015** | **6.4× better** |
| 24h chained monotonicity violations | 59.0% | **8.1%** | 7.3× better |
| 7-day M(7d)/M(0), 6 cells | 0.075 … 2.64 (two cells destroy 68% / 92%) | **1.05 … 3.25 (all ≥ 1)** | no destruction |

The 6h gate (Nk p90 ≤ 0.24) is **narrowly missed at 0.259**. The trade-off is
monotone in replay weight and now quantified: replay 1.0 → 6h +39% / 19h 0.064;
replay 3.0 → 6h **+10%** / 19h 0.078. Pushing replay to ~6 should close the
remaining 10% at a modest chained cost — the obvious next iteration.

### Out-of-sample generalization (the confound test)

Generated `v4_long24h_oos`: 1,000 more 24h trajectories at CESM hours 127452 and
144242 — a ±6-month offset, **outside the v4 training window entirely**, so
out-of-sample for *both* models. cons2's advantage holds (19h Nk p90 0.408 →
0.109, totM 0.090 → 0.018), so the gain is real skill, not memorization of the
six training `t_starts`.

It is still not purely a chaining fix: cons2 also leads 4× at 4h under
GT-restart, a single window where chaining plays no role, so part of the gain is
adaptation to the new generator's distribution (which differs from v4 in IC
construction and nh3 sampling). The unconfounded amplification ratio
self/gtstart at 19h improves only modestly, **2.57× → 2.37×**.

### Honest summary of what each ingredient bought

- **Monotonicity projection**: unambiguous win. Cheap at 6h (+0.004 Mk p90),
  cuts chained violations 7×, and on the 7-day probe converts sulfur destruction
  into physical accumulation. Works post-hoc on any checkpoint; works better when
  trained with (it removes the geometry-ceiling infeasibility that the projection
  alone cannot fix — BL lev61 goes 0.069 → 1.16).
- **24h ground-truth data**: the enabling deliverable. It is what turned "does it
  blow up?" into measured error curves, and it exposed both the domain-shift
  confound and the t=0 IC problem.
- **Pushforward training**: lowers absolute long-range error a lot, but does
  **not** materially reduce relative error amplification (2.57× → 2.37×). Most of
  the headline gain is distributional. Reported as a partial result, not a fix
  for compounding.

## 7. Files

- `model_conserved_axial.py` — `ConservativeAxialTraj` (`mode` in none/geom/mono/equality)
  plus `sulfur_budget_penalty` (soft, one-sided) for training.
- `gen_longhorizon.py` — 24 h CESM-forced TOMAS generator storing true post-step gas.
- `dataset_long.py` — loader for the 24 h schema; mirrors `dataset_v4` conventions.
- `evaluate_chain.py` — chained evaluation vs 24 h truth, self- vs GT-restarted.
- `train_axial_stable.py` + `scripts/train_axial_stable.sh` — the fine-tune.
- `diagnostics/` — the scripts backing the claims above: `verify_constraints.py`
  (are the constraints exact on ground truth?), `budget_variants.py` (the refuted
  equality-budget anchorings), `domain_shift.py` (per-horizon deficit profile).

## 8. Recommended next steps

1. **Close the last 10% of the 6h gap**: rerun `train_axial_stable.py` with
   `--replay-weight 6` (frontier is monotone; everything else unchanged).
2. **Fix the IC path** (section 4) — the largest single risk to deployment. Needs
   the v4 generator's mass-distribution convention, or bin-consistent
   regeneration + retraining.
3. **Reduce compounding directly**, since pushforward barely moved the 2.4x
   amplification: try 3+ chained windows in training, or backprop through the
   window boundary instead of detaching.
