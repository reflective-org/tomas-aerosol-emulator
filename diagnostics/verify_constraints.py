"""Is the sulfur-increment BOX constraint exact on ground truth?

  0 <= sum(Mk)[t+dt] - sum(Mk)[t] <= sum_{j<dt} H_forced[t+j]

Unlike the single-step equality budget, this needs no intermediate residual gas
and is valid at ANY dt: aerosol sulfur can only be gained from the injected gas
(no deposition in this box model) and coagulation conserves mass.
"""
import os
import numpy as np
from dataset_v4 import load_v4_batches, build_state_array_v4, transform_cond

LOG_CONST = float(np.log(1e-6*1e6*98e-3/6.022e23/1.380649e-23))
Nk, Mk, Gc, cond = load_v4_batches(os.environ.get("DATA_DIR", "data/v4_repr_1M"), max_scenarios=140000,
                                   rank=0, world_size=7, verbose=False)
sel = np.random.default_rng(0).permutation(Nk.shape[0])[:4000][:800]
ls = build_state_array_v4(Nk[sel], Mk[sel], Gc[sel]); ct = transform_cond(cond[sel])
gt = np.exp(ls.astype(np.float64)); Ms = gt[:, :, 40:80].sum(-1)
H = np.exp(ct[..., 4]+ct[..., 0]-np.log(ct[..., 2])+LOG_CONST).astype(np.float64)
GRID = list(range(30, 361, 30)); prev = [0]+GRID[:-1]
d = Ms[:, GRID]-Ms[:, prev]
base = Ms[:, prev]
Hw = np.stack([H[:, p:p+30].sum(1) for p in prev], 1)

print("=== lower bound: is sum(Mk) non-decreasing over 30-min windows? ===")
print(f"  frac dM >= 0                 : {(d >= 0).mean():.6f}")
print(f"  frac dM >= -1e-9 * M         : {(d >= -1e-9*base).mean():.6f}")
neg = (d/np.maximum(base, 1e-300))[d < 0]
if neg.size:
    a = np.abs(neg)
    print(f"  when negative, |dM|/M        : p50 {np.median(a):.2e} p90 {np.percentile(a,90):.2e} "
          f"max {a.max():.2e}  (n={neg.size}/{d.size})")

print("\n=== upper bound: is dM <= gas injected over the window? ===")
ok = d <= Hw*(1+1e-9)
print(f"  frac dM <= sum_H             : {ok.mean():.6f}")
ex = ((d-Hw)/np.maximum(Hw, 1e-300))[~ok]
if ex.size:
    print(f"  when violated, excess/sum_H  : p50 {np.median(ex):.2e} p90 {np.percentile(ex,90):.2e} "
          f"max {ex.max():.2e}  (n={ex.size}/{d.size})")

print("\n=== how tight is the upper bound in practice? dM / sum_H ===")
fr = d/np.maximum(Hw, 1e-300)
for q in (10, 50, 90, 99, 100):
    print(f"  p{q:<3d}: {np.percentile(fr, q):.4f}")

print("\n=== single-step equality budget, for contrast ===")
Hs = H[:, prev]
print(f"  dM vs single-step H: median ratio dM/H = {np.median(d/np.maximum(Hs,1e-300)):.3f}"
      f"  (30 would mean 'all 30 minutes of gas'; 1 means 'one minute')")
