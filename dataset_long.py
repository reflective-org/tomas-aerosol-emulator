"""Loader for the 24h long-horizon dataset (gen_v4_longhorizon.py output).

Same NPZ schema as v4_repr_1M except the trajectory is `hours*60+1` minutes with
`hours+1` forcing anchors instead of 361/7. Mirrors dataset_v4.load_v4_batches
exactly (same cond channel order and log transforms) so norm_stats from the 6h
runs apply unchanged.
"""
import numpy as np
from pathlib import Path

from dataset_v4 import N_BINS, N_COND, LOG_EPS   # noqa: F401  (re-export for callers)


def _interp_hourly(hourly, n_steps):
    """hourly: (B, n_h+1) at whole hours -> (B, n_steps+1) per minute."""
    n_h = hourly.shape[1] - 1
    f = np.arange(n_steps + 1, dtype=np.float32) / 60.0
    lo = np.floor(f).astype(np.int32).clip(0, n_h)
    hi = np.minimum(lo + 1, n_h)
    w = (f - lo).astype(np.float32)
    return (1.0 - w) * hourly[:, lo] + w * hourly[:, hi]


def load_long_batches(root, max_scenarios=None, rank=0, world_size=1, verbose=True):
    """Returns Nk (N,S+1,40), Mk (N,S+1,40), Gc (N,S+1), cond (N,S+1,6), n_steps."""
    all_files = sorted(Path(root).glob("t*/batch_s*_b*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No batch files under {root}")
    files = ([f for i, f in enumerate(all_files) if i % world_size == rank]
             if world_size > 1 else all_files)
    if verbose:
        print(f"  rank {rank}/{world_size}: {len(files)}/{len(all_files)} long files")

    Nl, Ml, Gl, Cl = [], [], [], []
    total = 0
    n_steps = None
    for f in files:
        d = np.load(f)
        Nk = d["Nk"]; Mk = d["Mk_SO4"]; Gc = d["Gc_H2SO4"]
        B, S = Nk.shape[0], Nk.shape[1] - 1
        if n_steps is None:
            n_steps = S
        elif S != n_steps:
            raise ValueError(f"{f}: {S} steps, expected {n_steps}")
        valid = (np.isfinite(Nk.reshape(B, -1)).all(1)
                 & np.isfinite(Mk.reshape(B, -1)).all(1) & np.isfinite(Gc).all(1))
        T_h = d["T_hourly"].T[valid]; RH_h = d["RH_hourly"].T[valid]
        H_h = d["H2SO4_vmr_hourly"].T[valid]
        pres = d["pres"][valid]; nh3 = d["nh3_conc"][valid]
        fion = (d["fion"][valid] if "fion" in d.files
                else np.full(valid.sum(), 3.0, np.float32))
        Nk = Nk[valid]; Mk = Mk[valid]; Gc = Gc[valid]; B = Nk.shape[0]

        cond = np.empty((B, S + 1, N_COND), np.float32)
        cond[:, :, 0] = pres[:, None]; cond[:, :, 1] = nh3[:, None]
        cond[:, :, 2] = _interp_hourly(T_h, S)
        cond[:, :, 3] = _interp_hourly(RH_h, S)
        cond[:, :, 4] = _interp_hourly(H_h, S)
        cond[:, :, 5] = fion[:, None]

        Nl.append(Nk); Ml.append(Mk); Gl.append(Gc); Cl.append(cond)
        total += B
        if max_scenarios is not None and total >= max_scenarios:
            break

    Nk = np.concatenate(Nl, 0); Mk = np.concatenate(Ml, 0)
    Gc = np.concatenate(Gl, 0); cond = np.concatenate(Cl, 0)
    if max_scenarios is not None:
        Nk, Mk, Gc, cond = Nk[:max_scenarios], Mk[:max_scenarios], Gc[:max_scenarios], cond[:max_scenarios]
    return Nk, Mk, Gc, cond, n_steps
