"""TOMAS-JAX v4 dataset: load v4_repr_1M-style NPZ batches with evolving forcings.

Schema differences vs v2 dataset.py:
  - Gc renamed: Gc_SO4 -> Gc_H2SO4
  - Constant scalar params reduced to (pres, nh3_conc)
  - Evolving forcings (T, RH, H2SO4_vmr) come as 7 hourly anchors per scenario,
    interpolated to per-step (361 steps over 6h, anchored at steps 0,60,...,360).
  - Conditioning vector is per-step, length 5: [pres, nh3_conc, T, RH, H2SO4_vmr]
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

N_BINS = 40
N_STATE = 81           # 40 Nk + 40 Mk_SO4 + 1 Gc_H2SO4
N_COND = 5             # pres, nh3_conc, T, RH, H2SO4_vmr
N_STEPS = 361          # 0..360 over 6h
LOG_EPS = 1e-30

COND_NAMES = ["pres", "nh3_conc", "T", "RH", "H2SO4_vmr"]
LOG_COND_IDX = [0, 1, 4]   # pres (Pa), nh3_conc, H2SO4_vmr span many decades


def _interp_hourly_to_steps(hourly):
    """hourly: (B, 7) at hours 0..6. Returns (B, 361) interpolated to per-step."""
    step = np.arange(N_STEPS, dtype=np.float32)
    f_hour = step / 60.0
    lo = np.floor(f_hour).astype(np.int32).clip(0, 6)
    hi = np.minimum(lo + 1, 6)
    w = (f_hour - lo).astype(np.float32)
    return (1.0 - w) * hourly[:, lo] + w * hourly[:, hi]


def load_v4_batches(root, max_scenarios=None, verbose=True, rank=0, world_size=1):
    """Glob batch_*.npz under root/t*, return Nk, Mk, Gc, cond.

    With world_size > 1, each rank loads only files where (idx % world_size == rank)
    -- a striped shard. Total memory per rank scales as ~1/world_size of the full
    set, which is required for DDP runs on hosts where N full copies would OOM.
    Callers must NOT then use DistributedSampler, since each rank already holds a
    non-overlapping subset of the data.

    Nk:   (N, 361, 40) float32
    Mk:   (N, 361, 40) float32
    Gc:   (N, 361)     float32
    cond: (N, 361, 5)  float32   [pres, nh3_conc, T, RH, H2SO4_vmr]
    """
    all_files = sorted(Path(root).glob("t*/batch_s*_b*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No batch files under {root}")
    if world_size > 1:
        files = [f for i, f in enumerate(all_files) if i % world_size == rank]
        if verbose:
            print(f"  rank {rank}/{world_size}: loading {len(files)}/{len(all_files)} files (shard)")
    else:
        files = all_files

    Nk_l, Mk_l, Gc_l, cond_l = [], [], [], []
    total = 0
    for f in files:
        d = np.load(f)
        Nk = d["Nk"]
        Mk = d["Mk_SO4"]
        Gc = d["Gc_H2SO4"]

        T_h = d["T_hourly"].T            # (B, 7)
        RH_h = d["RH_hourly"].T
        H2SO4_h = d["H2SO4_vmr_hourly"].T
        pres = d["pres"]
        nh3 = d["nh3_conc"]
        B = Nk.shape[0]

        # Drop scenarios with any NaN/Inf in the rollout outputs (should be 0 per scan)
        valid = (
            np.isfinite(Nk.reshape(B, -1)).all(axis=1)
            & np.isfinite(Mk.reshape(B, -1)).all(axis=1)
            & np.isfinite(Gc).all(axis=1)
        )
        if not valid.all():
            n_bad = (~valid).sum()
            if verbose:
                print(f"  {f.name}: dropped {n_bad}/{B} non-finite scenarios")
            Nk = Nk[valid]; Mk = Mk[valid]; Gc = Gc[valid]
            T_h = T_h[valid]; RH_h = RH_h[valid]; H2SO4_h = H2SO4_h[valid]
            pres = pres[valid]; nh3 = nh3[valid]
            B = Nk.shape[0]

        T_step = _interp_hourly_to_steps(T_h)
        RH_step = _interp_hourly_to_steps(RH_h)
        H2SO4_step = _interp_hourly_to_steps(H2SO4_h)

        cond = np.empty((B, N_STEPS, N_COND), dtype=np.float32)
        cond[:, :, 0] = pres[:, None]
        cond[:, :, 1] = nh3[:, None]
        cond[:, :, 2] = T_step
        cond[:, :, 3] = RH_step
        cond[:, :, 4] = H2SO4_step

        Nk_l.append(Nk); Mk_l.append(Mk); Gc_l.append(Gc); cond_l.append(cond)
        total += B
        if verbose and len(Nk_l) % 25 == 0:
            print(f"  loaded {len(Nk_l)}/{len(files)} batches, {total:,} scenarios")
        if max_scenarios is not None and total >= max_scenarios:
            break

    Nk = np.concatenate(Nk_l, 0)
    Mk = np.concatenate(Mk_l, 0)
    Gc = np.concatenate(Gc_l, 0)
    cond = np.concatenate(cond_l, 0)
    if max_scenarios is not None:
        Nk, Mk, Gc, cond = Nk[:max_scenarios], Mk[:max_scenarios], Gc[:max_scenarios], cond[:max_scenarios]
    return Nk, Mk, Gc, cond


def build_state_array_v4(Nk, Mk, Gc):
    """Concatenate to (N, 361, 81) and log-transform."""
    Gc_e = Gc[:, :, None]
    state = np.concatenate([Nk, Mk, Gc_e], axis=-1)
    return np.log(np.maximum(state, LOG_EPS)).astype(np.float32)


def transform_cond(cond):
    """Log-transform pres, nh3_conc, H2SO4_vmr; leave T, RH linear."""
    out = cond.copy().astype(np.float32)
    for i in LOG_COND_IDX:
        out[..., i] = np.log(np.maximum(out[..., i], LOG_EPS))
    return out


class NormStatsV4:
    def __init__(self, state_mean, state_std, cond_mean, cond_std, delta_std=None):
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        self.cond_mean = torch.as_tensor(cond_mean, dtype=torch.float32)
        self.cond_std = torch.as_tensor(cond_std, dtype=torch.float32)
        self.delta_std = (torch.as_tensor(delta_std, dtype=torch.float32)
                          if delta_std is not None else None)

    def normalize_state(self, s):
        return (s - self.state_mean.to(s.device)) / self.state_std.to(s.device)

    def denormalize_state(self, s):
        return s * self.state_std.to(s.device) + self.state_mean.to(s.device)

    def normalize_cond(self, c):
        return (c - self.cond_mean.to(c.device)) / self.cond_std.to(c.device)

    def save(self, path):
        d = {
            "state_mean": self.state_mean, "state_std": self.state_std,
            "cond_mean": self.cond_mean, "cond_std": self.cond_std,
        }
        if self.delta_std is not None:
            d["delta_std"] = self.delta_std
        torch.save(d, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=True)
        return cls(d["state_mean"], d["state_std"],
                   d["cond_mean"], d["cond_std"], d.get("delta_std"))


def compute_norm_stats_v4(log_states, cond):
    flat = log_states.reshape(-1, N_STATE)
    state_mean = flat.mean(0)
    state_std = np.maximum(flat.std(0), 1e-6)
    cf = cond.reshape(-1, N_COND)
    cond_mean = cf.mean(0)
    cond_std = np.maximum(cf.std(0), 1e-6)
    normalized = (log_states - state_mean) / state_std
    deltas = np.diff(normalized, axis=1)
    delta_std = np.maximum(deltas.reshape(-1, N_STATE).std(0), 1e-6).astype(np.float32)
    print(f"Delta std range: min={delta_std.min():.4f}, max={delta_std.max():.4f}")
    return NormStatsV4(state_mean, state_std, cond_mean, cond_std, delta_std)


class TomasTrajectoryDatasetV4(Dataset):
    """Trajectory chunks with per-step conditioning."""

    def __init__(self, log_states, cond, norm_stats, chunk_len=8, stride=None):
        self.n_scenarios, self.n_steps, _ = log_states.shape
        self.chunk_len = chunk_len
        self.stride = stride if stride is not None else chunk_len
        self.starts = list(range(0, self.n_steps - chunk_len, self.stride))
        self.chunks_per_scenario = len(self.starts)

        ls = torch.as_tensor(log_states, dtype=torch.float32)
        cd = torch.as_tensor(cond, dtype=torch.float32)
        self.states = norm_stats.normalize_state(ls)
        self.cond = norm_stats.normalize_cond(cd)

    def __len__(self):
        return self.n_scenarios * self.chunks_per_scenario

    def __getitem__(self, idx):
        scn = idx // self.chunks_per_scenario
        ch = idx % self.chunks_per_scenario
        s = self.starts[ch]
        state_chunk = self.states[scn, s:s + self.chunk_len + 1]   # (chunk_len+1, 81)
        cond_chunk = self.cond[scn, s:s + self.chunk_len + 1]       # (chunk_len+1, 5)
        return state_chunk, cond_chunk
