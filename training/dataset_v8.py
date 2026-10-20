"""v8-family dataset for emulator training, on the 30-minute grid.

State (82) = log(max([Nk(40), Mk_SO4(40), Gc_H2SO4, Gc_SO2], LOG_EPS)) — SO2 is
prognostic in v8, so both gases are state. Grid: every 30 min (6h sets: states
[::30] of 361 -> 13; 2wk sets: native 673). Conditioning (6, per grid point):
[log pres, T, RH, log OH, log so2_src, log kdil] — so2_src is the continuous-
injection source [kg/box per 30 min] (zero elsewhere); kdil the dilution rate.

Split roles follow /data/tomas-jax-output/emulator_v1/split_manifest.json:
test starts {0, 704, 1379}, val start {381}, lon bands 72..95 (test) /
192..215 (val); v10_ideal: batch b%20==0 test, ==1 val.
"""
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

N_BINS = 40
N_STATE = 82
N_COND = 6
LOG_EPS = 1e-30
TEST_STARTS = {0, 704, 1379}
VAL_STARTS = {381}
TEST_LON = (72, 96)
VAL_LON = (192, 216)
KB = 1.380649e-23


def _roles(z, fname):
    """Per-scenario role array ('train'/'val'/'test') for one npz."""
    B = z["Nk"].shape[0]
    if "t_idx" not in z.files:                       # v10_ideal
        b = int(str(fname).split("_b")[-1].split(".")[0])
        r = {0: "test", 1: "val"}.get(b % 20, "train")
        return np.full(B, r, dtype=object)
    g = int(z["t_idx"])
    if g in TEST_STARTS:
        return np.full(B, "test", dtype=object)
    if g in VAL_STARTS:
        return np.full(B, "val", dtype=object)
    lon = z["lon_idx"]
    out = np.full(B, "train", dtype=object)
    out[(lon >= TEST_LON[0]) & (lon < TEST_LON[1])] = "test"
    out[(lon >= VAL_LON[0]) & (lon < VAL_LON[1])] = "val"
    return out


def _cond_series(z, T_grid):
    """(B, T_grid, N_COND) linear-space conditioning at the 30-min grid."""
    B = z["Nk"].shape[0]
    c = np.zeros((B, T_grid, N_COND), dtype=np.float64)
    if "T_halfhourly" in z.files:                    # CESM-driven sets
        hh = z["T_halfhourly"].shape[0]
        m = min(hh, T_grid)
        pres = (z["pres_halfhourly"].T if "pres_halfhourly" in z.files
                else np.repeat(z["pres"][:, None], hh, 1))     # (B, hh)
        c[:, :m, 0] = pres[:, :m]
        c[:, :m, 1] = z["T_halfhourly"].T[:, :m]
        c[:, :m, 2] = z["RH_halfhourly"].T[:, :m]
        if "OH_molec_cm3_halfhourly" in z.files:
            c[:, :m, 3] = z["OH_molec_cm3_halfhourly"].T[:, :m]
        if m < T_grid:                               # pad with last value
            c[:, m:, :4] = c[:, m - 1:m, :4]
    else:                                            # v10_ideal synthetic
        c[:, :, 0] = z["pres"][:, None]
        c[:, :, 1] = z["temp"][:, None]
        c[:, :, 2] = z["rh"][:, None]
        t_h = (np.arange(T_grid) * 0.5)[None, :]
        diur = np.maximum(0.0, np.cos(2 * np.pi *
                                      (t_h - z["oh_phase_h"][:, None]) / 24.0))
        c[:, :, 3] = np.where(z["oh_diurnal"][:, None], diur, 1.0) * \
            z["oh_peak"][:, None]
    # continuous-injection source [kg/box per 30-min step]
    if "inj_duration_h" in z.files and (z["inj_duration_h"] > 0).any():
        P0 = c[:, 0, 0]; T0 = c[:, 0, 1]
        n_air = P0 / (KB * T0) * 1e-6
        total_kg = (z["inj_so2_vmr"].astype(np.float64) * n_air * 1e6
                    * (64.0 / 1000.0) / 6.02214076e23)
        dur_steps = np.maximum(z["inj_duration_h"].astype(np.float64) * 2, 1.0)
        steps = np.arange(T_grid)[None, :]
        c[:, :, 4] = np.where(steps < dur_steps[:, None],
                              (total_kg / dur_steps)[:, None], 0.0)
    if "kdil" in z.files:
        c[:, :, 5] = z["kdil"].astype(np.float64)[:, None]
    return c


def transform_cond_v8(c):
    """log on pres(0), OH(3), so2_src(4), kdil(5); T, RH linear."""
    out = c.astype(np.float32).copy()
    for i in (0, 3, 4, 5):
        out[..., i] = np.log(np.maximum(c[..., i], LOG_EPS)).astype(np.float32)
    return out


def load_v8_set(root, role="train", max_scenarios=None, grid_stride=None,
                verbose=True, seed=0, rank_stripe=None):
    """Load one dataset root at the 30-min grid.

    Returns states_log (N, T, 82) float32, cond (N, T, 6) float32 (transformed).
    grid_stride: for 6h sets states are (B,361,·) -> stride 30; 2wk sets are
    native 30-min (673) -> stride 1. Auto-detected per file if None.
    """
    # fast path: consolidated cache built by build_cache_v8.py
    cache = Path("/data/tomas-jax-output/emulator_v1/cache")
    name = Path(root).name
    if (cache / f"{name}.done").exists():
        Sm = np.load(cache / f"{name}_{role}_S.npy", mmap_mode="r")
        Cm = np.load(cache / f"{name}_{role}_C.npy", mmap_mode="r")
        if rank_stripe is not None:
            r, w = rank_stripe
            Sm, Cm = Sm[r::w], Cm[r::w]
        if max_scenarios:
            Sm, Cm = Sm[:max_scenarios], Cm[:max_scenarios]
        S, C = np.array(Sm), np.array(Cm)      # materialize to RAM
        if verbose:
            print(f"  {name}[{role}]: cache {S.shape[0]:,} x T={S.shape[1]}",
                  flush=True)
        return S, C

    files = sorted(Path(root).glob("t*/batch_*.npz")) or \
        sorted(Path(root).glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(root)
    # deterministic shuffle BEFORE striping so ranks see disjoint file sets
    np.random.default_rng(12345).shuffle(files)
    if rank_stripe is not None:
        r, w = rank_stripe
        files = files[r::w]
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    S_l, C_l, total = [], [], 0
    for f in files:
        z = np.load(f)
        roles = _roles(z, f.name)
        keep = roles == role
        if not keep.any():
            continue
        Nk = z["Nk"]; Mk = z["Mk_SO4"]
        stride = grid_stride or (30 if Nk.shape[1] == 361 else 1)
        Nk = Nk[keep][:, ::stride]; Mk = Mk[keep][:, ::stride]
        g4 = z["Gc_H2SO4"][keep][:, ::stride]
        g2 = (z["Gc_SO2"][keep][:, ::stride] if "Gc_SO2" in z.files
              else np.zeros_like(g4))
        T_grid = Nk.shape[1]
        state = np.concatenate(
            [Nk, Mk, g4[:, :, None], g2[:, :, None]], axis=-1)
        S_l.append(np.log(np.maximum(state, LOG_EPS)).astype(np.float32))
        c = _cond_series(z, T_grid)
        C_l.append(transform_cond_v8(c[keep] if c.shape[0] != keep.sum() else c))
        total += int(keep.sum())
        if verbose and len(S_l) % 25 == 0:
            print(f"  {Path(root).name}[{role}]: {total:,} scenarios", flush=True)
        if max_scenarios and total >= max_scenarios:
            break
    S = np.concatenate(S_l, 0)[:max_scenarios]
    C = np.concatenate(C_l, 0)[:max_scenarios]
    if verbose:
        print(f"  {Path(root).name}[{role}]: DONE {S.shape[0]:,} x T={S.shape[1]}",
              flush=True)
    return S, C


class NormStatsV8:
    def __init__(self, sm, ss, cm, cs):
        self.state_mean = torch.as_tensor(sm, dtype=torch.float32)
        self.state_std = torch.as_tensor(ss, dtype=torch.float32)
        self.cond_mean = torch.as_tensor(cm, dtype=torch.float32)
        self.cond_std = torch.as_tensor(cs, dtype=torch.float32)

    @classmethod
    def compute(cls, S, C):
        f = S.reshape(-1, N_STATE)
        cf = C.reshape(-1, N_COND)
        return cls(f.mean(0), np.maximum(f.std(0), 1e-6),
                   cf.mean(0), np.maximum(cf.std(0), 1e-6))

    def save(self, path):
        np.savez(path, sm=self.state_mean.numpy(), ss=self.state_std.numpy(),
                 cm=self.cond_mean.numpy(), cs=self.cond_std.numpy())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["sm"], d["ss"], d["cm"], d["cs"])

    def norm_state(self, s):
        return (s - self.state_mean.to(s.device)) / self.state_std.to(s.device)

    def denorm_state(self, s):
        return s * self.state_std.to(s.device) + self.state_mean.to(s.device)

    def norm_cond(self, c):
        return (c - self.cond_mean.to(c.device)) / self.cond_std.to(c.device)


class WindowDataset(Dataset):
    """AR training windows: (state_t, cond[t+1..t+k], states[t+1..t+k]) at a
    step of `dt_steps` grid points (dt_steps=1 -> 30 min)."""

    def __init__(self, S, C, norm, k, dt_steps=1, samples_per_scen=2, seed=0):
        self.S = torch.from_numpy(S); self.C = torch.from_numpy(C)
        self.norm = norm; self.k = k; self.d = dt_steps
        T = S.shape[1]
        self.max_t0 = T - 1 - k * dt_steps
        self.empty = self.max_t0 < 0     # set too short for this k: contributes nothing
        self.spc = samples_per_scen
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return 0 if self.empty else self.S.shape[0] * self.spc

    def __getitem__(self, i):
        s = i // self.spc
        t0 = int(self.rng.integers(0, self.max_t0 + 1)) if self.max_t0 > 0 else 0
        idx = t0 + self.d * np.arange(self.k + 1)
        st = self.norm.norm_state(self.S[s][idx])          # (k+1, 82)
        cd = self.norm.norm_cond(self.C[s][idx])           # (k+1, 6)
        return st[0], cd, st[1:]
