"""State / forcing layout, physical floors, log transform and normalization.

State vector (82), all natural logs of per-box quantities, box volume 1e6 cm3 (1 m3):
    [0:40]   Nk   particle number per bin           [particles / box]   floor 1e-20
    [40:80]  Mk   sulfate mass per bin               [kg / box]          floor 1e-30
    [80]     H2SO4 gas                               [kg / box]          floor 1e-30
    [81]     SO2 gas                                 [kg / box]          floor 1e-17 (see below)

Forcing (13 points per 6-hour block, 30-min spacing, t = 0 .. 6 h inclusive), 6 channels:
    0  log pressure            [Pa]
    1  temperature             [K]         (linear)
    2  relative humidity       [0..1]      (linear)
    3  log OH                  [molec / cm3]   floor 100 (see below)
    4  log SO2 source          [kg / box per 30 min]   log(1e-30) when there is no source
    5  log dilution rate       [1 / s]                 log(1e-30) when there is no dilution

Floors. The training corpus never contains zeros, so a zero fed through the log is
many standard deviations outside the training distribution and the network answers
nonsense. Two floors were found the hard way in a coupled climate run and are
load-bearing: SO2 at 1e-17 kg/box (a zero SO2 at 1e-30 is 7 sigma out and produced
1660x the physics' mass) and OH at 100 molec/cm3 (night-time OH of exactly zero is
17 sigma out and produced spurious 1e8 cm-3 nucleation bursts in every polar cell).
Use state_to_log / cond_to_log and these floors are applied for you.

Bin geometry: bin k spans dry particle mass XK0 * 2**k .. 2**(k+1) with
XK0 = pi/6 * (1.7 nm)**3 * 1770 kg/m3; bin-centre diameter d_k = 1.7 nm * 2**((k+0.5)/3).
"""
import numpy as np
import torch

N_BINS, N_COND, N_STATE = 40, 6, 82
BOX = 1e6                              # cm3 per box
NK_FLOOR, MK_FLOOR, GAS_FLOOR, SO2_FLOOR = 1e-20, 1e-30, 1e-30, 1e-17
OH_FLOOR, LOG_ZERO = 100.0, 1e-30
XK0 = np.pi / 6.0 * (1.7e-9) ** 3 * 1770.0
XK = XK0 * 2.0 ** np.arange(N_BINS + 1)                 # bin edges [kg]
M_MID = np.sqrt(XK[:-1] * XK[1:])                       # bin-centre particle mass [kg]
D_EDGE_M = 1.7e-9 * 2.0 ** (np.arange(N_BINS + 1) / 3.0)
D_MID_UM = 1.7e-3 * 2.0 ** ((np.arange(N_BINS) + 0.5) / 3.0)


def state_to_log(Nk, Mk, h2so4_kg, so2_kg):
    """Per-box linear quantities -> (..., 82) log-state with the floors applied. Nk in particles/box."""
    Nk = np.maximum(np.asarray(Nk, dtype=np.float64), NK_FLOOR)
    Mk = np.maximum(np.asarray(Mk, dtype=np.float64), MK_FLOOR)
    g4 = np.maximum(np.asarray(h2so4_kg, dtype=np.float64), GAS_FLOOR)[..., None]
    g2 = np.maximum(np.asarray(so2_kg, dtype=np.float64), SO2_FLOOR)[..., None]
    return np.log(np.concatenate([Nk, Mk, g4, g2], -1)).astype(np.float32)


def log_to_state(s):
    """(..., 82) log-state -> dict of linear per-box quantities (exp clamped at e^60)."""
    s = np.minimum(np.asarray(s, dtype=np.float64), 60.0)
    return {"Nk": np.exp(s[..., :N_BINS]), "Mk": np.exp(s[..., N_BINS:2 * N_BINS]),
            "h2so4_kg": np.exp(s[..., 80]), "so2_kg": np.exp(s[..., 81])}


def cond_to_log(pres_pa, temp_k, rh, oh_cm3, so2_src_kg_per_30min=0.0, kdil_per_s=0.0):
    """Forcing arrays of shape (..., 13) [scalars broadcast] -> (..., 13, 6) forcing in the training convention."""
    pres_pa, temp_k, rh, oh_cm3, src, kdil = np.broadcast_arrays(
        *[np.asarray(a, dtype=np.float64) for a in (pres_pa, temp_k, rh, oh_cm3, so2_src_kg_per_30min, kdil_per_s)])
    c = np.stack([np.log(pres_pa), temp_k, np.clip(rh, 1e-3, 1.0),
                  np.log(np.maximum(oh_cm3, OH_FLOOR)),
                  np.log(np.maximum(src, LOG_ZERO)), np.log(np.maximum(kdil, LOG_ZERO))], -1)
    return c.astype(np.float32)


class NormStats:
    """Per-channel mean / std of the log-state and forcing over the training corpus (checkpoints/norm_stats.npz).
    Both shipped checkpoints use the same file. Never recompute it: the weights are tied to these numbers."""

    def __init__(self, sm, ss, cm, cs):
        self.state_mean = torch.as_tensor(sm, dtype=torch.float32)
        self.state_std = torch.as_tensor(ss, dtype=torch.float32)
        self.cond_mean = torch.as_tensor(cm, dtype=torch.float32)
        self.cond_std = torch.as_tensor(cs, dtype=torch.float32)

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
