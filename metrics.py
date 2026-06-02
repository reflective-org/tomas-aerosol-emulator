"""Physical-space metrics for the TOMAS emulator.

These helpers convert the model's log-state output back to physical
concentrations and reproduce the train/val split used during training, so
evaluation runs on exactly the held-out scenarios. Extracted from the original
``evaluate_v4.py`` so the eval path has no dependency on the full research tree.
"""

import numpy as np
import torch

from dataset_v4 import N_BINS


def log_state_to_physical(log_state_denorm):
    """Map a denormalized log-state vector back to physical Nk, Mk, Gc.

    log_state_denorm: (..., 81) = [log Nk(40), log Mk_SO4(40), log Gc_H2SO4(1)].
    Returns (Nk, Mk, Gc) in physical units.
    """
    state = torch.exp(log_state_denorm)
    Nk = state[..., :N_BINS]
    Mk = state[..., N_BINS:2 * N_BINS]
    Gc = state[..., 2 * N_BINS]
    return Nk, Mk, Gc


def compute_diameter_nm(Mk, Nk, density=1770.0):
    """Per-bin volume-equivalent diameter (nm) from mass and number."""
    safe_Nk = torch.clamp(Nk, min=1e-20)
    vol = Mk / (safe_Nk * density)
    vol = torch.clamp(vol, min=1e-40)
    return (6.0 * vol / np.pi) ** (1.0 / 3.0) * 1e9


def number_weighted_mean_diameter(Nk, diam_nm):
    safe_Ntot = torch.clamp(Nk.sum(dim=-1, keepdim=True), min=1e-20)
    return (Nk * diam_nm).sum(dim=-1) / safe_Ntot.squeeze(-1)


def geometric_mean_diameter(Nk, diam_nm):
    safe_diam = torch.clamp(diam_nm, min=1e-6)
    log_diam = torch.log(safe_diam)
    safe_Ntot = torch.clamp(Nk.sum(dim=-1, keepdim=True), min=1e-20)
    mean_log_d = (Nk * log_diam).sum(dim=-1) / safe_Ntot.squeeze(-1)
    return torch.exp(mean_log_d)


def geometric_std_dev(Nk, diam_nm, gmd):
    safe_diam = torch.clamp(diam_nm, min=1e-6)
    log_diam = torch.log(safe_diam)
    log_gmd = torch.log(torch.clamp(gmd, min=1e-6)).unsqueeze(-1)
    safe_Ntot = torch.clamp(Nk.sum(dim=-1, keepdim=True), min=1e-20)
    var_log = (Nk * (log_diam - log_gmd) ** 2).sum(dim=-1) / safe_Ntot.squeeze(-1)
    return torch.exp(torch.sqrt(torch.clamp(var_log, min=0.0)))


def split_train_val(n_total, val_frac, seed):
    """Reproduce the trainer's deterministic train/val split exactly.

    Eval must use the SAME (val_frac, seed) the model was trained with, or it
    will leak training scenarios into the reported numbers.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_frac))
    return perm[n_val:], perm[:n_val]
