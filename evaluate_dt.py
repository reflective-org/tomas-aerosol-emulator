"""Evaluate the Δt-conditioned ("time-skip") emulator checkpoint.

Rolls the model out to 6 h at several step sizes Δt and reports pooled per-bin
relative error (physical space, floor-masked) at 1 h / 3 h / 6 h. The
conservation projection (ConservativeMultiHeadMLPV2) is applied during the
rollout iff the checkpoint was trained with it (proj_n_iter > 0), so eval
matches training.

Usage:
    DATA_DIR=/path/to/v4_repr_1M python evaluate_dt.py
Optional env: RUN_DIR (default checkpoints/dt_champion), CKPT (best_model.pt),
N_SCEN (500), ROLL_DTS (comma list, default "1,5,30").
"""
import os
import sys

import numpy as np
import torch

from dataset_v4 import (load_v4_batches, build_state_array_v4, transform_cond,
                        NormStatsV4, N_BINS, LOG_EPS)
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2
from metrics import log_state_to_physical, split_train_val
from dt_features import fourier_dt_feat, N_COND_BASE, N_COND_TOTAL

REPO = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.environ.get("RUN_DIR", os.path.join(REPO, "checkpoints", "dt_champion"))
DATA_DIR = os.environ.get("DATA_DIR", "data/v4_repr_1M")
N_SCEN = int(os.environ.get("N_SCEN", 500))
STATE_CLAMP = 20.0
ROLL_DTS = [int(d) for d in os.environ.get("ROLL_DTS", "1,5,30").split(",")]
HORIZON_MIN = [60, 180, 360]
FLOOR_TOL = 1.5
EMPTY_FRAC = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"RUN_DIR: {RUN_DIR}")
Nk, Mk, Gc, cond = load_v4_batches(DATA_DIR, max_scenarios=20000, verbose=False)
_, val_idx = split_train_val(Nk.shape[0], val_frac=0.05, seed=0)
eval_idx = val_idx[:N_SCEN]
log_states = build_state_array_v4(Nk[eval_idx], Mk[eval_idx], Gc[eval_idx])
# The checkpoint was trained on the first 5 forcing features; drop any extras
# (dataset_v4 later grew a 6th, fion) to match its norm_stats.
cond_t = transform_cond(cond[eval_idx])[..., :N_COND_BASE]

CKPT = os.environ.get("CKPT", "best_model.pt")
norm_stats = NormStatsV4.load(f"{RUN_DIR}/norm_stats.pt")
print(f"CKPT: {CKPT}")
ckpt = torch.load(f"{RUN_DIR}/{CKPT}", weights_only=False, map_location=device)
ma = ckpt["args"]
if ma.get("model_type", "mlp") != "mlp":
    sys.exit(f"This release ships the 'mlp' architecture only; checkpoint has "
             f"model_type={ma.get('model_type')!r}")
base = TomasMultiHeadMLP(
    hidden_dim=ma["hidden_dim"], n_backbone_layers=ma.get("n_layers", 5),
    head_hidden_dim=ma.get("head_hidden_dim", 256),
    nuc_head_hidden_dim=ma.get("nuc_head_hidden_dim", 512),
    dropout=ma.get("dropout", 0.0), n_cond=N_COND_TOTAL,
    core_end_excl=ma.get("core_end_excl", 26)).to(device)
base.load_state_dict(ckpt["model_state_dict"]); base.eval()

_ep = ckpt.get("epoch", "ema"); _v = ckpt.get("val_mse")
_vs = f"{_v:.6f}" if isinstance(_v, (int, float)) else "n/a"
proj_n_iter = int(ma.get("proj_n_iter", 0) or 0)
if proj_n_iter > 0:
    lfc = ma.get("log_factor_clamp", 2.302585)
    lfc = lfc if lfc and lfc > 0 else None
    net = ConservativeMultiHeadMLPV2(base, norm_stats, log_factor_clamp=lfc,
                                     n_iter=proj_n_iter).to(device)
    net.eval()
    print(f"Model: epoch-{_ep} val_mse={_vs}  "
          f"CONSERVATION ON (proj_n_iter={proj_n_iter})")
else:
    net = base
    print(f"Model: epoch-{_ep} val_mse={_vs}  conservation OFF")

states_t = torch.as_tensor(log_states, dtype=torch.float32)
cond_t_t = torch.as_tensor(cond_t, dtype=torch.float32)
states_norm = norm_stats.normalize_state(states_t).to(device)
cond_norm = norm_stats.normalize_cond(cond_t_t).to(device)
n_steps = states_norm.shape[1]


def rollout(dt_min):
    feat = fourier_dt_feat(torch.tensor([float(dt_min)], device=device)).expand(states_norm.shape[0], -1)
    cur = states_norm[:, 0, :]
    t = 0
    preds = {}
    with torch.no_grad():
        while t < 360:
            step = min(dt_min, 360 - t)
            c = cond_norm[:, min(t, n_steps - 1), :]
            x = torch.cat([cur, c, feat], dim=-1)
            cur = net(x).clamp(-STATE_CLAMP, STATE_CLAMP)
            t += step
            if t in HORIZON_MIN:
                preds[t] = cur.clone()
    return preds


def summary(pred_norm, step):
    pred_log = norm_stats.denormalize_state(pred_norm).cpu()
    gt_log = norm_stats.denormalize_state(states_norm[:, step, :]).cpu()
    pN, pM, _ = log_state_to_physical(pred_log)
    gN, gM, _ = log_state_to_physical(gt_log)
    pN = pN.numpy(); gN = gN.numpy(); pM = pM.numpy(); gM = gM.numpy()
    def agg(pred, gt):
        rels = []
        for b in range(N_BINS):
            g = gt[:, b]; p = pred[:, b]
            pos = g[g > 10 * LOG_EPS]
            if pos.size < 10: continue
            fl = pos.min(); nf = g <= FLOOR_TOL * fl
            if nf.mean() > EMPTY_FRAC: continue
            keep = ~nf
            if keep.sum() < 10: continue
            r = np.abs(p[keep] - g[keep]) / g[keep]
            r = r[np.isfinite(r)]
            if r.size: rels.append(r)
        ar = np.concatenate(rels)
        return np.median(ar), np.percentile(ar, 90), np.percentile(ar, 99)
    return agg(pN, gN), agg(pM, gM)


print(f"\n{'Δt':>5s} {'horizon':>8s} | {'Nk_p50':>7s} {'Nk_p90':>7s} {'Nk_p99':>8s} | "
      f"{'Mk_p50':>7s} {'Mk_p90':>7s} {'Mk_p99':>8s}")
for dt in ROLL_DTS:
    if 360 % dt != 0: continue
    preds = rollout(dt)
    n_steps_6h = 360 // dt
    for h in HORIZON_MIN:
        if h not in preds: continue
        nk, mk = summary(preds[h], h)
        print(f"{dt:>4d}m {h:>6d}m | {nk[0]:>7.3f} {nk[1]:>7.3f} {nk[2]:>8.2f} | "
              f"{mk[0]:>7.3f} {mk[1]:>7.3f} {mk[2]:>8.2f}   ({n_steps_6h} steps to 6h)")
print("\nDone.")
