"""Held-out evaluation of the axial non-AR trajectory operator (one forward -> 12 states).

Reproduces the pooled per-bin relative-error protocol used during development
(same floor masking as evaluate.py). Configuration via environment variables:

    RUN_DIR=checkpoints/axial_aug5 DATA_DIR=data/v4_repr_1M python evaluate_axial.py

WS/MAXS/PRV/SEED reconstruct the rank-0 validation shard the model never
trained on; keep the defaults to match the reported numbers.
"""
import os
import numpy as np, torch
from dataset_v4 import (load_v4_batches, build_state_array_v4, transform_cond, NormStatsV4, N_BINS, LOG_EPS)
from model_axial_traj import AxialTrajOperator, N_OUT, DT_MIN
from metrics import log_state_to_physical
from burst_feats import dunne_logJ

RUN_DIR = os.environ.get("RUN_DIR", "checkpoints/axial_aug5")
CKPT = os.environ.get("CKPT", "best_model.pt")
DATA_DIR = os.environ.get("DATA_DIR", "data/v4_repr_1M")
WS = int(os.environ.get("WS", 7)); MAXS = int(os.environ.get("MAXS", 140000))
PRV = int(os.environ.get("PRV", 4000)); SEED = int(os.environ.get("SEED", 0))
N_EVAL = 800; FT = 1.5; EF = 0.5; HMIN = [60, 180, 360]
dev = os.environ.get("DEVICE", "cuda")

print(f"RUN_DIR {RUN_DIR} CKPT {CKPT} (HELD-OUT rank0 val_sel WS={WS} MAXS={MAXS} PRV={PRV})")
Nk, Mk, Gc, cond = load_v4_batches(DATA_DIR, max_scenarios=MAXS, rank=0, world_size=WS, verbose=False)
rng = np.random.default_rng(SEED + 0); sel = rng.permutation(Nk.shape[0])[:PRV][:N_EVAL]
ls = build_state_array_v4(Nk[sel], Mk[sel], Gc[sel]); cond_raw = cond[sel].copy(); ct = transform_cond(cond[sel])
ns = NormStatsV4.load(f"{RUN_DIR}/norm_stats.pt")
ct = ct[..., :ns.cond_mean.shape[-1]]   # older checkpoints use fewer cond channels
ck = torch.load(f"{RUN_DIR}/{CKPT}", weights_only=False, map_location=dev); ma = ck["args"]
GRID = [int(x) for x in ma["time_grid"].split(",")] if ma.get("time_grid") else list(range(DT_MIN, 361, DT_MIN))
TP = [0] + GRID
n_cond = ck["model_state_dict"]["cond_in.weight"].shape[1]   # infer from weights (5, 6 with fion, +1 with burst feats)
m = AxialTrajOperator(d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"],
                      dim_ff=ma["dim_ff"], dropout=0.0, ff_experts=ma.get("ff_experts", 1),
                      n_out=len(GRID), n_cond=n_cond).to(dev)
m.load_state_dict(ck["model_state_dict"]); m.eval()
print(f"epoch={ck.get('epoch')} val={ck.get('val_mse'):.5f} params={sum(p.numel() for p in m.parameters()):,}")
s0 = ns.normalize_state(torch.as_tensor(ls[:, 0, :], dtype=torch.float32)).to(dev)
c13 = ns.normalize_cond(torch.as_tensor(ct[:, TP, :], dtype=torch.float32))
if ma.get("burst_feats"):
    lj = (dunne_logJ(cond_raw[:, TP, :]) - ma["lj_mean"]) / ma["lj_std"]
    c13 = torch.cat([c13, torch.as_tensor(lj[..., None], dtype=torch.float32)], dim=-1)
c13 = c13.to(dev)
with torch.no_grad(): pred = m(s0, c13)
gt = torch.as_tensor(ls, dtype=torch.float32)
def agg(pp, gg):
    rels = []
    for b in range(N_BINS):
        g = gg[:, b]; p = pp[:, b]; pos = g[g > 10 * LOG_EPS]
        if pos.size < 10: continue
        fl = pos.min(); nf = g <= FT * fl
        if nf.mean() > EF: continue
        k = ~nf; r = np.abs(p[k] - g[k]) / g[k]; r = r[np.isfinite(r)]
        if r.size: rels.append(r)
    ar = np.concatenate(rels); return np.median(ar), np.percentile(ar, 90), np.percentile(ar, 99)
print(f"\n=== AXIAL non-AR operator (held-out) ===")
print(f"{'h':>5s} | {'Nk_p50':>7s} {'Nk_p90':>7s} {'Nk_p99':>8s} | {'Mk_p50':>7s} {'Mk_p90':>7s} {'Mk_p99':>8s}")
for h in HMIN:
    k = GRID.index(h); pl = ns.denormalize_state(pred[:, k, :].cpu()); gl = gt[:, h, :]
    pN, pM, _ = log_state_to_physical(pl); gN, gM, _ = log_state_to_physical(gl)
    nk = agg(pN.numpy(), gN.numpy()); mk = agg(pM.numpy(), gM.numpy())
    print(f"{h:5d} | {nk[0]:7.3f} {nk[1]:7.3f} {nk[2]:8.2f} | {mk[0]:7.3f} {mk[1]:7.3f} {mk[2]:8.2f}")
