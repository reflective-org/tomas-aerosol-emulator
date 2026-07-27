"""Is cons1's 6h regression a domain shift, or a t=0 initial-condition effect?

cons1 beats aug5 by ~5x on the 24h CESM benchmark but loses on the canonical
v4_repr_1M 6h eval (Nk p90 0.236 -> 0.327). Two competing explanations:

  (a) DOMAIN SHIFT: cons1 adapted to the v4_long24h distribution (2/3 of its
      loss) and away from v4_repr_1M.
  (b) t=0 EFFECT: cons1 trained almost entirely on TOMAS-consistent interior
      states, and the canonical eval starts from the bin-mass-INCONSISTENT t=0
      state, so it only lost skill on that unusual input.

Discriminator: per-horizon error on the canonical set. Under (b) the deficit is
concentrated in the early horizons and shrinks as the t=0 transient washes out;
under (a) it is roughly uniform across horizons.
"""
import sys, os
import numpy as np, torch
torch.set_num_threads(16)
from dataset_v4 import load_v4_batches, build_state_array_v4, transform_cond, NormStatsV4, LOG_EPS
from model_axial_traj import AxialTrajOperator, DT_MIN

WS, MAXS, PRV, N_EVAL = 7, 140000, 4000, 800
G = list(range(DT_MIN, 361, DT_MIN)); TP = [0]+G

Nk, Mk, Gc, cond = load_v4_batches(os.environ.get("DATA_DIR", "data/v4_repr_1M"), max_scenarios=MAXS,
                                   rank=0, world_size=WS, verbose=False)
sel = np.random.default_rng(0).permutation(Nk.shape[0])[:PRV][:N_EVAL]
ls = build_state_array_v4(Nk[sel], Mk[sel], Gc[sel]); ct = transform_cond(cond[sel])
del Nk, Mk, Gc, cond
gt = np.exp(ls.astype(np.float64))


def agg(pp, gg):
    rl = []
    for b in range(40):
        g = gg[:, b]; p = pp[:, b]; pos = g[g > 10*LOG_EPS]
        if pos.size < 10: continue
        fl = pos.min(); nf = g <= 1.5*fl
        if nf.mean() > 0.5: continue
        kk = ~nf; r = np.abs(p[kk]-g[kk])/g[kk]; r = r[np.isfinite(r)]
        if r.size: rl.append(r)
    a = np.concatenate(rl); return float(np.median(a)), float(np.percentile(a, 90))


def run(run_name):
    rd = os.path.join(os.environ.get("CKPT_ROOT", "checkpoints"), run_name)
    ck = torch.load(f"{rd}/best_model.pt", weights_only=False, map_location="cpu"); ma = ck["args"]
    nc = ck["model_state_dict"]["cond_in.weight"].shape[1]
    m = AxialTrajOperator(d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"],
                          dim_ff=ma["dim_ff"], dropout=0.0, ff_experts=ma.get("ff_experts", 1),
                          n_out=12, n_cond=nc)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    ns = NormStatsV4.load(f"{rd}/norm_stats.pt")
    s0 = ns.normalize_state(torch.as_tensor(ls[:, 0, :], dtype=torch.float32))
    c = ns.normalize_cond(torch.as_tensor(ct[:, TP, :], dtype=torch.float32))
    with torch.no_grad():
        p = torch.cat([m(s0[i:i+100], c[i:i+100]) for i in range(0, N_EVAL, 100)])
    pl = ns.denormalize_state(p.reshape(-1, 81)).reshape(N_EVAL, 12, 81).numpy().astype(np.float64)
    return np.exp(pl[:, :, :40])


pa = run(os.environ.get("RUN_A", "axial_aug5"))
pc = run(os.environ.get("RUN_B", "axial_cons2"))
print(f"{'h(min)':>6s} | {'aug5 p50':>8s} {'aug5 p90':>8s} | {'cons1 p50':>9s} {'cons1 p90':>9s} | {'p90 ratio':>9s}")
for k, h in enumerate(G):
    a = agg(pa[:, k], gt[:, h, :40]); c = agg(pc[:, k], gt[:, h, :40])
    print(f"{h:6d} | {a[0]:8.3f} {a[1]:8.3f} | {c[0]:9.3f} {c[1]:9.3f} | {c[1]/max(a[1],1e-9):9.2f}")
print("\n(b) t=0 effect  -> ratio LARGE at h=30 and falling toward 1 by h=360")
print("(a) domain shift -> ratio roughly FLAT across horizons")
