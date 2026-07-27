"""Chained 24h evaluation of the non-AR axial emulator against real TOMAS truth.

The model maps (state0, 6h forcing) -> 12 states in one forward. To go past 6h it
must be CHAINED: the 6h output becomes the next window's initial condition. Until
now there was no ground truth past 6h, so chained accuracy had never been measured
(chain_axial_7day.py only checked for blow-up). gen_v4_longhorizon.py provides it.

Two chaining protocols, run side by side:
  self   : window w starts from the model's own w-1 output  (deployment)
  gtstart: window w starts from TOMAS truth at 6h*w          (no exposure bias)
The gap between them isolates window-level exposure bias -- i.e. how much of the
long-range error is compounding versus per-window skill.

Usage:
  python evaluate_chain.py --run checkpoints/axial_aug5 --mode none --tag aug5
  python evaluate_chain.py --run checkpoints/axial_cons2 --mode mono --tag cons2
"""
import os, json, argparse
import numpy as np, torch
from dataset_v4 import NormStatsV4, transform_cond, LOG_EPS
from dataset_long import load_long_batches
from model_axial_traj import AxialTrajOperator
from model_conserved_axial import ConservativeAxialTraj

DT = 30
STEPS_PER_WIN = 12
WIN_MIN = DT * STEPS_PER_WIN          # 360


def agg(pp, gg, ft=1.5, ef=0.5):
    """per-bin rel-err pooled over non-saturated bins (canonical protocol)."""
    rels = []
    for b in range(pp.shape[1]):
        g = gg[:, b]; p = pp[:, b]; pos = g[g > 10 * LOG_EPS]
        if pos.size < 10: continue
        fl = pos.min(); nf = g <= ft * fl
        if nf.mean() > ef: continue
        kk = ~nf; r = np.abs(p[kk] - g[kk]) / g[kk]; r = r[np.isfinite(r)]
        if r.size: rels.append(r)
    if not rels: return (np.nan,)*3
    a = np.concatenate(rels)
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.percentile(a, 99))


def tot(pp, gg):
    r = np.abs(pp.sum(-1) - gg.sum(-1)) / np.maximum(gg.sum(-1), 1e-300)
    r = r[np.isfinite(r)]
    return float(np.median(r)), float(np.percentile(r, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/axial_cons2")
    ap.add_argument("--ckpt", default="best_model.pt")
    ap.add_argument("--data", default="data/v4_long24h")
    ap.add_argument("--mode", default="none", choices=["none", "geom", "mono", "equality"])
    ap.add_argument("--n-eval", type=int, default=800)
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="eval on the FIRST this-fraction of scenarios; training must use the rest")
    ap.add_argument("--start-min", type=int, default=60,
                    help="minute the chain starts from. t=0 states are bin-mass-INCONSISTENT "
                         "(Mk/Nk off the TOMAS grid in ~90%% of populated bins; TOMAS rebins at "
                         "the first step) and are a separate IC-construction problem, so chaining "
                         "is measured from a TOMAS-consistent state by default.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/chain")
    a = ap.parse_args()
    tag = a.tag or f"{Pathsafe(a.run)}_{a.mode}"
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(8, os.cpu_count() // 3))

    print("loading 24h truth ...", flush=True)
    Nk, Mk, Gc, cond, n_steps = load_long_batches(a.data, verbose=True)
    n_hold = max(a.n_eval, int(len(Nk) * a.holdout_frac))
    sel = np.arange(min(a.n_eval, n_hold))            # deterministic held-out prefix
    Nk, Mk, Gc, cond = Nk[sel], Mk[sel], Gc[sel], cond[sel]
    N = len(Nk)
    t0 = a.start_min
    n_win = (n_steps - t0) // WIN_MIN
    print(f"  {N} scenarios, start t={t0} min, {n_win} chained windows "
          f"-> {t0 + n_win*WIN_MIN} min", flush=True)

    state = np.log(np.maximum(np.concatenate([Nk, Mk, Gc[..., None]], -1), LOG_EPS)).astype(np.float32)
    ct = transform_cond(cond)
    del Nk, Mk, Gc, cond

    run_dir = a.run
    ns = NormStatsV4.load(f"{run_dir}/norm_stats.pt")
    ck = torch.load(f"{run_dir}/{a.ckpt}", weights_only=False, map_location=a.device)
    ma = ck["args"]
    n_cond = ck["model_state_dict"]["cond_in.weight"].shape[1]
    m = AxialTrajOperator(d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"],
                          dim_ff=ma["dim_ff"], dropout=0.0, ff_experts=ma.get("ff_experts", 1),
                          n_out=STEPS_PER_WIN, n_cond=n_cond).to(a.device)
    sd = ck["model_state_dict"]
    sd = {k[len("base."):] if k.startswith("base.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False); m.eval()
    net = m if a.mode == "none" else ConservativeAxialTraj(m, ns, mode=a.mode).to(a.device).eval()
    print(f"  model epoch={ck.get('epoch')} n_cond={n_cond} constraint={a.mode}", flush=True)

    # per-window cond anchors on the 30-min grid
    def cond_win(w):
        b = t0 + w * WIN_MIN
        idx = np.arange(b, b + WIN_MIN + 1, DT)
        return ns.normalize_cond(torch.as_tensor(ct[:, idx, :], dtype=torch.float32)).to(a.device)

    gt_norm = ns.normalize_state(torch.as_tensor(state, dtype=torch.float32))
    results = {}
    for proto in ("self", "gtstart"):
        pred_log = np.empty((N, n_win * STEPS_PER_WIN, 81), np.float32)
        cur = gt_norm[:, t0, :].to(a.device)
        with torch.no_grad():
            for w in range(n_win):
                if proto == "gtstart":
                    cur = gt_norm[:, t0 + w * WIN_MIN, :].to(a.device)
                c = cond_win(w)
                out = torch.cat([net(cur[i:i+100], c[i:i+100]) for i in range(0, N, 100)])
                pred_log[:, w*STEPS_PER_WIN:(w+1)*STEPS_PER_WIN] = \
                    ns.denormalize_state(out.reshape(-1, 81)).reshape(N, STEPS_PER_WIN, 81).cpu().numpy()
                cur = out[:, -1, :]
        results[proto] = pred_log
        print(f"  {proto}: done", flush=True)

    # ---- metrics vs truth at every 30-min horizon ----
    horizons = [(w*STEPS_PER_WIN + i, t0 + (w*WIN_MIN) + (i+1)*DT)
                for w in range(n_win) for i in range(STEPS_PER_WIN)]
    gt_phys = np.exp(state.astype(np.float64))
    metrics = {"run": a.run, "mode": a.mode, "n_eval": N, "n_win": n_win,
               "start_min": t0, "per_horizon": {}}
    for proto, pl in results.items():
        p = np.exp(pl.astype(np.float64))
        rec = {}
        for j, tmin in horizons:
            pN, pM = p[:, j, :40], p[:, j, 40:80]
            gN, gM = gt_phys[:, tmin, :40], gt_phys[:, tmin, 40:80]
            rec[tmin] = {"Nk": agg(pN, gN), "Mk": agg(pM, gM),
                         "totN": tot(pN, gN), "totM": tot(pM, gM)}
        # physical-consistency diagnostics over the whole chain
        Ms = p[:, :, 40:80].sum(-1)
        prev = np.concatenate([gt_phys[:, t0, 40:80].sum(1)[:, None], Ms[:, :-1]], 1)
        rec["mono_viol_frac"] = float((Ms < prev*(1-1e-6)).mean())
        rec["finite_frac"] = float(np.isfinite(pl).all(axis=(1, 2)).mean())
        metrics["per_horizon"][proto] = rec

    with open(f"{a.out}/chain_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    lines = [f"CHAINED 24h EVAL | run={a.run} ckpt={a.ckpt} constraint={a.mode} N={N} "
             f"windows={n_win} start={t0}min",
             "", "per-bin rel-err vs TOMAS truth (pooled over non-saturated bins)", ""]
    lines.append(f"{'t(h)':>5s} | {'self Nk_p50':>11s} {'self Nk_p90':>11s} | {'gt Nk_p50':>9s} {'gt Nk_p90':>9s} | "
                 f"{'self Mk_p90':>11s} | {'self totM_p90':>13s}")
    for j, tmin in horizons:
        if (tmin - t0) % 180: continue
        s = metrics["per_horizon"]["self"][tmin]; g = metrics["per_horizon"]["gtstart"][tmin]
        lines.append(f"{tmin/60:5.1f} | {s['Nk'][0]:11.3f} {s['Nk'][1]:11.3f} | {g['Nk'][0]:9.3f} {g['Nk'][1]:9.3f} | "
                     f"{s['Mk'][1]:11.3f} | {s['totM'][1]:13.3f}")
    for proto in ("self", "gtstart"):
        r = metrics["per_horizon"][proto]
        lines += ["", f"{proto}: monotonicity violations {r['mono_viol_frac']*100:.2f}%  finite {r['finite_frac']*100:.1f}%"]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(f"{a.out}/chain_{tag}.txt", "w") as f:
        f.write(txt + "\n")


def Pathsafe(s):
    return s.strip("/").replace("/", "_")


if __name__ == "__main__":
    main()
