"""Phase A2: why the chained projection hurt, and which anchoring actually works.

Phase A found: exact budget enforcement via the champion's uniform (Mk,Gc)
log-shift, chained across the 12 horizons, holds the invariant to 1e-6 but
blows up totM p90 0.034 -> 0.437. This script diagnoses the cause and tests
alternative anchorings.

Variants:
  raw          aug5 as-is
  chain        anchor = PROJECTED previous horizon (Phase A; telescoping)
  local        anchor = RAW model prediction at k-1 (per-window budget, no
               accumulation of the 30-min budget approximation)
  gc_local     Mk untouched; Gc completes the budget (gas absorbs the residual),
               anchor = raw prev.  Falls back to a shift only if Gc would go <= 0.
  gc_chain     same completion but anchored on the projected chain.

Diagnostic: the gas/aerosol split of total sulfur, which controls how much a
uniform shift on the SUM perturbs the aerosol.
"""
import os, json
import numpy as np, torch
from dataset_v4 import load_v4_batches, build_state_array_v4, transform_cond, NormStatsV4, N_BINS, LOG_EPS
from model_axial_traj import AxialTrajOperator, DT_MIN

RUN_DIR = os.environ.get("RUN_DIR", "checkpoints/axial_aug5")
OUT = os.environ.get("OUT", "reports/conservation")
WS, MAXS, PRV, N_EVAL, SEED = 7, 140000, 4000, 800, 0
FT, EF = 1.5, 0.5
LOG_CONST = float(np.log(1e-6 * 1e6 * 98e-3 / 6.022e23 / 1.380649e-23))
XK0 = (np.pi/6.0)*(1.7e-9)**3*1770.0
XK = XK0 * 2.0**np.arange(40)
SLACK = 1.25
torch.set_num_threads(max(8, os.cpu_count() // 3))
os.makedirs(OUT, exist_ok=True)

print("loading held-out ...", flush=True)
Nk, Mk, Gc, cond = load_v4_batches(os.environ.get("DATA_DIR", "data/v4_repr_1M"), max_scenarios=MAXS, rank=0, world_size=WS, verbose=False)
sel = np.random.default_rng(SEED).permutation(Nk.shape[0])[:PRV][:N_EVAL]
ls = build_state_array_v4(Nk[sel], Mk[sel], Gc[sel])
ct = transform_cond(cond[sel])
del Nk, Mk, Gc, cond
ns = NormStatsV4.load(f"{RUN_DIR}/norm_stats.pt")

ck = torch.load(f"{RUN_DIR}/best_model.pt", weights_only=False, map_location="cpu"); ma = ck["args"]
GRID = [int(x) for x in ma["time_grid"].split(",")] if ma.get("time_grid") else list(range(DT_MIN, 361, DT_MIN))
TP = [0] + GRID
n_cond = ck["model_state_dict"]["cond_in.weight"].shape[1]
m = AxialTrajOperator(d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"], dim_ff=ma["dim_ff"],
                      dropout=0.0, ff_experts=ma.get("ff_experts", 1), n_out=len(GRID), n_cond=n_cond)
m.load_state_dict(ck["model_state_dict"]); m.eval()

s0 = ns.normalize_state(torch.as_tensor(ls[:, 0, :], dtype=torch.float32))
c13 = ns.normalize_cond(torch.as_tensor(ct[:, TP, :], dtype=torch.float32))
print("forward ...", flush=True)
with torch.no_grad():
    pred_raw = torch.cat([m(s0[i:i+100], c13[i:i+100]) for i in range(0, N_EVAL, 100)])
pl = ns.denormalize_state(pred_raw.reshape(-1, 81)).reshape(N_EVAL, len(GRID), 81).numpy().astype(np.float64)

# physical arrays
pN = np.exp(pl[:, :, :40]); pM = np.exp(pl[:, :, 40:80]); pG = np.exp(pl[:, :, 80])   # (N,12,*)
gt = np.exp(ls.astype(np.float64))
gN = gt[:, GRID, :40]; gM = gt[:, GRID, 40:80]; gG = gt[:, GRID, 80]
M0 = gt[:, 0, 40:80].sum(1)
H = np.exp(ct[..., 4] + ct[..., 0] - np.log(ct[..., 2]) + LOG_CONST).astype(np.float64)  # (N,361)
Hk = H[:, [0] + GRID[:-1]]        # start-of-interval forcing for each of the 12 windows

# ---- diagnostic: gas share of total sulfur, and the budget's own error ----
gas_share = gG / np.maximum(gM.sum(-1) + gG, 1e-300)
print("\nGT gas share of total sulfur (Gc/(sumMk+Gc)):")
for q in (10, 50, 90, 99):
    print(f"  p{q}: {np.percentile(gas_share, q):.4f}")
tgt_gt = np.concatenate([M0[:, None], gM[:, :-1].sum(-1)], 1) + Hk
budget_gt = np.abs(gM.sum(-1) + gG - tgt_gt) / np.maximum(tgt_gt, 1e-300)
print(f"GT 30-min budget closure (start-anchor): p50={np.median(budget_gt):.4f} "
      f"p90={np.percentile(budget_gt,90):.4f} p99={np.percentile(budget_gt,99):.4f}")
# how much of the target is the fresh gas term vs the carried aerosol?
frac_fresh = Hk / np.maximum(tgt_gt, 1e-300)
print(f"fresh-gas fraction of budget target: p50={np.median(frac_fresh):.3f} p90={np.percentile(frac_fresh,90):.3f}")

# ---------------- projection variants (numpy, log-space) ----------------
def bin_clamp(logM, logN):
    lo = np.log(XK/SLACK)[None, :]; hi = np.log(2*XK*SLACK)[None, :]
    mm = np.clip(logM - logN, lo, hi)
    return logN + mm

def project(mode, clamp=np.log(10.0), n_iter=3):
    """Returns (N,12,40) Mk, (N,12) Gc, physical."""
    oM = np.empty_like(pM); oG = np.empty_like(pG)
    anchor = M0.copy()
    for k in range(len(GRID)):
        logN = np.log(np.maximum(pN[:, k], 1e-300))
        logM = np.log(np.maximum(pM[:, k], 1e-300))
        logG = np.log(np.maximum(pG[:, k], 1e-300))
        target = anchor + Hk[:, k]
        if mode.startswith("gc"):
            # gas completes the budget; aerosol untouched where possible
            Ms = np.exp(logM).sum(1)
            gnew = target - Ms
            bad = gnew <= 0
            logG = np.where(bad, logG, np.log(np.maximum(gnew, 1e-300)))
            if bad.any():   # fallback: uniform shift for scenarios where gas can't absorb it
                lps = np.log(Ms + np.exp(logG))
                f = np.clip(np.log(np.maximum(target, 1e-300)) - lps, -clamp, clamp)
                logM = np.where(bad[:, None], logM + f[:, None], logM)
                logG = np.where(bad, logG + f, logG)
            logM = bin_clamp(logM, logN)
        else:
            for _ in range(n_iter):
                lps = np.log(np.exp(logM).sum(1) + np.exp(logG))
                f = np.clip(np.log(np.maximum(target, 1e-300)) - lps, -clamp, clamp)
                logM = logM + f[:, None]; logG = logG + f
                logM = bin_clamp(logM, logN)
        oM[:, k] = np.exp(logM); oG[:, k] = np.exp(logG)
        anchor = oM[:, k].sum(1) if mode in ("chain", "gc_chain") else pM[:, k].sum(1)
    return oM, oG

variants = {"raw": (pM, pG)}
variants["chain"] = project("chain")
variants["local"] = project("local")
variants["gc_local"] = project("gc_local")
variants["gc_chain"] = project("gc_chain")

# ---------------- metrics ----------------
def agg(pp, gg):
    rels = []
    for b in range(N_BINS):
        g = gg[:, b]; p = pp[:, b]; pos = g[g > 10*LOG_EPS]
        if pos.size < 10: continue
        fl = pos.min(); nf = g <= FT*fl
        if nf.mean() > EF: continue
        kk = ~nf; r = np.abs(p[kk]-g[kk])/g[kk]; r = r[np.isfinite(r)]
        if r.size: rels.append(r)
    ar = np.concatenate(rels)
    return float(np.median(ar)), float(np.percentile(ar, 90))

def tot(pp, gg):
    r = np.abs(pp.sum(-1)-gg.sum(-1))/np.maximum(gg.sum(-1), 1e-300); r = r[np.isfinite(r)]
    return float(np.median(r)), float(np.percentile(r, 90))

res = {}
lines = []
lines.append(f"{'variant':10s} | {'Mk_p50':>7s} {'Mk_p90':>7s} | {'totM_p50':>8s} {'totM_p90':>8s} | "
             f"{'Gc_p50':>7s} {'Gc_p90':>7s} | {'resid_p50':>9s} {'resid_p90':>9s}   (@6h)")
for name, (vM, vG) in variants.items():
    k = len(GRID)-1
    mk = agg(vM[:, k], gM[:, k]); tm = tot(vM[:, k], gM[:, k])
    rg = np.abs(vG[:, k]-gG[:, k])/np.maximum(gG[:, k], 1e-300); rg = rg[np.isfinite(rg)]
    # residual of the variant's OWN self-consistent chain (its prev sum as anchor)
    anc = np.concatenate([M0[:, None], vM[:, :-1].sum(-1)], 1) + Hk
    rr = np.abs(vM.sum(-1)+vG-anc)/np.maximum(anc, 1e-300)
    res[name] = {"Mk": mk, "totM": tm, "Gc": [float(np.median(rg)), float(np.percentile(rg, 90))],
                 "resid": [float(np.median(rr)), float(np.percentile(rr, 90))]}
    lines.append(f"{name:10s} | {mk[0]:7.3f} {mk[1]:7.3f} | {tm[0]:8.3f} {tm[1]:8.3f} | "
                 f"{np.median(rg):7.3f} {np.percentile(rg,90):7.3f} | "
                 f"{np.median(rr):9.2e} {np.percentile(rr,90):9.2e}")

txt = "\n".join(lines)
print("\n" + txt)
with open(f"{OUT}/variants.txt", "w") as f:
    f.write("PHASE A2 — projection anchoring variants (aug5, 800 held-out, @6h)\n\n")
    f.write(f"GT gas share p50={np.median(gas_share):.4f} p90={np.percentile(gas_share,90):.4f}\n")
    f.write(f"GT 30-min budget closure: p50={np.median(budget_gt):.4f} p90={np.percentile(budget_gt,90):.4f} "
            f"p99={np.percentile(budget_gt,99):.4f}\n")
    f.write(f"fresh-gas fraction of target: p50={np.median(frac_fresh):.3f}\n\n")
    f.write(txt + "\n")
with open(f"{OUT}/variants.json", "w") as f:
    json.dump({"res": res, "gas_share_p50": float(np.median(gas_share)),
               "budget_gt": [float(np.median(budget_gt)), float(np.percentile(budget_gt, 90))]}, f, indent=2)
print("\nwritten ->", OUT)
