"""After the finishing run: rank per-epoch checkpoints by the 14-day self-rollout metric (parsed from the log), average the
weights of the best K (default 3) and evaluate the averaged model (self-roll on injection + continuous burst cells, 6h all-sets)."""
import re, json, numpy as np, torch, torch.nn as nn
from frontier_common import *
_tf = open("/data/tomas-emulator/frontier_transformers.py").read(); ns = {"torch": torch, "nn": nn, "N_BINS": N_BINS, "P": N_BINS + 2, "T": 12}
exec(_tf[_tf.index("def enc("):_tf.index("torch.manual_seed(0)")], ns); BinTF = ns["BinTF"]
D_ = int(os.environ.get("D", "256")); NL_ = int(os.environ.get("NL", "4"))
LOG = os.environ.get("LOG", "/data/tomas-emulator/frontier_logs/single_finish.log"); DIR = os.environ.get("DIR", f"{EMU}/frontier_head_kchain_single_finish"); TOPK = int(os.environ.get("TOPK", "3"))
rows = []
for m in re.finditer(r"K=48 ep(\d+): self-roll v9_sai \[(.*?)\]\n", open(LOG).read()):
    ep = int(m.group(1)); vals = re.findall(r"\((\d+), (\d+), (\d+), ([\d.]+)\)", m.group(2)); d14 = [v for v in vals if v[0] == "14"][0]
    rows.append((ep, int(d14[2]), float(d14[3])))
rows.sort(key=lambda r: r[1] + r[2]); log(f"epochs ranked by day-14 totN+totM: {rows}")
best = [r[0] for r in rows[:TOPK]]
sds = [torch.load(f"{DIR}/stage_K48_ep{e}.pt", map_location=dev) for e in best]
avg_m = {k: sum(sd["model"][k].float() for sd in sds) / len(sds) for k in sds[0]["model"]}
bt = BinTF(D_, NL_).to(dev).eval(); bt.load_state_dict(avg_m)
torch.save({"model": bt.state_dict(), "avg_of": best}, f"{DIR}/avg_top{TOPK}.pt")
def block(cur, cb): return bt(cur, cb)[:, -1]
def cells(src_, burst=True, n=1000):
    S, C = load_v8_set(f"/data/tomas-jax-output/{src_}", "test", 6000, verbose=False); S, C = torch.from_numpy(S), torch.from_numpy(C)
    keep = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0; S, C = S[keep], C[keep]
    if burst: b = burst_scen_mask(S); S, C = S[b], C[b]
    g = torch.Generator().manual_seed(0); idx = torch.randperm(S.shape[0], generator=g)[:n]; return S[idx].to(dev), C[idx].to(dev)
@torch.no_grad()
def selfroll(S, C):
    cur = norm.norm_state(S[:, 0]); cn = norm.norm_cond(C); out = []
    for b in range(56):
        cur = torch.nan_to_num(block(cur, cn[:, b * 12:b * 12 + 13]), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50); step = (b + 1) * 12
        if step in (48, 144, 336, 480, 672):
            p = norm.denorm_state(cur); t = S[:, step]; pN, tN = torch.exp(p[:, Nsl].clamp(max=60)), torch.exp(t[:, Nsl])
            rN = (pN.sum(1) - tN.sum(1)).abs() / tN.sum(1).clamp(min=1e-25); rM = (tot(p, Msl) - tot(t, Msl)).abs() / tot(t, Msl).clamp(min=1e-25)
            out.append((step // 48, round(float(rN.nanmedian()) * 100), round(float(rN.nanquantile(0.9)) * 100), round(float(rM.nanmedian()) * 100), round(float(rM.nanquantile(0.9)) * 100)))
    return out
for name, src_ in (("v9_sai", "v9_sai"), ("v9_sai_cont", "v9_sai_cont")):
    S, C = cells(src_); log(f"[avg top{TOPK} of ep{best}] self-roll {name} (day, totN p50, p90, totM p50, p90): {selfroll(S, C)}")
with torch.no_grad(): eval_all_sets(lambda s, c: norm.denorm_state(block(s, c)), load_all_tests(), f"[avg top{TOPK}]")
log("DONE")
