"""K-curriculum ROLLOUT training for the bin-token transformer (non-AR, bounded), warm-started from
bintf_multi_bw4, on chains from all long-window populations. Loss = MSE + totals + BW x burst-bin MSE
(all steps). Stages K=2,4,8 (12h..48h). Per-epoch checkpoints. Eval: 6h all-sets + 14d self-roll (v9_sai, cont)."""
import numpy as np, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from frontier_common import *
import torch.distributed as dist
DDP_ON = "RANK" in os.environ
if DDP_ON:
    dist.init_process_group("nccl"); RANK = dist.get_rank(); WORLD = dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else: RANK, WORLD = 0, 1
IS_MAIN = RANK == 0
_log = log; log = (lambda *a, **k: _log(*a, **k)) if IS_MAIN else (lambda *a, **k: None)
BW = float(os.environ.get("BW", "4")); EXT = os.environ.get("EXT", "0") == "1"; LR = float(os.environ.get("LR", "1e-4"))
TAG = os.environ.get("TAG", ""); HW = float(os.environ.get("HW", "0"))        # horizon weighting: block b weight 1 + HW*b/(K-1)
NOISE = float(os.environ.get("NOISE", "0")); LAMM = float(os.environ.get("LAMM", "0.15")); LAMN = float(os.environ.get("LAMN", "0.35"))
WARM = os.environ.get("WARM", ""); STAGES_ENV = os.environ.get("STAGES", "")   # "K:epochs:bs,..."
BURSTOS = float(os.environ.get("BURSTOS", "1"))
MIX = float(os.environ.get("MIX", "0")); D_ = int(os.environ.get("D", "256")); NL_ = int(os.environ.get("NL", "4"))
BLOCKNORM = os.environ.get("BLOCKNORM", "0") == "1"   # equalize per-block gradient contributions (weights ~ 1/detached block MSE)
TF1 = float(os.environ.get("TF1", "0"))               # teacher-forced one-step loss from every true chain state (K fresh 6h samples per chain)
BASE = f"{EMU}/frontier_bintf_kchain"; OUT = BASE + (f"_{TAG}" if TAG else ""); os.makedirs(OUT, exist_ok=True)
log(f"TAG={TAG} HW={HW} NOISE={NOISE} LAMM={LAMM} LAMN={LAMN} LR={LR} MIX={MIX} BLOCKNORM={BLOCKNORM} TF1={TF1} D={D_} NL={NL_} STAGES={STAGES_ENV or 'default'} WARM={WARM or 'default'}")
src = open("/data/tomas-emulator/frontier_transformers.py").read(); ns = {"torch": torch, "nn": nn, "N_BINS": N_BINS, "P": N_BINS + 2, "T": 12}
exec(src[src.index("def enc("):src.index("torch.manual_seed(0)")], ns); BinTF = ns["BinTF"]
TESTS = load_all_tests()
def load_long(s, n, inj):
    S, C = load_v8_set(f"/data/tomas-jax-output/{s}", "train", n, verbose=False); S, C = torch.from_numpy(S), torch.from_numpy(C)
    if inj: k = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0; S, C = S[k], C[k]
    return S, C
parts = [load_long("v9_sai", 4000 if SMOKE else 120000, True), load_long("v8_carma_2wk", 2000 if SMOKE else 30000, False),
         load_long("v9_sai_cont", 3000 if SMOKE else 67344, True), load_long("v10_ideal", 3000 if SMOKE else 86690, True)]
St = torch.cat([p[0] for p in parts]); Ct = torch.cat([p[1] for p in parts]); n = St.shape[0]; del parts
def burst_cells(s):
    S, C = load_v8_set(f"/data/tomas-jax-output/{s}", "test", 2000 if SMOKE else 6000, verbose=False); S, C = torch.from_numpy(S), torch.from_numpy(C)
    k = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0; S, C = S[k], C[k]; b = burst_scen_mask(S); return S[b][:1000].to(dev), C[b][:1000].to(dev)
ROLL = {"v9_sai": burst_cells("v9_sai"), "v9_sai_cont": burst_cells("v9_sai_cont")}
log(f"train chains {n} | self-roll cells " + ", ".join(f"{k}:{v[0].shape[0]}" for k, v in ROLL.items()))
N0 = torch.exp(St[:, 0, :16]) / BOX; Nmax = torch.exp(St[:, :97, :16]).amax(1) / BOX
BURST_CHAIN = ((N0 <= 1e3) & (Nmax > 1e3)).any(1); CHAIN_W = torch.where(BURST_CHAIN, torch.tensor(BURSTOS), torch.tensor(1.0))
log(f"burst-active chains: {int(BURST_CHAIN.sum())} / {n} ({BURST_CHAIN.float().mean()*100:.0f}%), oversampled x{BURSTOS}")
if MIX > 0: SIX = load_multiset(); mix6 = MultiMixer(SIX, 64, seed=5 + RANK); log("mixed objective: 6h windows from all populations")
m = BinTF(D_, NL_).to(dev); m.load_state_dict(torch.load(WARM or (f"{BASE}/stage_K8.pt" if EXT else f"{EMU}/frontier_tf_bintf_d256_l4_bw4_multi/best.pt"), map_location=dev)["model"]); m.train(); log("EXT warm from stage_K8" if EXT else "warm from bintf_multi_bw4")
if DDP_ON:
    m_ddp = torch.nn.parallel.DistributedDataParallel(m, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False, broadcast_buffers=False)
    log(f"DDP world size {WORLD}; per-rank chains/epoch = CAP48, effective batch = bs x {WORLD}")
else: m_ddp = m
def chain(model, s0, cond, K, ck=False, noise=0.0):
    cur, outs = s0, []
    for b in range(K):
        cb = cond[:, b * 12:b * 12 + 13]
        if noise > 0 and model.training and b > 0: cur = cur + noise * torch.randn_like(cur)   # pushforward-style robustness
        o = checkpoint(model, cur, cb, use_reentrant=False) if ck else model(cur, cb); outs.append(o); cur = o[:, -1]
    return torch.cat(outs, 1)
@torch.no_grad()
def selfroll(model, which):
    S, C = ROLL[which]; model.eval(); cur = norm.norm_state(S[:, 0]); cn = norm.norm_cond(C); out = []
    for b in range(56):
        cur = model(cur, cn[:, b * 12:b * 12 + 13])[:, -1]; step = (b + 1) * 12
        if step in (48, 144, 288, 480, 672):
            p = norm.denorm_state(cur); t = S[:, step]; pN, tN = torch.exp(p[:, Nsl].clamp(max=60)), torch.exp(t[:, Nsl]); occ = tN / BOX > 1e-2
            pb = ((pN - tN).abs() / tN.clamp(min=1e-25))[occ]; rN = ((pN.sum(1) - tN.sum(1)).abs() / tN.sum(1).clamp(min=1e-25)).nanmedian().item()
            rM = ((tot(p, Msl) - tot(t, Msl)).abs() / tot(t, Msl).clamp(min=1e-25)).nanmedian().item(); r = lambda v: round(v) if np.isfinite(v) else float("nan")
            out.append((step // 48, r(pb.nanmedian().item() * 100), r(rN * 100), r(rM * 100)))
    model.train(); return out
f6 = lambda s, c: norm.denorm_state(m(s, c)[:, -1])
def eval6(tag): m.eval(); eval_all_sets(f6, TESTS, tag); m.train()
if IS_MAIN:
    eval6("[start]")
    for w in ROLL: log(f"[start] self-roll {w}: {selfroll(m, w)}")
if DDP_ON: dist.barrier()
bi = list(range(16)) + list(range(N_BINS, N_BINS + 16))
STAGES = [(2, 1, 64, False)] if SMOKE else ([(16, 5, 8, True), (24, 5, 6, True)] if EXT else [(2, 3, 64, False), (4, 3, 32, False), (8, 5, 16, True)])
if STAGES_ENV: STAGES = [(int(a), int(b), int(c), int(a) >= 8) for a, b, c in (x.split(":") for x in STAGES_ENV.split(","))]
for si, (K, eps, bs, ck) in enumerate(STAGES):
    STEPS = K * 12; opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eps)
    log(f"\n=== STAGE {si}: K={K} ({K*6}h chains), {eps}ep, bs={bs} ===")
    for ep in range(eps):
        n_ep = 300 if SMOKE else (int(os.environ.get("CAP48", "4000")) if K >= 48 else 20000 if K >= 16 else 40000)
        g_ = torch.Generator().manual_seed(1000 * si + 17 * ep + RANK)
        perm = torch.multinomial(CHAIN_W, n_ep, replacement=False, generator=g_) if BURSTOS != 1.0 else torch.randperm(n, generator=g_)[:n_ep]
        for b in range(0, len(perm) - bs, bs):
            idx = perm[b:b + bs]; s = St[idx].to(dev); c = Ct[idx].to(dev); sn = norm.norm_state(s[:, :STEPS + 1]); cn = norm.norm_cond(c[:, :STEPS + 1])
            pred = chain(m_ddp, sn[:, 0], cn, K, ck, NOISE); tgt = sn[:, 1:STEPS + 1]
            se = (pred - tgt) ** 2
            if BLOCKNORM:
                with torch.no_grad():
                    mb = se.view(se.shape[0], K, 12, -1).mean((0, 2, 3)).clamp(min=1e-8); wb = mb.mean() / mb   # block weight ~ 1/err
                wgt = wb.repeat_interleave(12).view(1, STEPS, 1)
                mse = (se * wgt).mean() / wgt.mean(); bmse = (se[..., bi] * wgt).mean() / wgt.mean()
            elif HW > 0:
                wgt = (1 + HW * torch.arange(K, device=dev).float() / max(K - 1, 1)).repeat_interleave(12).view(1, STEPS, 1)
                mse = (se * wgt).mean() / wgt.mean(); bmse = se[..., bi].mean()
            else: mse = se.mean(); bmse = se[..., bi].mean()
            loss = mse + totals_loss(pred, tgt, LAMM, LAMN) + BW * bmse
            if TF1 > 0:   # one-step (6h) loss from each TRUE chain state: K fresh 6h windows per chain, on the regimes the chain visits
                Bn = sn.shape[0]; s_tf = sn[:, 0:STEPS:12].reshape(Bn * K, -1)
                widx = (torch.arange(K, device=dev) * 12).view(K, 1) + torch.arange(13, device=dev).view(1, 13)   # (K,13)
                c_tf = cn[:, widx].reshape(Bn * K, 13, -1); t_tf = sn[:, 1:STEPS + 1].reshape(Bn, K, 12, -1).reshape(Bn * K, 12, -1)
                p_tf = m(s_tf, c_tf)
                loss = loss + TF1 * (((p_tf - t_tf) ** 2).mean() + totals_loss(p_tf, t_tf, LAMM, LAMN) + BW * ((p_tf[..., bi] - t_tf[..., bi]) ** 2).mean())
            if MIX > 0:   # keep the plain 6h objective alive while chains lengthen
                sn6, cn6 = mix6.next(); p6 = m(sn6[:, 0], cn6); t6 = sn6[:, 1:]
                loss = loss + MIX * (((p6 - t6) ** 2).mean() + totals_loss(p6, t6, LAMM, LAMN) + BW * ((p6[..., bi] - t6[..., bi]) ** 2).mean())
            if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        if IS_MAIN:
            for w in ROLL: log(f"  K={K} ep{ep+1}: self-roll {w} {selfroll(m, w)}")
            m.eval(); eval_all_sets(f6, {"v9_sai": TESTS["v9_sai"]}, f"  K={K} ep{ep+1}: 6h"); m.train()
            torch.save({"model": m.state_dict(), "K": K, "ep": ep + 1}, f"{OUT}/stage_K{K}_ep{ep+1}.pt"); torch.save({"model": m.state_dict(), "K": K, "ep": ep + 1}, f"{OUT}/latest.pt")
        if DDP_ON: dist.barrier()
    if IS_MAIN: torch.save({"model": m.state_dict(), "K": K}, f"{OUT}/stage_K{K}.pt"); eval6(f"[stage K={K}]")
    if DDP_ON: dist.barrier()
log("DONE")
if DDP_ON: dist.destroy_process_group()
