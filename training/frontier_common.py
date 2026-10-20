"""Shared harness for the burst-frontier probe suite (2026-09-22).

Data: INJ = v9_sai (filtered N>1/cm3), BG = v8_carma_2wk. 6h windows t=0..12.
Metrics at 6h: totals, per-bin (legacy per-box burst def + per-cm3 def), occupied
per-bin, Wasserstein-1 along the bin axis, integrated N>10nm / N>100nm / surface
area, speed. Also routed-backbone feature extraction for head probes.
"""
import os, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/data/tomas-emulator")
from dataset_v8 import load_v8_set, NormStatsV8, N_BINS
from model_routed import BinRoutedHybrid

dev = "cuda"; BOX = 1e6
SMOKE = os.environ.get("SMOKE", "0") == "1"
EMU = "/data/tomas-jax-output/emulator_v1"
norm = NormStatsV8.load(f"{EMU}/foundation_mlp180/norm_stats.npz")
Nsl = slice(0, N_BINS); Msl = slice(N_BINS, 2 * N_BINS)
idx13 = np.arange(13)
D_UM = 1.7e-3 * 2.0 ** ((np.arange(N_BINS) + 0.5) / 3.0)      # bin-centre diameter, um
BIN_10NM = int(np.argmax(D_UM > 0.010))                       # first bin > 10 nm
BIN_100NM = int(np.argmax(D_UM > 0.100))                      # first bin > 100 nm
D2 = torch.tensor(D_UM ** 2, dtype=torch.float32)


def log(*a):
    print(*a, flush=True)


def load_inj(role, n, T=13):
    S, C = load_v8_set("/data/tomas-jax-output/v9_sai", role, n, verbose=False)
    S = torch.from_numpy(S[:, :T]).contiguous(); C = torch.from_numpy(C[:, :T]).contiguous()
    keep = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0
    return S[keep], C[keep]


def load_bg(role, n, T=13):
    S, C = load_v8_set("/data/tomas-jax-output/v8_carma_2wk", role, n, verbose=False)
    return torch.from_numpy(S[:, :T]).contiguous(), torch.from_numpy(C[:, :T]).contiguous()


def burst_scen_mask(S, t1=12):
    """scenario-level: some bin 0..15 goes from <=1e3/cm3 to >1e3/cm3 by t1."""
    return ((torch.exp(S[:, 0, :16]) / BOX <= 1e3) & (torch.exp(S[:, t1, :16]) / BOX > 1e3)).any(1)


def load_test_sets():
    n_i, n_b = (2000, 1000) if SMOKE else (16000, 6000)
    SvI, CvI = load_inj("test", n_i); SvB, CvB = load_bg("test", n_b)
    log(f"test INJ {SvI.shape[0]} (burst scen {int(burst_scen_mask(SvI).sum())}) | BG {SvB.shape[0]}")
    return SvI, CvI, SvB, CvB


def load_train_sets(n_inj=160000, n_bg=40000):
    if SMOKE: n_inj, n_bg = 4000, 2000
    Sin, Cin = load_inj("train", n_inj); Sbg, Cbg = load_bg("train", n_bg)
    log(f"train INJ {Sin.shape[0]} | BG {Sbg.shape[0]}")
    return Sin, Cin, Sbg, Cbg


def tot(x, sl):
    return torch.exp(torch.clamp(x[..., sl], max=30.0)).sum(-1)


def totals_loss(pred_n, tgt_n, lam_m=0.15, lam_n=0.35):
    """numaware recipe: mass rel-L1 + tail-weighted number rel-L1 (denormed)."""
    pd, td = norm.denorm_state(pred_n), norm.denorm_state(tgt_n)
    m_re = (torch.abs(tot(pd, Msl) - tot(td, Msl)) / torch.clamp(tot(td, Msl), min=1e-25)).clamp(max=5.0)
    n_re = (torch.abs(tot(pd, Nsl) - tot(td, Nsl)) / torch.clamp(tot(td, Nsl), min=1e-25)).clamp(max=5.0)
    with torch.no_grad():
        thr = torch.quantile(n_re.flatten(), 0.9); w = 1.0 + (n_re > thr).float()
    return lam_m * m_re.mean() + lam_n * (w * n_re).mean()


def _q(v, p):
    return float(v.quantile(p)) * 100 if v.numel() else float("nan")


@torch.no_grad()
def metrics6h(pred, t, s0):
    """pred, t, s0: (N,82) denormed log-state on the same device. Returns dict of %."""
    pN = torch.exp(torch.clamp(pred[:, Nsl], max=60)); tN = torch.exp(t[:, Nsl]); N0 = torch.exp(s0[:, Nsl])
    pM = torch.exp(torch.clamp(pred[:, Msl], max=60)); tM = torch.exp(t[:, Msl])
    re = lambda a, b: torch.abs(a - b) / torch.clamp(b, min=1e-25)
    out = {}
    rN, rM = re(pN.sum(1), tN.sum(1)), re(pM.sum(1), tM.sum(1))
    out["totN50"], out["totN90"], out["totM50"], out["totM90"] = _q(rN, .5), _q(rN, .9), _q(rM, .5), _q(rM, .9)
    pb = re(pN, tN)                                                    # (N,40) per-bin rel err
    bm_leg = (N0 <= 1e3) & (tN > 1e3)                                  # legacy per-box def
    bm_cc = (N0 / BOX <= 1e3) & (tN / BOX > 1e3); bm_cc[:, 16:] = False  # per-cm3 def, bins 0..15
    occ = tN / BOX > 1e-2
    out["burstL50"], out["burstL90"] = _q(pb[bm_leg], .5), _q(pb[bm_leg], .9)
    out["burstC50"], out["burstC90"] = _q(pb[bm_cc], .5), _q(pb[bm_cc], .9)
    out["occ50"], out["occ90"] = _q(pb[occ], .5), _q(pb[occ], .9)
    # W1 along bin axis (bins), number-fraction CDFs, burst scenarios only
    bs = bm_cc.any(1)
    if bs.any():
        cp = torch.cumsum(pN[bs] / pN[bs].sum(1, keepdim=True).clamp(min=1e-30), 1)
        ct = torch.cumsum(tN[bs] / tN[bs].sum(1, keepdim=True).clamp(min=1e-30), 1)
        w1 = (cp - ct).abs().sum(1)
        out["W1b50"], out["W1b90"] = float(w1.median()), float(w1.quantile(.9))
        # integrated quantities on burst scenarios
        d2 = D2.to(pred.device)
        for name, sl in (("N10", slice(BIN_10NM, N_BINS)), ("N100", slice(BIN_100NM, N_BINS))):
            r = re(pN[bs][:, sl].sum(1), tN[bs][:, sl].sum(1))
            out[name + "50"], out[name + "90"] = _q(r, .5), _q(r, .9)
        r = re((pN[bs] * d2).sum(1), (tN[bs] * d2).sum(1))
        out["SA50"], out["SA90"] = _q(r, .5), _q(r, .9)
    out["n_burst_scen"] = int(bs.sum())
    return out


def fmt(m):
    k = lambda a: f"{m.get(a, float('nan')):.1f}"
    s = (f"totN {k('totN50')}/{k('totN90')} totM {k('totM50')}/{k('totM90')} | "
         f"burstL {k('burstL50')}/{k('burstL90')} burstC {k('burstC50')}/{k('burstC90')} occ {k('occ50')}/{k('occ90')}")
    if "W1b50" in m:
        s += (f" | W1(bins) {m['W1b50']:.2f}/{m['W1b90']:.2f} N>10nm {k('N10' + '50')}/{k('N1090')} "
              f"N>100nm {k('N10050')}/{k('N10090')} SA {k('SA50')}/{k('SA90')} [{m['n_burst_scen']} burst]")
    return s


@torch.no_grad()
def eval_traj_model(fwd, Sv, Cv, bs=512):
    """fwd(sn0 (B,82), cn (B,13,6)) -> denormed 6h log-state (B,82). Accumulates metrics."""
    P, T0, S0 = [], [], []
    for a in range(0, Sv.shape[0], bs):
        s = Sv[a:a + bs].to(dev); c = Cv[a:a + bs].to(dev)
        sn = norm.norm_state(s[:, idx13]); cn = norm.norm_cond(c[:, idx13])
        P.append(fwd(sn[:, 0], cn).float()); T0.append(s[:, 12]); S0.append(s[:, 0])
    return metrics6h(torch.cat(P), torch.cat(T0), torch.cat(S0))


def speed_bench(fn, bs=4096, dtype=torch.float32, n=30):
    """fn(bs) runs one 6h inference for bs cells; returns us/cell."""
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        for _ in range(5): fn(bs)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n): fn(bs)
        torch.cuda.synchronize()
    return (time.time() - t0) / n / bs * 1e6


def load_routed(path, d=176, bb=4, burst=3, ff=352):
    m = BinRoutedHybrid(d=d, backbone_blocks=bb, burst_blocks=burst, nhead=4, ff=ff).to(dev)
    ck = torch.load(path, map_location=dev); m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m.eval()


def routed_features(m, s0, cond_seq):
    """Replicates BinRoutedHybrid.forward, returning (pred (B,T,82), feats x (B,T,42,d))."""
    B = s0.shape[0]
    nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]; gas = s0[:, 2 * N_BINS:]
    tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)
    tok_e = m.tok_in(tok)
    ctx = m.gas_bcast(gas) + m.glob_in(tok_e.mean(dim=1))
    tok_e = tok_e + ctx.unsqueeze(1)
    x = tok_e.unsqueeze(1).expand(B, m.n_out, m.P, -1).clone()
    x = x + m.bin_emb.view(1, 1, m.P, -1) + m.time_emb.view(1, m.n_out, 1, -1)
    x = x + m.cond_in(cond_seq[:, 1:1 + m.n_out]).unsqueeze(2)
    for blk in m.backbone: x = blk(x)
    gctx = m.gctx(x.mean(dim=2, keepdim=True)); burst = x[:, :, :m.BH]
    for blk in m.specialist: burst = blk(burst, gctx)
    refined = m.blend * burst + (1 - m.blend) * x[:, :, :m.BH]
    x = torch.cat([refined, x[:, :, m.BH:]], dim=2)
    d = m.head(m.norm(x))
    base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
    pred = base + torch.cat([d[:, :, :N_BINS, 0], d[:, :, :N_BINS, 1], d[:, :, N_BINS:, 0]], dim=-1)
    return pred, x


class Mixer:
    """50/50 bg/inj minibatch sampler over 6h windows."""
    def __init__(self, Sin, Cin, Sbg, Cbg, bs, seed=0):
        self.Sin, self.Cin, self.Sbg, self.Cbg, self.h = Sin, Cin, Sbg, Cbg, bs // 2
        self.g = torch.Generator().manual_seed(seed)
        self.ni, self.nb = Sin.shape[0], Sbg.shape[0]
        self.steps_per_epoch = (self.ni + self.nb) // bs
        self._reset()

    def _reset(self):
        self.pi = torch.randperm(self.ni, generator=self.g); self.pb = torch.randperm(self.nb, generator=self.g)
        self.ii = self.bi = 0

    def next(self):
        if self.ii + self.h > self.ni or self.bi + self.h > self.nb: self._reset()
        a, b = self.pi[self.ii:self.ii + self.h], self.pb[self.bi:self.bi + self.h]
        self.ii += self.h; self.bi += self.h
        s = torch.cat([self.Sin[a], self.Sbg[b]], 0).to(dev); c = torch.cat([self.Cin[a], self.Cbg[b]], 0).to(dev)
        return norm.norm_state(s[:, idx13]), norm.norm_cond(c[:, idx13])


# ---------------- multi-set support (added 03:50 after the all-sets gate) ----------------
ALL_SETS = ["v8_carma", "v8_carma_2wk", "v9_sai", "v9_sai_cont", "v10_ideal"]
DEFAULT_SPEC = "v9_sai:160000:0.30,v9_sai_cont:67344:0.20,v10_ideal:86690:0.20,v8_carma:300000:0.15,v8_carma_2wk:40000:0.15"


def load_multiset(spec=None, role="train", T=13):
    """spec 'name:n:weight,...' -> list of (name, S, C, weight). Filters measurable air (N>1/cm3)."""
    spec = spec or os.environ.get("SETS", DEFAULT_SPEC); out = []
    for item in spec.split(","):
        name, n, w = item.split(":"); n = int(n) if not SMOKE else min(int(n), 3000)
        S, C = load_v8_set(f"/data/tomas-jax-output/{name}", role, n, verbose=False)
        S = torch.from_numpy(S[:, :T]).contiguous(); C = torch.from_numpy(C[:, :T]).contiguous()
        keep = torch.exp(S[:, 0, :N_BINS]).sum(1) / BOX > 1.0; S, C = S[keep], C[keep]
        log(f"  {name}[{role}]: {S.shape[0]} measurable (weight {w})"); out.append((name, S, C, float(w)))
    return out


def load_all_tests(n=6000, T=13):
    n = 1000 if SMOKE else n
    return {name: (S, C) for name, S, C, _ in load_multiset(",".join(f"{s}:{n}:1" for s in ALL_SETS), "test", T)}


class MultiMixer:
    """weighted minibatch sampler over several (S,C) sets; returns normalized (sn (B,13,82), cn (B,13,6))."""
    def __init__(self, sets, bs, seed=0):
        self.sets = sets; self.bs = bs; self.g = torch.Generator().manual_seed(seed)
        w = torch.tensor([s[3] for s in sets]); self.counts = torch.round(w / w.sum() * bs).long()
        self.counts[0] += bs - self.counts.sum()
        self.perm = [torch.randperm(s[1].shape[0], generator=self.g) for s in sets]; self.pos = [0] * len(sets)
        self.steps_per_epoch = sum(s[1].shape[0] for s in sets) // bs

    def next(self):
        ss, cs = [], []
        for i, (name, S, C, _) in enumerate(self.sets):
            k = int(self.counts[i]); 
            if k == 0: continue
            if self.pos[i] + k > S.shape[0]: self.perm[i] = torch.randperm(S.shape[0], generator=self.g); self.pos[i] = 0
            idx = self.perm[i][self.pos[i]:self.pos[i] + k]; self.pos[i] += k; ss.append(S[idx]); cs.append(C[idx])
        s = torch.cat(ss).to(dev); c = torch.cat(cs).to(dev)
        return norm.norm_state(s[:, idx13]), norm.norm_cond(c[:, idx13])


def eval_all_sets(f, tests, tag):
    for name, (S, C) in tests.items():
        log(f"{tag} {name:13s} {fmt(eval_traj_model(f, S, C, bs=256))}")
