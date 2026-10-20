"""Probe E4: transformer variants trained from scratch with the numaware recipe (50/50 bg/inj,
MSE + mass + tail-weighted number), same budget, same eval.
  ARCH=fulljoint : one transformer over all 12x42=504 (time,bin) tokens (no axial factorization)
  ARCH=bintf     : 42 bin tokens only, the 12 time steps live in the channel dim (speed-oriented)
Usage: ARCH=fulljoint|bintf D=176 NL=6 python frontier_transformers.py"""
import numpy as np, torch, torch.nn as nn
from frontier_common import *

ARCH = os.environ.get("ARCH", "fulljoint"); D = int(os.environ.get("D", "176")); NL = int(os.environ.get("NL", "6"))
MULTI = os.environ.get("MULTI", "0") == "1"; BURSTW = float(os.environ.get("BURSTW", "0"))
OUT = f"{EMU}/frontier_tf_{ARCH}_d{D}_l{NL}" + (f"_bw{BURSTW:g}" if BURSTW else "") + ("_multi" if MULTI else ""); os.makedirs(OUT, exist_ok=True)
P = N_BINS + 2; T = 12


def enc(d, nl, nh=4):
    layer = nn.TransformerEncoderLayer(d, nh, 2 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, nl)


class FullJointTraj(nn.Module):
    def __init__(self, d, nl):
        super().__init__()
        self.tok_in = nn.Linear(2, d); self.gas_bcast = nn.Linear(2, d); self.glob_in = nn.Linear(d, d)
        self.cond_in = nn.Linear(6, d)
        self.bin_emb = nn.Parameter(torch.randn(P, d) * 0.02); self.time_emb = nn.Parameter(torch.randn(T, d) * 0.02)
        self.enc = enc(d, nl); self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq):
        B = s0.shape[0]; nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]; gas = s0[:, 2 * N_BINS:]
        tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)
        e = self.tok_in(tok); e = e + (self.gas_bcast(gas) + self.glob_in(e.mean(1))).unsqueeze(1)
        x = e.unsqueeze(1).expand(B, T, P, -1) + self.bin_emb.view(1, 1, P, -1) + self.time_emb.view(1, T, 1, -1)
        x = x + self.cond_in(cond_seq[:, 1:1 + T]).unsqueeze(2)
        x = self.enc(x.reshape(B, T * P, -1)).reshape(B, T, P, -1)
        d = self.head(self.norm(x)); base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        return base + torch.cat([d[:, :, :N_BINS, 0], d[:, :, :N_BINS, 1], d[:, :, N_BINS:, 0]], -1)


class BinTF(nn.Module):
    """42 tokens; token = [Nk0,Mk0] + gas + all 12 cond vectors (72) -> d; output 12x2 per token."""
    def __init__(self, d, nl):
        super().__init__()
        self.tok_in = nn.Linear(2 + 2 + 6 * T, d); self.bin_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        self.enc = enc(d, nl); self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 2 * T)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq):
        B = s0.shape[0]; nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]; gas = s0[:, 2 * N_BINS:]
        tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)   # (B,42,2)
        glob = torch.cat([gas, cond_seq[:, 1:1 + T].flatten(1)], 1)                                      # (B,2+72)
        x = self.tok_in(torch.cat([tok, glob.unsqueeze(1).expand(B, P, -1)], -1)) + self.bin_emb
        d = self.head(self.norm(self.enc(x))).reshape(B, P, T, 2).permute(0, 2, 1, 3)                    # (B,T,42,2)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        return base + torch.cat([d[:, :, :N_BINS, 0], d[:, :, :N_BINS, 1], d[:, :, N_BINS:, 0]], -1)


torch.manual_seed(0)
m = (FullJointTraj if ARCH == "fulljoint" else BinTF)(D, NL).to(dev)
log(f"{ARCH} d={D} layers={NL} params {sum(p.numel() for p in m.parameters())/1e6:.2f}M -> {OUT}")
SvI, CvI, SvB, CvB = load_test_sets()
if MULTI: TESTS = load_all_tests(); TRAIN_SETS = load_multiset()
else: Sin, Cin, Sbg, Cbg = load_train_sets()
EPOCHS = 1 if SMOKE else 30; BS = 64
mix = MultiMixer(TRAIN_SETS, BS) if MULTI else Mixer(Sin, Cin, Sbg, Cbg, BS)
bi = list(range(16)) + list(range(N_BINS, N_BINS + 16))
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); best = 1e9
f6 = lambda s, c: norm.denorm_state(m(s, c)[:, -1])
for ep in range(EPOCHS):
    m.train()
    for st in range(mix.steps_per_epoch):
        sn, cn = mix.next(); pred = m(sn[:, 0], cn); tgt = sn[:, 1:]
        loss = ((pred - tgt) ** 2).mean() + totals_loss(pred, tgt)
        if BURSTW: loss = loss + BURSTW * ((pred[..., bi] - tgt[..., bi]) ** 2).mean()
        if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sched.step()
    if (ep + 1) % 5 == 0 or ep == EPOCHS - 1 or SMOKE:
        m.eval(); mi = eval_traj_model(f6, SvI, CvI); mb = eval_traj_model(f6, SvB, CvB)
        log(f"ep{ep+1:02d} INJ {fmt(mi)}"); log(f"ep{ep+1:02d} BG  {fmt(mb)}")
        if MULTI: eval_all_sets(f6, TESTS, f"ep{ep+1:02d}")
        sc = mi["totN90"] + mi["totM90"]
        if sc < best: best = sc; torch.save({"model": m.state_dict(), "ep": ep + 1, "arch": ARCH, "d": D, "nl": NL}, f"{OUT}/best.pt"); log("   saved best")
m.eval(); s = torch.randn(4096, 82, device=dev); c = torch.randn(4096, 13, 6, device=dev)
log(f"speed: fp32 {speed_bench(lambda b: m(s[:b], c[:b])):.1f} us/cell | bf16-autocast {speed_bench(lambda b: m(s[:b], c[:b]), dtype=torch.bfloat16):.1f} us/cell")
log("DONE")
