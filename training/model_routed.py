"""BinRoutedHybrid: mix-and-match-per-bin, non-AR trajectory operator.

Motivation (from the bake-off): per-bin difficulty is LOCALIZED. The
accumulation/coarse bins are well-conditioned and a cheap local model nails
them; the nucleation-burst bins (0..BURST_HI) are hard and need GLOBAL bin
coupling (coagulation is all-pairs) — which is exactly where local convolution
failed and attention wins.

So route by bin:
  * BACKBONE (all 42 tokens): per-token MLP + BinConv(local) + TimeConv. Cheap,
    trainable, handles the easy bins and local condensation growth everywhere.
  * SPECIALIST (burst tokens 0..BURST_HI + a global-context token): bin-axis
    self-attention. Global coupling, but over ~BURST_HI+1 tokens, not 42, so it
    stays cheap. The global token lets the burst bins see the condensation sink
    carried by the accumulation mode.
  * BLEND: a smooth ramp over the boundary bins fuses specialist->backbone so
    there is no hard seam (the bin-25 multi-head lesson).

forward(state0 (B,82), cond_seq (B,13,6)) -> (B,12,82), residual/identity init.
"""
import torch
import torch.nn as nn

N_BINS = 40
N_STATE = 82


class BinConv(nn.Module):
    def __init__(self, d, k=5):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, d, k, padding=k // 2)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2)
        self.act = nn.GELU()

    def forward(self, x):                            # (B,T,P,d)
        B, T, P, d = x.shape
        h = self.ln(x).permute(0, 1, 3, 2).reshape(B * T, d, P)
        h = self.c2(self.act(self.c1(h)))
        return x + h.reshape(B, T, d, P).permute(0, 1, 3, 2)


class TimeConv(nn.Module):
    def __init__(self, d, k=5):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, d, k, padding=k // 2)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2)
        self.act = nn.GELU()

    def forward(self, x):                            # (B,T,P,d)
        B, T, P, d = x.shape
        h = self.ln(x).permute(0, 2, 3, 1).reshape(B * P, d, T)
        h = self.c2(self.act(self.c1(h)))
        return x + h.reshape(B, P, d, T).permute(0, 3, 1, 2)


class TokFF(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        return x + self.net(self.ln(x))


class BackboneBlock(nn.Module):
    def __init__(self, d, ff, k_bin, k_time):
        super().__init__()
        self.bin = BinConv(d, k_bin)
        self.time = TimeConv(d, k_time)
        self.ff = TokFF(d, ff)

    def forward(self, x):
        return self.ff(self.time(self.bin(x)))


class BurstAttn(nn.Module):
    """Bin-axis self-attention over [burst tokens + 1 global-context token],
    applied per time step. Refines the burst-token representation with global
    (all-bin) context. Cheap: O((BH+1)^2) with BH ~ 12."""
    def __init__(self, d, nhead, ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, burst, gctx):                  # (B,T,BH,d), (B,T,1,d)
        B, T, BH, d = burst.shape
        h = torch.cat([burst, gctx], dim=2)          # (B,T,BH+1,d)
        hn = self.ln1(h).reshape(B * T, BH + 1, d)
        a, _ = self.attn(hn, hn, hn)
        h = h + a.reshape(B, T, BH + 1, d)
        h = h + self.ff(self.ln2(h))
        return h[:, :, :BH]                          # refined burst tokens


class BinRoutedHybrid(nn.Module):
    def __init__(self, d=112, backbone_blocks=3, burst_blocks=2, nhead=4,
                 ff=256, k_bin=5, k_time=5, burst_hi=12, blend_lo=8,
                 n_out=12, n_cond=6, n_phys=0):
        super().__init__()
        self.n_out = n_out
        self.P = N_BINS + 2
        self.BH = burst_hi
        self.n_phys = n_phys
        # smooth blend ramp: 1 for bins < blend_lo, ->0 by burst_hi (register as buffer)
        w = torch.ones(burst_hi)
        for i in range(blend_lo, burst_hi):
            w[i] = 0.5 * (1 + torch.cos(torch.tensor(
                (i - blend_lo) / max(1, burst_hi - blend_lo) * 3.14159265)))
        self.register_buffer("blend", w.view(1, 1, burst_hi, 1))

        self.tok_in = nn.Linear(2, d)
        self.cond_in = nn.Linear(n_cond, d)
        # broadcast gas state to EVERY bin token: injection is a global gas-driven
        # perturbation; local conv can't carry the 2 gas tokens to far bins.
        self.gas_bcast = nn.Linear(2, d)
        # global pooled context of the whole input distribution -> every token
        self.glob_in = nn.Linear(d, d)
        # physics descriptors (J, CS, GR, CS/GR, ...) -> every token (semi-hybrid)
        if n_phys > 0:
            self.phys_in = nn.Linear(n_phys, d)
        self.bin_emb = nn.Parameter(torch.randn(self.P, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(n_out, d) * 0.02)

        self.backbone = nn.ModuleList(
            [BackboneBlock(d, ff, k_bin, k_time) for _ in range(backbone_blocks)])
        self.gctx = nn.Linear(d, d)                  # global-context projection
        self.specialist = nn.ModuleList(
            [BurstAttn(d, nhead, ff) for _ in range(burst_blocks)])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq, phys=None):
        B = s0.shape[0]
        nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]
        gas = s0[:, 2 * N_BINS:]                      # (B,2)
        tok = torch.cat([torch.stack([nk, mk], -1),
                         torch.stack([gas, torch.zeros_like(gas)], -1)], 1)  # (B,42,2)
        tok_e = self.tok_in(tok)                                 # (B,42,d)
        # gas + global-context broadcast to every token (fixes gas-blind far bins)
        gas_e = self.gas_bcast(gas)                              # (B,d)
        glob_e = self.glob_in(tok_e.mean(dim=1))                 # (B,d)
        ctx = gas_e + glob_e
        if self.n_phys > 0 and phys is not None:                 # semi-hybrid
            ctx = ctx + self.phys_in(phys)
        tok_e = tok_e + ctx.unsqueeze(1)
        x = tok_e.unsqueeze(1).expand(B, self.n_out, self.P, -1).clone()
        x = x + self.bin_emb.view(1, 1, self.P, -1) \
              + self.time_emb.view(1, self.n_out, 1, -1)
        x = x + self.cond_in(cond_seq[:, 1:1 + self.n_out]).unsqueeze(2)

        # cheap backbone over all 42 tokens
        for blk in self.backbone:
            x = blk(x)

        # expensive specialist only on burst tokens + global context
        gctx = self.gctx(x.mean(dim=2, keepdim=True))            # (B,T,1,d)
        burst = x[:, :, :self.BH]                                # (B,T,BH,d)
        for blk in self.specialist:
            burst = blk(burst, gctx)
        # soft-blend specialist into backbone across the boundary
        refined = self.blend * burst + (1 - self.blend) * x[:, :, :self.BH]
        x = torch.cat([refined, x[:, :, self.BH:]], dim=2)

        d = self.head(self.norm(x))                              # (B,T,42,2)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        dNk = d[:, :, :N_BINS, 0]
        dMk = d[:, :, :N_BINS, 1]
        dG = d[:, :, N_BINS:, 0]
        return base + torch.cat([dNk, dMk, dG], dim=-1)
