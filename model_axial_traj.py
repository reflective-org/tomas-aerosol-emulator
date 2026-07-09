"""Non-autoregressive AXIAL (time x bin) trajectory operator for the TOMAS emulator.

Designed from the failure modes of prior non-AR attempts:
 - Probe B (runs_traj_p1) failed: from-scratch, CAUSAL, TIME-ONLY attention over whole-state
   tokens -> no bin coupling, representational deficit even at 1 step.
 - Corrector (runs_corrector_p3) worked (0.42->0.30) via non-AR BIDIRECTIONAL global view, but
   was CAPPED by correcting the champion's AR stability-taxed trajectory.

This model: STANDALONE (generates the 12-state Δt=30 trajectory from state0 + full forcing, no AR
feedback -> kills the dominant rollout-amplification error) with AXIAL attention:
 - BIN-axis attention (within each timestep, over 41 tokens = 40 size bins + gas): the all-pairs
   coupling that matches coagulation's NiNj structure (the confirmed bin-0 driver).
 - TIME-axis attention (within each bin, over the 12 steps, BIDIRECTIONAL): non-AR temporal
   composition with the full forcing in view.
Persistence-residual (zero-init head) -> starts at the identity, a good basin (avoids the
from-scratch Jensen trap). Optional per-step-physics warm-up handled by the trainer (curriculum).
"""
import torch
import torch.nn as nn

N_STATE = 81
N_BINS = 40
N_COND = 6
N_OUT = 12
DT_MIN = 30


class MoEFF(nn.Module):
    """Gated mixture-of-experts FFN: a gate routes each token to E expert MLPs. The gate can
    switch sharply between regimes (e.g. calm vs burst) -> threshold-friendly inductive bias."""
    def __init__(self, d, ff, E):
        super().__init__()
        self.gate = nn.Linear(d, E)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) for _ in range(E)])
    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=-1)                  # (..., E)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (..., E, d)
        return (g.unsqueeze(-1) * outs).sum(-2)


class AxialBlock(nn.Module):
    def __init__(self, d, nhead, ff, dropout, ff_experts=1):
        super().__init__()
        self.bin_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.time_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
        self.ff = (MoEFF(d, ff, ff_experts) if ff_experts > 1
                   else nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)))

    def forward(self, x):                       # x: (B, T, P, d)
        B, T, P, d = x.shape
        h = self.ln1(x).reshape(B * T, P, d)    # bin attention (over P tokens within each time)
        a, _ = self.bin_attn(h, h, h); x = x + a.reshape(B, T, P, d)
        h = self.ln2(x).permute(0, 2, 1, 3).reshape(B * P, T, d)   # time attention (bidirectional)
        a, _ = self.time_attn(h, h, h); x = x + a.reshape(B, P, T, d).permute(0, 2, 1, 3)
        x = x + self.ff(self.ln3(x))
        return x


class AxialTrajOperator(nn.Module):
    def __init__(self, d_model=192, nhead=6, num_blocks=5, dim_ff=512, dropout=0.0, n_out=N_OUT, ff_experts=1, n_cond=N_COND):
        super().__init__()
        self.n_out = n_out; self.P = N_BINS + 1          # 40 size bins + 1 gas token
        self.tok_in = nn.Linear(2, d_model)              # per-token (Nk,Mk) for bins / (Gc,0) for gas
        self.cond_in = nn.Linear(n_cond, d_model)        # forcing (+optional burst feats) at each output time
        self.bin_emb = nn.Parameter(torch.zeros(self.P, d_model))
        self.time_emb = nn.Parameter(torch.zeros(n_out, d_model))
        self.blocks = nn.ModuleList([AxialBlock(d_model, nhead, dim_ff, dropout, ff_experts) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)                # (dNk, dMk) per token
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # persistence-residual
        nn.init.normal_(self.bin_emb, std=0.02); nn.init.normal_(self.time_emb, std=0.02)

    def forward(self, state0_norm, cond_seq_norm):
        """state0_norm: (B,81). cond_seq_norm: (B,13,5) at minutes [0,30,...,360];
        we use the 12 OUTPUT times cond[:,1:13]. Returns (B,12,81)."""
        B = state0_norm.shape[0]
        Nk0 = state0_norm[:, :40]; Mk0 = state0_norm[:, 40:80]; Gc0 = state0_norm[:, 80:81]
        bin_feat = torch.stack([Nk0, Mk0], dim=-1)                       # (B,40,2)
        gas_feat = torch.cat([Gc0, torch.zeros_like(Gc0)], dim=-1).unsqueeze(1)  # (B,1,2)
        tok0 = torch.cat([bin_feat, gas_feat], dim=1)                    # (B,41,2)
        x = self.tok_in(tok0)                                            # (B,41,d)
        x = x.unsqueeze(1).expand(B, self.n_out, self.P, x.shape[-1]).clone()  # (B,T,41,d)
        x = x + self.bin_emb.view(1, 1, self.P, -1) + self.time_emb.view(1, self.n_out, 1, -1)
        x = x + self.cond_in(cond_seq_norm[:, 1:1 + self.n_out, :]).unsqueeze(2)   # forcing per time
        for blk in self.blocks:
            x = blk(x)
        delta = self.head(self.norm(x))                                  # (B,T,41,2)
        dNk = delta[:, :, :40, 0]; dMk = delta[:, :, :40, 1]; dGc = delta[:, :, 40, 0:1]
        return torch.cat([Nk0.unsqueeze(1) + dNk, Mk0.unsqueeze(1) + dMk, Gc0.unsqueeze(1) + dGc], dim=-1)
