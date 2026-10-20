"""Bin-token transformer ("BinTF") for 6-hour TOMAS microphysics emulation.

One grid box is a sequence of 42 tokens: 40 size bins (number, sulfate mass) plus
two gas tokens (H2SO4, SO2). Each token also receives the gases and the whole
6-hour forcing window (12 x 6 values) so the attention stack sees the complete
problem. The head emits 12 outputs per token, one per 30-minute step, as a
residual on the initial state. Self-attention runs across bins only, which is
what makes the model ~3 us per cell: the 12 time steps live in the channel
dimension, not in the sequence.

The class is identical to the one used in training (training/frontier_transformers.py).
"""
import torch
import torch.nn as nn

N_BINS = 40           # TOMAS size bins (mass doubling, 1.7 nm .. ~10 um)
N_COND = 6            # forcing channels per 30-min point
N_STATE = 2 * N_BINS + 2   # 82 = Nk (40) + Mk (40) + H2SO4 + SO2
P = N_BINS + 2        # tokens per box
T = 12                # 30-min outputs per 6-hour block


def enc(d, nl, nh=4):
    layer = nn.TransformerEncoderLayer(d, nh, 2 * d, dropout=0.0, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, nl)


class BinTF(nn.Module):
    """42 tokens; token = [Nk0, Mk0] + gases + all 12 forcing vectors (72) -> d; output 12 x 2 per token."""

    def __init__(self, d, nl):
        super().__init__()
        self.tok_in = nn.Linear(2 + 2 + 6 * T, d)
        self.bin_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        self.enc = enc(d, nl)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 * T)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq):
        """s0: (B, 82) normalized log-state at t=0.
        cond_seq: (B, 13, 6) normalized forcing at t = 0, 0.5, ..., 6 h (both endpoints).
        Returns (B, 12, 82): normalized log-states at t = 0.5, 1, ..., 6 h."""
        B = s0.shape[0]
        nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]
        gas = s0[:, 2 * N_BINS:]
        tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)   # (B,42,2)
        glob = torch.cat([gas, cond_seq[:, 1:1 + T].flatten(1)], 1)                                      # (B,2+72)
        x = self.tok_in(torch.cat([tok, glob.unsqueeze(1).expand(B, P, -1)], -1)) + self.bin_emb
        d = self.head(self.norm(self.enc(x))).reshape(B, P, T, 2).permute(0, 2, 1, 3)                    # (B,T,42,2)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        return base + torch.cat([d[:, :, :N_BINS, 0], d[:, :, :N_BINS, 1], d[:, :, N_BINS:, 0]], -1)


# Shipped checkpoints: directory -> (d, layers)
CHECKPOINTS = {
    "bintf_d384_l6": (384, 6),   # 7.16 M parameters, production model
    "bintf_d256_l4": (256, 4),   # 2.15 M parameters, fast model
}
