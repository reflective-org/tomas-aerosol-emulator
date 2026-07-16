"""Δt (time-skip) Fourier conditioning features for the Δt-conditioned emulator.

The network takes, alongside the 5 atmospheric forcing features, an 8-dim
Fourier encoding of the step size Δt (in minutes). One trained network then
serves every step size it saw in training (Δt = 1…30 min): pick a Δt at
inference and the model advances the state that far in a single forward pass.
"""
import math

import torch

N_STATE = 81

# The shipped dt_champion checkpoint was trained on the 5 original forcing
# features (pres, nh3_conc, T, RH, H2SO4_vmr). dataset_v4.py has since grown a
# 6th (fion, for the axial OOD work) — slice cond to [..., :N_COND_BASE]
# before normalizing with this model's norm_stats.
N_COND_BASE = 5

# Fourier features for Δt: 4 octaves × {sin, cos} = 8 dims.
FOURIER_OCTAVES = [-2.0, -1.0, 0.0, 1.0]
N_DT_FEAT = 2 * len(FOURIER_OCTAVES)            # 8
N_COND_TOTAL = N_COND_BASE + N_DT_FEAT          # 13


def fourier_dt_feat(d_minutes):
    """d_minutes: (B,) tensor in minutes -> (B, N_DT_FEAT) Fourier features.

    Uses log-scale: sinusoidal in log(d) at multiple octaves. Same form as
    NeRF/positional encodings; gives the network sharper discrimination across
    Δt than a single scalar.
    """
    log_d = torch.log(d_minutes.float().clamp(min=1e-6))         # (B,)
    octaves = torch.tensor(FOURIER_OCTAVES, dtype=torch.float32,
                           device=d_minutes.device)              # (K,)
    scaled = (math.tau * log_d.unsqueeze(-1) * (2.0 ** octaves))  # (B, K)
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (B, 2K)
