"""Moments + bin-geometry projected multihead MLP for TOMAS-v4.

Extends `model_conserved.ConservativeMultiHeadMLP` with one extra constraint:

  Per-bin physical sanity:  XK[k] <= Mk_pred[k] / Nk_pred[k] <= 2*XK[k]
  (TOMAS doubling-mass bin grid: bin k holds particles of dry mass in
  [XK0 * 2^k, XK0 * 2^(k+1)] kg, so the mean particle mass in the bin
  must lie inside that range.)

Conservation projections applied (alternating, n_iter passes; n_iter=3 is
enough in practice for both constraints to hold to <1% after iteration):

  (a) Per-bin clamp: log_Mk - log_Nk clamped to [log XK[k], log(2*XK[k])].
  (b) Total sulfur: uniform additive log-shift on (Mk, Gc) so
        ΣMk + Gc = ΣMk_in + H2SO4_kg_forced  (same as v1).

The original v1 projection is just (b) with a uniform shift — which preserves
log-bin shape but cannot fix per-bin error if the raw model puts mass in the
wrong bin. (a) is what bites the network for predicting (Nk, Mk) pairs that
are off-grid (e.g. predicting 10x more sulfate mass in a bin than its
particles can physically hold).

Verified-in-spirit: alternating projection is the standard tool for
inequality + equality constraints. Single pass works if very few bins are
out-of-bound; we default n_iter=3 which converges in tests.
"""

import math

import torch
import torch.nn as nn

from model_multihead import TomasMultiHeadMLP

BOXVOL = 1e6
KB = 1.380649e-23
MW_H2SO4 = 98e-3
AVOGADRO = 6.022e23

NK_SLICE = slice(0, 40)
MK_SLICE = slice(40, 80)
GC_IDX = 80

COND_PRES = 0
COND_T = 2
COND_H2SO4_VMR = 4

# TOMAS bin grid (must match generate_training_data_v4.py):
#   XK0 = (pi/6) * d0^3 * rho_so4 with d0=1.7e-9 m, rho=1770 kg/m^3
#   XK[k] = XK0 * 2^k for k=0..40
RHO_SO4 = 1770.0
D0 = 1.7e-9
XK0 = (math.pi / 6.0) * D0 ** 3 * RHO_SO4  # ~4.553e-24 kg


class ConservativeMultiHeadMLPV2(nn.Module):
    """v1 sulfur-conservation projection + per-bin geometry clamp.

    Args:
        base: instantiated TomasMultiHeadMLP.
        norm_stats: NormStatsV4.
        log_factor_clamp: clamp the per-iteration additive log-factor for
            stability. Default ln(10)≈2.30 -> factor∈[0.1, 10]. None disables.
        n_iter: alternating-projection iterations between (clamp, shift).
            Default 3.
        bin_slack: extra slack on the bin-mass clamp in log-space (additive).
            Default math.log(1.25) so the clamp is [XK[k]/1.25, 2*XK[k]*1.25],
            avoiding hard saturation that produces zero gradient.
    """

    def __init__(self, base: TomasMultiHeadMLP, norm_stats,
                 log_factor_clamp=2.302585,
                 n_iter=3,
                 bin_slack=math.log(1.25),
                 total_n_log_clamp=None):
        """
        total_n_log_clamp: if not None, applies a per-step rate cap (deadzone
            projection) on log(total_N). NOTE: this proved to NOT work for
            TOMAS — at 2026-05-25 we found gt has legitimate per-step
            log(total_N) changes up to +18 (10^8 factor) during nucleation
            bursts, so any cap small enough to catch tail outliers also
            clamps legitimate physics. Kept in the code as a knob but not
            recommended; prefer a training-time soft constraint instead.
            None disables (default).
        """
        super().__init__()
        self.base = base
        self.log_factor_clamp = log_factor_clamp
        self.n_iter = n_iter
        self.total_n_log_clamp = total_n_log_clamp

        self.register_buffer("state_mean", norm_stats.state_mean.clone())
        self.register_buffer("state_std", norm_stats.state_std.clone())
        self.register_buffer("cond_mean", norm_stats.cond_mean.clone())
        self.register_buffer("cond_std", norm_stats.cond_std.clone())

        log_const = torch.log(torch.tensor(
            1e-6 * BOXVOL * MW_H2SO4 / AVOGADRO / KB,
            dtype=torch.float64,
        )).float()
        self.register_buffer("log_h2so4_const", log_const)

        # Per-bin log-mass bounds for Mk/Nk:
        #   log Mk[k] - log Nk[k] in [log XK[k] - slack, log(2*XK[k]) + slack]
        k_idx = torch.arange(40, dtype=torch.float32)
        log_xk = math.log(XK0) + k_idx * math.log(2.0)
        self.register_buffer("log_bin_lower", log_xk - bin_slack)             # (40,)
        self.register_buffer("log_bin_upper", log_xk + math.log(2.0) + bin_slack)

    def forward(self, x):
        state_norm_in = x[:, :81]
        # Take only the real conditioning dims (length == cond_mean). A
        # Δt-conditioned base appends extra Fourier-Δt dims after the cond;
        # those must NOT be fed to the forcing reconstruction. For the legacy
        # 86-dim input (81 state + 5 cond) this slice is identical to x[:,81:].
        nc = self.cond_mean.shape[0]
        cond_norm = x[:, 81:81 + nc]

        pred_norm = self.base(x)  # (B, 81)

        # ---- Input state in log-space ----
        state_log_in = state_norm_in * self.state_std + self.state_mean
        log_Mk_in = state_log_in[:, MK_SLICE]
        log_Mk_sum_in = torch.logsumexp(log_Mk_in, dim=-1)              # (B,)
        log_Nk_in = state_log_in[:, NK_SLICE]
        log_Nk_sum_in = torch.logsumexp(log_Nk_in, dim=-1)              # (B,) — anchor for total-N

        # ---- H2SO4_kg_forced from cond ----
        cond_phys = cond_norm * self.cond_std + self.cond_mean
        log_pres = cond_phys[:, COND_PRES]
        T = cond_phys[:, COND_T]
        log_H2SO4_vmr = cond_phys[:, COND_H2SO4_VMR]
        log_H2SO4_kg_forced = (
            log_H2SO4_vmr + log_pres - torch.log(T) + self.log_h2so4_const
        )
        log_target_sum = torch.logsumexp(
            torch.stack([log_Mk_sum_in, log_H2SO4_kg_forced], dim=-1), dim=-1
        )                                                               # (B,)

        # ---- Raw model output, log-space ----
        pred_log = pred_norm * self.state_std + self.state_mean
        log_Nk_pred = pred_log[:, NK_SLICE]                             # (B, 40)
        log_Mk = pred_log[:, MK_SLICE]                                  # (B, 40)
        log_Gc = pred_log[:, GC_IDX]                                    # (B,)

        # ---- Alternating projection ----
        # Shift first, clamp last on each iteration so the FINAL operation
        # is the per-bin physics clamp (otherwise the shift can push bins
        # back out of the bin-mass range -- worst when log_factor is large,
        # e.g. early in training).
        for _ in range(self.n_iter):
            # (a) Total-sulfur log-shift on (Mk, Gc) -- v1 projection
            log_pred_sum = torch.logsumexp(
                torch.cat([log_Mk, log_Gc.unsqueeze(-1)], dim=-1), dim=-1,
            )                                                           # (B,)
            log_factor = log_target_sum - log_pred_sum
            if self.log_factor_clamp is not None:
                log_factor = log_factor.clamp(
                    -self.log_factor_clamp, self.log_factor_clamp
                )
            log_Mk = log_Mk + log_factor.unsqueeze(-1)
            log_Gc = log_Gc + log_factor

            # (b) Per-bin geometry clamp on log(Mk/Nk)
            log_mean_mass = log_Mk - log_Nk_pred                        # (B, 40)
            log_mean_mass = torch.maximum(log_mean_mass, self.log_bin_lower)
            log_mean_mass = torch.minimum(log_mean_mass, self.log_bin_upper)
            log_Mk = log_Nk_pred + log_mean_mass

            # (c) Optional: total-N per-step rate CAP (deadzone projection).
            # Bounds the magnitude of per-step change in total_N: when the
            # natural pred satisfies |log(pred_total) - log(input_total)| <=
            # total_n_log_clamp, the projection is a NO-OP. When the natural
            # pred exceeds that, it's pulled back to the boundary of the
            # allowed band. This catches pathological single-step jumps (e.g.
            # the 6h p99 = 2e6 outlier in p7) WITHOUT fighting normal
            # coag/nucleation dynamics (which produce small per-step changes).
            if self.total_n_log_clamp is not None:
                log_Nk_sum_pred = torch.logsumexp(log_Nk_pred, dim=-1)   # (B,)
                delta = log_Nk_sum_pred - log_Nk_sum_in                  # natural per-step log-change
                delta_clamped = delta.clamp(
                    -self.total_n_log_clamp, self.total_n_log_clamp
                )
                shift = delta_clamped - delta                             # 0 inside the band; pulls back outside
                log_Nk_pred = log_Nk_pred + shift.unsqueeze(-1)
                # Re-apply per-bin clamp because Nk shifted but Mk did not.
                log_mean_mass = log_Mk - log_Nk_pred
                log_mean_mass = torch.maximum(log_mean_mass, self.log_bin_lower)
                log_mean_mass = torch.minimum(log_mean_mass, self.log_bin_upper)
                log_Mk = log_Nk_pred + log_mean_mass

        # ---- Reassemble and renormalize ----
        log_state_proj = torch.cat([
            log_Nk_pred,
            log_Mk,
            log_Gc.unsqueeze(-1),
        ], dim=-1)
        return (log_state_proj - self.state_mean) / self.state_std
