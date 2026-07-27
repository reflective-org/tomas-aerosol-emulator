"""Physical constraints for the NON-AR axial trajectory operator.

WHY THIS IS NOT A PORT OF model_conserved_v2.ConservativeMultiHeadMLPV2
-----------------------------------------------------------------------
The AR-MLP champion enforced the sulfur EQUALITY budget

    sum(Mk)[t+dt] + Gc[t+dt] = sum(Mk)[t] + H2SO4_kg_forced[t]        (E)

with start-of-interval forcing. Measured on 800 held-out v4 scenarios
(eval_axial_conserved.py / eval_cons_variants.py / eval_cons_gated.py):

  * (E) closes on GT to p50 0.0016 / p90 0.039 -- but p99 2405.
  * total sulfur is BIMODAL between reservoirs: gas share p50 1e-4, p90 0.9985.
    When the gas dominates, (E) says nothing about the aerosol AND is wrong,
    because the gas is re-prescribed every minute so Gc[t+30] tracks H[t+30],
    not H[t].
  * when the aerosol dominates, (E) forces the 30-min mass gain to equal ONE
    minute of injected gas. Truth: median dM/H_single = 0.43 while only 1.4% of
    the 30 minutes' injected gas actually condenses -- the equality is
    "accurate" only in the trivial sense that sum(Mk) barely moves.
  * consequently, applying (E) post-hoc to aug5, chained across the 12 horizons,
    holds the invariant to 1e-6 and wrecks the mass: totM p90 0.034 -> 0.437,
    Mk p90 0.209 -> 0.347. Gating by regime, damping, and tightening the clamp
    all still land at totM p90 0.09-0.17. Putting the correction entirely in the
    gas keeps the aerosol (Mk p90 0.210) but destroys Gc by catastrophic
    cancellation (p90 1e7).

The AR champion could use (E) only because it TRAINED with it and learned to
compensate. Post-hoc on a model trained without it, (E) injects its own error.

WHAT IS ACTUALLY EXACT, AND IS WHAT THIS MODULE ENFORCES
---------------------------------------------------------
(1) MONOTONICITY.  There is no deposition in this box model and coagulation
    conserves mass, so aerosol sulfate is non-decreasing:

        sum(Mk)[t+dt] >= sum(Mk)[t]                                    (M)

    Verified on GT 30-min windows: holds for 99.24% of (scenario,horizon) pairs
    and every violation is float32 storage noise (max |dM|/M = 1.0e-6).
    Unlike (E) this is one-sided, needs no forcing reconstruction, is valid at
    ANY dt, and is INERT whenever the model is already physical -- so it cannot
    damage accuracy the way (E) does. It is also precisely the failure mode that
    matters when chaining: spurious sulfur LOSS compounding window over window.

(2) BIN GEOMETRY.  TOMAS mass-doubling grid: bin k holds particles of dry mass
    in [XK0*2^k, XK0*2^(k+1)], so XK[k] <= Mk[k]/Nk[k] <= 2*XK[k]. Carried over
    from model_conserved_v2 (lines 164-168), slacked to avoid zero gradients.
    Matters most when a prediction is fed back as an initial condition.

The equality budget is still available (mode="equality") for ablation, and as a
SOFT training term via sulfur_budget_penalty() -- teaching the model the budget
costs nothing at inference, unlike enforcing it.
"""

import math

import torch
import torch.nn as nn

BOXVOL = 1e6
KB = 1.380649e-23
MW_H2SO4 = 98e-3
AVOGADRO = 6.022e23

NK_SLICE = slice(0, 40)
MK_SLICE = slice(40, 80)
GC_IDX = 80

COND_PRES = 0       # log Pa
COND_T = 2          # K (linear)
COND_H2SO4_VMR = 4  # log mol/mol

RHO_SO4 = 1770.0
D0 = 1.7e-9
XK0 = (math.pi / 6.0) * D0 ** 3 * RHO_SO4  # ~4.553e-24 kg


class ConservativeAxialTraj(nn.Module):
    """Wrap an AxialTrajOperator with physically exact constraints.

    Args:
        base: AxialTrajOperator; forward(state0_norm (B,81), cond_seq (B,T+1,C)) -> (B,T,81).
        norm_stats: NormStatsV4 belonging to the wrapped run (aug5 has its own file).
        mode: "mono" (default; monotonicity + bin geometry), "geom" (bin geometry
            only), "equality" (the refuted chained sulfur equality, for ablation),
            or "none".
        n_iter: alternating-projection passes. Default 2 (mono+geom converge fast).
        bin_slack: log-space slack on the geometry clamp. Default log(1.25).
        mono_tol: relative tolerance below which a dip in sum(Mk) is treated as
            storage noise and left alone. Default 1e-6 (the measured GT floor).
    """

    def __init__(self, base, norm_stats, mode="mono", n_iter=2,
                 bin_slack=math.log(1.25), mono_tol=1e-6, log_factor_clamp=2.302585):
        super().__init__()
        self.base = base
        self.mode = mode
        self.n_iter = n_iter
        self.mono_tol = mono_tol
        self.log_factor_clamp = log_factor_clamp

        self.register_buffer("state_mean", norm_stats.state_mean.clone())
        self.register_buffer("state_std", norm_stats.state_std.clone())
        self.register_buffer("cond_mean", norm_stats.cond_mean.clone())
        self.register_buffer("cond_std", norm_stats.cond_std.clone())

        log_const = torch.log(torch.tensor(
            1e-6 * BOXVOL * MW_H2SO4 / AVOGADRO / KB, dtype=torch.float64)).float()
        self.register_buffer("log_h2so4_const", log_const)

        k_idx = torch.arange(40, dtype=torch.float32)
        log_xk = math.log(XK0) + k_idx * math.log(2.0)
        self.register_buffer("log_bin_lower", log_xk - bin_slack)
        self.register_buffer("log_bin_upper", log_xk + math.log(2.0) + bin_slack)

    # ---- forcing reconstruction (shared with the soft penalty) ----
    def log_h2so4_kg(self, cond_norm_1):
        """(B,C) normalized cond at one anchor -> (B,) log kg/box of forced H2SO4."""
        nc = self.cond_mean.shape[0]
        cp = cond_norm_1[:, :nc] * self.cond_std + self.cond_mean
        return (cp[:, COND_H2SO4_VMR] + cp[:, COND_PRES]
                - torch.log(cp[:, COND_T]) + self.log_h2so4_const)

    def _geom_clamp(self, log_Mk, log_Nk):
        mm = log_Mk - log_Nk
        mm = torch.maximum(mm, self.log_bin_lower)
        mm = torch.minimum(mm, self.log_bin_upper)
        return log_Nk + mm

    # ---- the projection ----
    def project_traj(self, pred_norm, state0_norm, cond_seq_norm, return_diag=False):
        """pred_norm (B,T,81) normalized -> constrained (B,T,81) normalized."""
        if self.mode == "none":
            return (pred_norm, None) if return_diag else pred_norm

        s0_log = state0_norm * self.state_std + self.state_mean
        log_Mk_sum_prev = torch.logsumexp(s0_log[:, MK_SLICE], dim=-1)   # running floor

        T = pred_norm.shape[1]
        outs, lifts = [], []
        for k in range(T):
            p = pred_norm[:, k, :] * self.state_std + self.state_mean
            log_Nk = p[:, NK_SLICE]
            log_Mk = p[:, MK_SLICE]
            log_Gc = p[:, GC_IDX]

            if self.mode == "equality":
                log_H = self.log_h2so4_kg(cond_seq_norm[:, k, :])
                target = torch.logsumexp(
                    torch.stack([log_Mk_sum_prev, log_H], dim=-1), dim=-1)
                for _ in range(self.n_iter):
                    cur = torch.logsumexp(
                        torch.cat([log_Mk, log_Gc.unsqueeze(-1)], dim=-1), dim=-1)
                    f = (target - cur).clamp(-self.log_factor_clamp, self.log_factor_clamp)
                    log_Mk = log_Mk + f.unsqueeze(-1)
                    log_Gc = log_Gc + f
                    log_Mk = self._geom_clamp(log_Mk, log_Nk)
                lift = torch.zeros_like(log_Gc)
            else:
                lift = torch.zeros_like(log_Gc)
                # geometry first, monotonicity LAST: the two can conflict (if every
                # bin sits at its mass ceiling, sum(Mk) cannot rise without moving
                # particles between bins, which this projection does not do). The
                # geometry clamp carries 25% slack, so when they conflict we spend
                # that slack rather than leave sulfur destroyed.
                for _ in range(self.n_iter):
                    log_Mk = self._geom_clamp(log_Mk, log_Nk)
                    if self.mode == "mono":
                        # (M): lift sum(Mk) back to the running floor if it dipped.
                        # A uniform log-shift preserves the predicted SHAPE of the
                        # distribution and is a no-op when the prediction already
                        # satisfies the constraint.
                        cur = torch.logsumexp(log_Mk, dim=-1)
                        deficit = log_Mk_sum_prev - cur
                        deficit = torch.where(
                            deficit > math.log1p(self.mono_tol),
                            deficit, torch.zeros_like(deficit))
                        deficit = deficit.clamp(max=self.log_factor_clamp)
                        log_Mk = log_Mk + deficit.unsqueeze(-1)
                        lift = lift + deficit

            out_log = torch.cat([log_Nk, log_Mk, log_Gc.unsqueeze(-1)], dim=-1)
            outs.append((out_log - self.state_mean) / self.state_std)
            lifts.append(lift)
            # running floor advances with the CONSTRAINED sum, so monotonicity
            # holds across the whole trajectory, not just neighbouring pairs
            log_Mk_sum_prev = torch.logsumexp(out_log[:, MK_SLICE], dim=-1)

        out = torch.stack(outs, dim=1)
        if return_diag:
            return out, torch.stack(lifts, dim=1)
        return out

    def forward(self, state0_norm, cond_seq_norm):
        pred = self.base(state0_norm, cond_seq_norm)
        return self.project_traj(pred, state0_norm, cond_seq_norm)


def sulfur_budget_penalty(pred_norm, state0_norm, cond_seq_norm, norm_stats,
                          log_h2so4_const, huber_beta=0.1):
    """SOFT conservation term for training.

    Penalizes, in log space, the two exact physical facts:
      (a) monotonicity   : sum(Mk)[k] >= sum(Mk)[k-1]   (one-sided)
      (b) gas ceiling    : sum(Mk)[k] - sum(Mk)[k-1] <= gas injected over the
                           window (one-sided; the window's forcing anchors)

    Both are INEQUALITIES, so a model that is already physical pays nothing --
    unlike a squared-error term on the equality budget, which would fight the
    data. Returns a scalar.
    """
    sm = norm_stats.state_mean.to(pred_norm.device)
    ss = norm_stats.state_std.to(pred_norm.device)
    p = pred_norm * ss + sm                                   # (B,T,81) log
    s0 = state0_norm * ss + sm
    log_sum = torch.logsumexp(p[:, :, MK_SLICE], dim=-1)      # (B,T)
    log_sum0 = torch.logsumexp(s0[:, MK_SLICE], dim=-1, keepdim=True)
    prev = torch.cat([log_sum0, log_sum[:, :-1]], dim=1)      # (B,T)

    # (a) hinge on a DROP in log sum(Mk)
    drop = torch.clamp(prev - log_sum, min=0.0)

    # (b) hinge on exceeding the injected gas: sum(Mk)[k] <= prev + n_min*H
    nc = norm_stats.cond_mean.shape[0]
    cp = cond_seq_norm[:, :-1, :nc] * norm_stats.cond_std.to(p.device) + norm_stats.cond_mean.to(p.device)
    log_H = (cp[..., COND_H2SO4_VMR] + cp[..., COND_PRES]
             - torch.log(cp[..., COND_T]) + log_h2so4_const)  # (B,T)
    n_min = 30.0
    ceil_log = torch.logsumexp(
        torch.stack([prev, log_H + math.log(n_min)], dim=-1), dim=-1)
    over = torch.clamp(log_sum - ceil_log, min=0.0)

    h = lambda x: torch.where(x < huber_beta, 0.5 * x * x / huber_beta, x - 0.5 * huber_beta)
    return h(drop).mean() + h(over).mean()
