"""TOMAS v4 training v2 -- two changes on top of train_v4_conserved.py:

  1. ConservativeMultiHeadMLPV2: adds per-bin geometry clamp
     (XK[k] <= Mk/Nk <= 2*XK[k]) on top of the v1 total-sulfur projection.

  2. Pushforward: after `chunk_len` rollout-with-gradient steps, take K
     additional steps from a *detached* prediction with gradient only on
     those K extra steps. Forces the network to recover from its own drift
     at multi-step horizons -- targets the bin-7/19 spikes which compound
     across the 360-step eval horizon.

Dataset chunks are sized to (chunk_len + pushforward_steps + 1) so the
extra steps have ground truth available.

Everything else (delta-norm, delta-std floor, regression guard, state clamp)
matches train_v4_conserved.py 1:1.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_v4 import (
    load_v4_batches, build_state_array_v4, transform_cond,
    compute_norm_stats_v4, TomasTrajectoryDatasetV4,
    N_STATE, N_COND,
)
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2


def _ddp_setup():
    """Initialize DDP if launched under torchrun. Returns (world_size, rank, local_rank, is_ddp)."""
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_world_size(), dist.get_rank(), local_rank, True


def _ddp_cleanup(is_ddp):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def _any_rank_true(flag, device, is_ddp):
    """All-reduce a boolean: True if any rank is True. No-op when not DDP."""
    if not is_ddp:
        return flag
    t = torch.tensor(int(flag), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def _broadcast_float(value, device, is_ddp, src=0):
    """Broadcast a scalar float from `src` to all ranks. No-op when not DDP."""
    if not is_ddp:
        return value
    t = torch.tensor(float(value), device=device)
    dist.broadcast(t, src=src)
    return t.item()


def build_bin_weights(gc_weight=1.0, edge_weight=1.0, overrides=None, device=None):
    """Per-state-component loss weights.

    Layout: [Nk_0..Nk_39, Mk_0..Mk_39, Gc] (length 81).

    `overrides` is an optional iterable of (idx, weight) pairs applied *after*
    edge/gc broad strokes — use it to single out specific bins (e.g. the
    bin-25 transition outlier) or to upweight the corresponding Mk index
    (idx+40). The final vector is normalized to mean 1 so adjusting weights
    doesn't also scale overall loss magnitude.
    """
    w = torch.ones(81, device=device)
    if edge_weight != 1.0:
        w[0:6] = edge_weight       # nucleation Nk
        w[40:46] = edge_weight     # nucleation Mk
        w[26:40] = edge_weight     # coarse Nk
        w[66:80] = edge_weight     # coarse Mk
    if gc_weight != 1.0:
        w[80] = gc_weight
    if overrides:
        for idx, wv in overrides:
            w[idx] = float(wv)
    return w / w.mean()


def parse_bin_weight_overrides(spec):
    """Parse "25:5.0,65:5.0" -> [(25, 5.0), (65, 5.0)]. Returns None if empty."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out.append((int(k), float(v)))
    return out if out else None


def rollout_loss(model, state_chunk, cond_chunk,
                 noise_std=0.0,
                 gc_weight=1.0, edge_weight=1.0, bin_weight_overrides=None,
                 delta_norm=False, norm_stats=None,
                 state_clamp=None,
                 grad_steps=None,
                 pushforward_steps=0,
                 pushforward_weight=1.0,
                 relerr_weight=0.0,
                 relerr_clip_max=10.0,
                 phys_floor=None,
                 tail_quantile=0.0,
                 total_n_loss_weight=0.0,
                 total_m_loss_weight=0.0):
    """Roll forward over the chunk.

    - `grad_steps` of in-chunk steps get gradient through the model output.
    - Then `pushforward_steps` extra steps run with *detached* model input;
      their losses are gradient-bearing (gradient flows only through the
      step transitions, not back through the rolled-out history).

    state_chunk: (B, grad_steps + pushforward_steps + 1, 81)
    cond_chunk:  (B, grad_steps + pushforward_steps + 1, N_COND)
    """
    n_total_steps = state_chunk.shape[1] - 1
    if grad_steps is None:
        grad_steps = n_total_steps - pushforward_steps
    assert grad_steps + pushforward_steps == n_total_steps, (
        f"grad_steps({grad_steps}) + pushforward_steps({pushforward_steps}) "
        f"!= chunk steps ({n_total_steps})"
    )

    current = state_chunk[:, 0, :]
    if noise_std > 0:
        current = current + torch.randn_like(current) * noise_std

    use_w = (gc_weight != 1.0) or (edge_weight != 1.0) or bool(bin_weight_overrides)
    w = build_bin_weights(gc_weight, edge_weight, overrides=bin_weight_overrides,
                          device=state_chunk.device) if use_w else None

    if delta_norm and norm_stats is not None and norm_stats.delta_std is not None:
        inv_dvar = (1.0 / norm_stats.delta_std.to(state_chunk.device)) ** 2
        inv_dvar = inv_dvar / inv_dvar.mean()
    else:
        inv_dvar = None

    def step_loss(pred, gt):
        err_sq = (pred - gt) ** 2
        if inv_dvar is not None:
            err_sq = inv_dvar * err_sq
        if w is not None:
            err_sq = w * err_sq
        log_mse = err_sq.mean()

        need_denorm = (
            (relerr_weight > 0 and phys_floor is not None)
            or total_n_loss_weight > 0
            or total_m_loss_weight > 0
        ) and (norm_stats is not None)
        if not need_denorm:
            return log_mse

        # Denormalize log-state → log, then exp to physical units.
        # CRITICAL: clamp the denormalized log-state before exp(), because
        # state_clamp=±20 in NORMALIZED space becomes ±200+ after multiplying
        # by state_std, and exp(200) overflows float32 to inf.
        pred_log = norm_stats.denormalize_state(pred).clamp(min=-60.0, max=60.0)
        gt_log = norm_stats.denormalize_state(gt).clamp(min=-60.0, max=60.0)

        total = (1.0 - relerr_weight) * log_mse if relerr_weight > 0 else log_mse

        if relerr_weight > 0 and phys_floor is not None:
            pred_phys = torch.exp(pred_log)
            gt_phys = torch.exp(gt_log)
            denom = torch.clamp(gt_phys, min=phys_floor)
            rel_err = (pred_phys - gt_phys).abs() / denom
            rel_err = torch.clamp(rel_err, max=relerr_clip_max)
            if w is not None:
                rel_err = w * rel_err
            if 0.0 < tail_quantile < 1.0:
                # Focus on the worst (1-q) fraction of rel-err entries per step.
                flat = rel_err.flatten()
                k = max(1, int(flat.numel() * (1.0 - tail_quantile)))
                top_k, _ = torch.topk(flat, k)
                relerr_loss = top_k.mean()
            else:
                relerr_loss = rel_err.mean()
            total = total + relerr_weight * relerr_loss

        # Total-quantity losses operate in log-space (symmetric, stable):
        # |log(sum exp Nk_pred)) - log(sum exp Nk_gt)| corresponds to log-factor
        # error in physical space. A loss of 0.1 means total is off by ~10%.
        if total_n_loss_weight > 0:
            pred_log_tot_N = torch.logsumexp(pred_log[:, 0:40], dim=-1)
            gt_log_tot_N = torch.logsumexp(gt_log[:, 0:40], dim=-1)
            total = total + total_n_loss_weight * (pred_log_tot_N - gt_log_tot_N).abs().mean()
        if total_m_loss_weight > 0:
            pred_log_tot_M = torch.logsumexp(pred_log[:, 40:80], dim=-1)
            gt_log_tot_M = torch.logsumexp(gt_log[:, 40:80], dim=-1)
            total = total + total_m_loss_weight * (pred_log_tot_M - gt_log_tot_M).abs().mean()

        return total

    total = 0.0
    # ---- grad rollout ----
    for t in range(grad_steps):
        x = torch.cat([current, cond_chunk[:, t, :]], dim=-1)
        current = model(x)
        if state_clamp is not None:
            current = current.clamp(-state_clamp, state_clamp)
        total = total + step_loss(current, state_chunk[:, t + 1, :])

    # ---- pushforward steps ----
    for k in range(pushforward_steps):
        t = grad_steps + k
        # detach so no gradient flows through rolled-out history;
        # the next model() call still gets gradient.
        current_in = current.detach()
        x = torch.cat([current_in, cond_chunk[:, t, :]], dim=-1)
        current = model(x)
        if state_clamp is not None:
            current = current.clamp(-state_clamp, state_clamp)
        total = total + pushforward_weight * step_loss(
            current, state_chunk[:, t + 1, :]
        )

    return total / n_total_steps


def evaluate_full_rollout(model, log_states, cond, norm_stats, device, n=200):
    model.eval()
    log_states = log_states[:n]; cond = cond[:n]
    n_steps = log_states.shape[1] - 1

    s_t = torch.as_tensor(log_states, dtype=torch.float32)
    c_t = torch.as_tensor(cond, dtype=torch.float32)
    s_norm = norm_stats.normalize_state(s_t).to(device)
    c_norm = norm_stats.normalize_cond(c_t).to(device)

    current = s_norm[:, 0, :]
    results = {}
    with torch.no_grad():
        for t in range(n_steps):
            x = torch.cat([current, c_norm[:, t, :]], dim=-1)
            current = model(x)
            hour = (t + 1) / 60.0
            for h in (1, 3, 6):
                if abs(hour - h) < 1 / 120:
                    pred_log = norm_stats.denormalize_state(current.cpu())
                    gt_log = norm_stats.denormalize_state(s_norm[:, t + 1, :].cpu())
                    rmse_log = torch.sqrt(((pred_log - gt_log) ** 2).mean()).item()
                    rmse_norm = torch.sqrt(((current - s_norm[:, t + 1, :]) ** 2).mean()).item()
                    results[h] = {"rmse_norm": rmse_norm, "rmse_log": rmse_log}
    return results


def _grads_finite(m):
    for p in m.parameters():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all(): return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/v4_repr_1M")
    p.add_argument("--out-dir", default="runs/p7")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--subsample-dt", type=int, default=1,
                   help="Subsample the trajectory timestep by this factor. 1 = "
                        "native 1-min steps (360 to 6h). 30 = 30-min steps (12 "
                        "to 6h). Subsamples states AND cond by taking every Nth "
                        "frame; chunk_len is then in units of this coarse step.")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--nuc-head-hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--noise-std", type=float, default=0.03)
    p.add_argument("--chunk-len", type=int, default=60,
                   help="In-chunk steps with gradient (the grad horizon).")
    p.add_argument("--pushforward-steps", type=int, default=4,
                   help="Extra detached-input rollout steps appended per chunk. "
                        "0 disables pushforward. Default 4 (= +4 min on top of "
                        "60-min in-grad rollout).")
    p.add_argument("--pushforward-weight", type=float, default=1.0,
                   help="Weight on pushforward-step losses (relative to in-grad).")
    p.add_argument("--gc-weight", type=float, default=20.0)
    p.add_argument("--edge-weight", type=float, default=3.0)
    p.add_argument("--core-end-excl", type=int, default=26,
                   help="First bin (Nk index) that belongs to the coarse head. "
                        "Default 26 means core covers bins 6-25. Use 25 to move "
                        "bin 25 (and Mk 65) into the coarse head.")
    p.add_argument("--bin-weight-overrides", type=str, default="",
                   help='Per-bin weight overrides, applied AFTER edge/gc. '
                        'Format: "idx:weight,idx:weight,...". State layout: '
                        'Nk=0..39, Mk=40..79, Gc=80. Example: "25:5,65:5" '
                        'upweights bin-25 Nk and Mk by 5x.')
    p.add_argument("--delta-norm", action="store_true")
    p.add_argument("--rollout-every", type=int, default=5)
    p.add_argument("--relerr-weight", type=float, default=0.0,
                   help="Weight on the physical-space relative-error loss term "
                        "(hybrid loss = (1-w)*log_mse + w*rel_err). 0 disables.")
    p.add_argument("--relerr-clip-max", type=float, default=10.0,
                   help="Per-sample relative-error values above this are clipped "
                        "to bound outlier contribution.")
    p.add_argument("--tail-quantile", type=float, default=0.0,
                   help="If in (0,1), the rel-err loss averages only the worst "
                        "(1-q) fraction of entries per step (top-k focus). "
                        "0 disables (use mean). Use e.g. 0.9 to put loss on the "
                        "top 10%% of rel-errors per batch -- targets the tail "
                        "directly so p90 of the rel-err distribution comes down.")
    p.add_argument("--total-n-loss-weight", type=float, default=0.0,
                   help="Additive loss weight on |log(sum(Nk_pred)) - log(sum(Nk_gt))| "
                        "(log-space MAE on total number). Soft constraint pushing "
                        "the integrated total_N to match gt. 0 disables.")
    p.add_argument("--total-m-loss-weight", type=float, default=0.0,
                   help="Additive loss weight on |log(sum(Mk_pred)) - log(sum(Mk_gt))| "
                        "(log-space MAE on total aerosol mass). 0 disables.")
    p.add_argument("--lr-min-frac", type=float, default=0.0,
                   help="Cosine LR floor as a fraction of peak LR (eta_min = lr * "
                        "lr_min_frac). Default 0 (decay to zero). Use ~0.1 so late "
                        "epochs keep learning when a phase's epoch budget is short.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default="auto",
                   help='Resume from this checkpoint path, "auto" to use '
                        'out_dir/latest_checkpoint.pt if present, or "none".')
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--skip-loss-thresh", type=float, default=float("inf"),
                   help="Skip batches whose loss exceeds this absolute threshold. "
                        "Use to ignore outlier 'stiff' chunks that would otherwise "
                        "produce huge gradients and destabilize fine-tuning.")
    p.add_argument("--state-clamp", type=float, default=20.0)
    p.add_argument("--delta-std-floor", type=float, default=0.05)
    p.add_argument("--regression-thresh", type=float, default=5.0)
    p.add_argument("--log-factor-clamp", type=float, default=2.302585)
    p.add_argument("--proj-n-iter", type=int, default=3,
                   help="Alternating-projection iterations in v2 model.")
    p.add_argument("--total-n-log-clamp", type=float, default=0.0,
                   help="If > 0, enables a soft total-Nk anchor in the "
                        "conservation projection: the predicted total-Nk is "
                        "shifted toward the input total-Nk with a clamped "
                        "additive log-factor of this magnitude. ln(2)=0.693 "
                        "caps per-step total-N change to a factor of 2x. "
                        "0 disables (default).")
    p.add_argument("--init-from", type=str, default=None,
                   help="Initialize base weights from another checkpoint "
                        "(e.g. v1 best_model.pt) without loading optimizer "
                        "state. Useful for warm-starting v2 from v1.")
    args = p.parse_args()

    world_size, rank, local_rank, is_ddp = _ddp_setup()
    is_rank0 = (rank == 0)

    def log(*a, **k):
        if is_rank0:
            print(*a, **k)

    out_dir = Path(args.out_dir)
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  DDP: world_size={world_size} rank={rank} local_rank={local_rank}")

    # Each rank uses the same seed for its own (shard-local) permutation. Because
    # load_v4_batches striped-shards files by (idx % world_size == rank), each rank
    # holds a non-overlapping subset; train/val are computed within that shard.
    rng = np.random.default_rng(args.seed)
    log(f"Loading v4 data from {args.data_dir} (world_size={world_size}, "
        f"per-rank shard)...")
    Nk, Mk, Gc, cond = load_v4_batches(args.data_dir,
                                       max_scenarios=args.max_scenarios,
                                       rank=rank, world_size=world_size,
                                       verbose=is_rank0)
    n_total = Nk.shape[0]
    log(f"Rank-0 shard scenarios: {n_total:,}  (total across ranks ≈ {n_total*world_size:,})")

    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    log(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    log("Building log-state arrays...")
    log_states = build_state_array_v4(Nk, Mk, Gc)
    cond_t = transform_cond(cond)
    del Nk, Mk, Gc, cond

    if args.subsample_dt > 1:
        # Coarse-timestep emulation: keep every Nth frame so each model step
        # spans subsample_dt minutes. The model then rolls out far fewer steps
        # to 6h (e.g. 12 at 30-min vs 360 at 1-min), reducing compounding.
        # NOTE: conservation projection assumes a 1-min H2SO4 forcing, so for
        # coarse steps run with --proj-n-iter 0 (projection disabled).
        before = log_states.shape[1]
        log_states = log_states[:, ::args.subsample_dt]
        cond_t = cond_t[:, ::args.subsample_dt]
        log(f"  subsample_dt={args.subsample_dt}: {before} -> {log_states.shape[1]} frames/trajectory")

    train_log = log_states[train_idx]; val_log = log_states[val_idx]
    train_cond = cond_t[train_idx]; val_cond = cond_t[val_idx]
    del log_states, cond_t

    resume_path = None
    if args.resume == "auto":
        cand = out_dir / "latest_checkpoint.pt"
        resume_path = str(cand) if cand.is_file() else None
    elif args.resume and args.resume.lower() != "none":
        resume_path = args.resume

    saved_ns = out_dir / "norm_stats.pt"
    from dataset_v4 import NormStatsV4
    if is_rank0:
        if resume_path and saved_ns.is_file():
            norm_stats = NormStatsV4.load(saved_ns)
            log(f"Loaded norm_stats from {saved_ns} (resume).")
        else:
            log("Computing norm stats on training set...")
            norm_stats = compute_norm_stats_v4(train_log, train_cond)
            if args.delta_norm and args.delta_std_floor > 0:
                orig = norm_stats.delta_std.clone()
                norm_stats.delta_std = torch.clamp(norm_stats.delta_std,
                                                   min=args.delta_std_floor)
                n_floored = (orig < args.delta_std_floor).sum().item()
                log(f"  delta-std floor: {n_floored}/{N_STATE} dims clipped to {args.delta_std_floor}")
                log(f"  weight ratio  before: {(1.0/orig.min())**2 / (1.0/orig.max())**2:.0f}×")
                log(f"  weight ratio   after: "
                    f"{(1.0/norm_stats.delta_std.min())**2 / (1.0/norm_stats.delta_std.max())**2:.1f}×")
            norm_stats.save(saved_ns)
    if is_ddp:
        dist.barrier()
    if not is_rank0:
        norm_stats = NormStatsV4.load(saved_ns)
    log(f"  cond_mean={norm_stats.cond_mean.numpy()}")
    log(f"  cond_std ={norm_stats.cond_std.numpy()}")

    # Per-bin physical floor for the rel-err loss. Use the 1st-percentile of
    # exp(train_log) per dim, with a hard floor of 1e-30 (LOG_EPS).
    # Sampling cap to keep this fast.
    if args.relerr_weight > 0.0:
        cap = min(100_000, train_log.shape[0])
        sample = train_log[:cap].reshape(-1, train_log.shape[-1])
        phys = np.exp(sample.astype(np.float32))
        phys_floor_np = np.maximum(np.percentile(phys, 1.0, axis=0), 1e-30).astype(np.float32)
        phys_floor_tensor = torch.as_tensor(phys_floor_np, device=device)
        log(f"  rel-err loss ON: weight={args.relerr_weight}, clip_max={args.relerr_clip_max}")
        log(f"  phys_floor min={phys_floor_np.min():.3e}, max={phys_floor_np.max():.3e}")
    else:
        phys_floor_tensor = None

    bin_weight_overrides = parse_bin_weight_overrides(args.bin_weight_overrides)
    if bin_weight_overrides:
        log(f"  bin-weight overrides: {bin_weight_overrides}")

    # Dataset chunk_len = grad_steps + pushforward_steps so we have GT for both.
    ds_chunk_len = args.chunk_len + args.pushforward_steps
    train_ds = TomasTrajectoryDatasetV4(train_log, train_cond, norm_stats, chunk_len=ds_chunk_len)
    val_ds = TomasTrajectoryDatasetV4(val_log, val_cond, norm_stats, chunk_len=ds_chunk_len)
    log(f"Train chunks: {len(train_ds):,}  Val chunks: {len(val_ds):,} "
        f"(ds_chunk_len={ds_chunk_len} = grad {args.chunk_len} + pf {args.pushforward_steps})")

    # Each rank already holds a non-overlapping data shard (see load_v4_batches),
    # so DistributedSampler would double-shard. Use a plain shuffled loader.
    train_sampler = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    # Val runs only on rank 0 with its own shard.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Equal-iteration guard: rank shards can differ by a few files (e.g. 29 vs 28),
    # giving different per-rank batch counts. The in-loop bad-loss all-reduce and
    # backward all-reduces require every rank to call the same number of collectives
    # per epoch -- otherwise NCCL deadlocks at the next collective boundary (val
    # broadcast / epoch barrier). Cap every rank to the global min batches/epoch.
    n_batches_local = len(train_loader)
    if is_ddp:
        t = torch.tensor(n_batches_local, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_batches_per_epoch = int(t.item())
    else:
        n_batches_per_epoch = n_batches_local
    log(f"Train batches per rank: local={n_batches_local} "
        f"-> using global min={n_batches_per_epoch}")

    base = TomasMultiHeadMLP(
        hidden_dim=args.hidden_dim,
        n_backbone_layers=args.n_layers,
        head_hidden_dim=args.head_hidden_dim,
        nuc_head_hidden_dim=args.nuc_head_hidden_dim,
        dropout=args.dropout,
        n_cond=N_COND,
        core_end_excl=args.core_end_excl,
    ).to(device)

    if args.init_from and not resume_path:
        ck = torch.load(args.init_from, weights_only=False, map_location=device)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        # Strip wrapper prefixes if present.
        sd = {k.replace("base.", "", 1) if k.startswith("base.") else k: v
              for k, v in sd.items()}
        # Filter out tensors whose shape doesn't match the current model — e.g.
        # head_core / head_coarse output layers change shape when core_end_excl
        # is shifted. Those heads will be reinitialized from scratch; the
        # backbone + matching heads still warm-start.
        own = base.state_dict()
        # Never load partition-determined index buffers from another
        # checkpoint: their shape may match across partitions (reassemble is
        # always (81,)) but their values are wrong for the new partition.
        # Always trust the freshly-constructed model's buffers.
        IDX_BUFFERS = {"nuc_idx", "core_idx", "coarse_idx", "gc_idx", "reassemble"}
        filtered = {}
        skipped_shape = []
        skipped_idx = []
        for k, v in sd.items():
            if k in IDX_BUFFERS:
                skipped_idx.append(k); continue
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
            elif k in own:
                skipped_shape.append((k, tuple(v.shape), tuple(own[k].shape)))
        missing, unexpected = base.load_state_dict(filtered, strict=False)
        log(f"Warm-started base from {args.init_from}  "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"skipped_shape_mismatch={len(skipped_shape)} "
            f"skipped_idx_buffers={len(skipped_idx)}")
        for k, shp_old, shp_new in skipped_shape:
            log(f"  reinit (shape {shp_old} -> {shp_new}): {k}")
        if skipped_idx:
            log(f"  idx buffers always model-computed, not loaded: {skipped_idx}")

        # Zero-init any final-layer head weights that were skipped (shape
        # mismatch from a partition change). Heads predict residual deltas, so
        # zeroing the *output* layer means the initial rollout is the identity
        # mapping (out = state + 0). That keeps the warm-up stable until the
        # backbone gradient signal can dial in the new partition; otherwise the
        # default kaiming init produces ~O(1) deltas that catastrophically blow
        # up multi-step rollouts on the very first batch (saw 200/200 bad
        # batches > skip-loss-thresh).
        zero_targets = []
        for k, _, _ in skipped_shape:
            # The .net.2.* are the output Linear layers of HeadMLP.
            if k.endswith(".net.2.weight") or k.endswith(".net.2.bias"):
                zero_targets.append(k)
        if zero_targets:
            # Mutate parameters directly by walking the module tree (not via
            # state_dict(), which returns detached views that don't write back
            # to the parameter in all PyTorch versions).
            with torch.no_grad():
                for k in zero_targets:
                    obj = base
                    parts = k.split(".")
                    for p_ in parts[:-1]:
                        obj = getattr(obj, p_) if not p_.isdigit() else obj[int(p_)]
                    last = parts[-1]
                    tensor = getattr(obj, last) if not last.isdigit() else obj[int(last)]
                    tensor.data.zero_()
            log(f"  zero-initialized {len(zero_targets)} reinit'd head output "
                f"tensor(s) (identity-rollout warm-up): {zero_targets}")

    log_factor_clamp = args.log_factor_clamp if args.log_factor_clamp > 0 else None
    total_n_log_clamp = args.total_n_log_clamp if args.total_n_log_clamp > 0 else None
    model = ConservativeMultiHeadMLPV2(base, norm_stats,
                                       log_factor_clamp=log_factor_clamp,
                                       n_iter=args.proj_n_iter,
                                       total_n_log_clamp=total_n_log_clamp).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Multihead MLP (conserved v2): backbone {args.n_layers}x{args.hidden_dim}, "
        f"head {args.head_hidden_dim} (nuc {args.nuc_head_hidden_dim}), params {n_params:,}")
    log(f"chunk_len={args.chunk_len}, pushforward_steps={args.pushforward_steps}, "
        f"pushforward_weight={args.pushforward_weight}, noise_std={args.noise_std}, "
        f"gc_weight={args.gc_weight}, edge_weight={args.edge_weight}, "
        f"delta_norm={args.delta_norm}, log_factor_clamp={log_factor_clamp}, "
        f"proj_n_iter={args.proj_n_iter}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # eta_min floors the cosine decay so late epochs still learn (avoids the
    # "LR hit 0 before the model converged" under-training seen with the 33M
    # model at 5 epochs/phase). lr_min_frac=0 reproduces the old behavior.
    eta_min = args.lr * args.lr_min_frac
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=eta_min)

    best_val = float("inf")
    log_rows = []
    consecutive_bad = 0
    MAX_CONSECUTIVE_BAD = 200
    n_regression_resets = 0
    start_epoch = 1

    if resume_path is not None:
        ck = torch.load(resume_path, weights_only=False, map_location=device)
        base.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        if "best_val" in ck:
            best_val = ck["best_val"]
        if "epoch" in ck:
            start_epoch = ck["epoch"] + 1
        if "rng_state" in ck and ck["rng_state"] is not None:
            torch.set_rng_state(ck["rng_state"].cpu())
        if "cuda_rng_state" in ck and ck["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng_state"].cpu())
        log(f"Resumed from {resume_path}: epoch {ck.get('epoch', '?')} -> "
            f"continuing at epoch {start_epoch}; best_val={best_val:.6f}")

    # Wrap with DDP after weights are loaded (cold-start, init_from, or resume) so
    # all ranks already hold the same parameters before DDP's init broadcast.
    # broadcast_buffers=False: our buffers (norm stats, bin bounds) are static
    # after init, and keeping the default would make every forward a collective op
    # -- which would deadlock when rank 0 alone runs evaluate_full_rollout.
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=False)

    def save_best(epoch, val_mse):
        torch.save({"epoch": epoch,
                    "model_state_dict": base.state_dict(),
                    "val_mse": val_mse,
                    "args": vars(args)},
                   out_dir / "best_model.pt")

    def save_full_checkpoint(epoch, val_mse, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": base.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "val_mse": val_mse,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state()
                               if torch.cuda.is_available() else None),
            "args": vars(args),
        }, path)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        ep_loss = 0.0; n_batches = 0
        n_bad_loss = 0; n_bad_grad = 0
        for batch_idx, (state_chunk, cond_chunk) in enumerate(train_loader):
            if batch_idx >= n_batches_per_epoch:
                break
            state_chunk = state_chunk.to(device, non_blocking=True)
            cond_chunk = cond_chunk.to(device, non_blocking=True)
            loss = rollout_loss(model, state_chunk, cond_chunk,
                                noise_std=args.noise_std,
                                gc_weight=args.gc_weight,
                                edge_weight=args.edge_weight,
                                bin_weight_overrides=bin_weight_overrides,
                                delta_norm=args.delta_norm,
                                norm_stats=norm_stats,
                                state_clamp=args.state_clamp,
                                grad_steps=args.chunk_len,
                                pushforward_steps=args.pushforward_steps,
                                pushforward_weight=args.pushforward_weight,
                                relerr_weight=args.relerr_weight,
                                relerr_clip_max=args.relerr_clip_max,
                                phys_floor=phys_floor_tensor,
                                tail_quantile=args.tail_quantile,
                                total_n_loss_weight=args.total_n_loss_weight,
                                total_m_loss_weight=args.total_m_loss_weight)
            v = loss.item()
            # Catch (a) NaN/Inf and (b) finite-but-huge outlier losses that would
            # blow up the optimizer step even after grad clipping. (b) is the typical
            # phase-2+ failure mode: a single "stiff" chunk produces loss >> normal.
            huge = (v == v) and (v > args.skip_loss_thresh)
            local_bad_loss = not (v == v and v != float("inf") and v != float("-inf"))
            local_bad_loss = local_bad_loss or huge
            # Sync across ranks: if ANY rank has a bad loss, ALL ranks skip the
            # backward, otherwise DDP's gradient all-reduce would deadlock.
            bad_loss_any = _any_rank_true(local_bad_loss, device, is_ddp)
            if bad_loss_any:
                n_bad_loss += int(local_bad_loss)
                consecutive_bad += 1
                optimizer.zero_grad(set_to_none=True)
                if consecutive_bad >= MAX_CONSECUTIVE_BAD:
                    raise RuntimeError(
                        f"Aborting: {consecutive_bad} consecutive bad batches")
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # After backward, DDP has all-reduced gradients across ranks, so a NaN
            # on any rank propagates everywhere; this check is consistent across ranks.
            if not _grads_finite(model):
                n_bad_grad += 1; consecutive_bad += 1
                optimizer.zero_grad(set_to_none=True)
                if consecutive_bad >= MAX_CONSECUTIVE_BAD:
                    raise RuntimeError(
                        f"Aborting: {consecutive_bad} consecutive bad batches")
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            consecutive_bad = 0
            ep_loss += v; n_batches += 1
        scheduler.step()
        train_mse = ep_loss / max(n_batches, 1)

        # Validation: only rank 0 runs; broadcast val_mse so all ranks stay in sync.
        if is_rank0:
            model.eval()
            v_loss, v_n, v_bad = 0.0, 0, 0
            with torch.no_grad():
                for state_chunk, cond_chunk in val_loader:
                    state_chunk = state_chunk.to(device, non_blocking=True)
                    cond_chunk = cond_chunk.to(device, non_blocking=True)
                    vb = rollout_loss(model, state_chunk, cond_chunk,
                                      gc_weight=args.gc_weight,
                                      edge_weight=args.edge_weight,
                                      bin_weight_overrides=bin_weight_overrides,
                                      delta_norm=args.delta_norm,
                                      norm_stats=norm_stats,
                                      state_clamp=args.state_clamp,
                                      grad_steps=args.chunk_len,
                                      pushforward_steps=args.pushforward_steps,
                                      relerr_weight=args.relerr_weight,
                                      relerr_clip_max=args.relerr_clip_max,
                                      phys_floor=phys_floor_tensor,
                                      pushforward_weight=args.pushforward_weight,
                                      tail_quantile=args.tail_quantile,
                                      total_n_loss_weight=args.total_n_loss_weight,
                                      total_m_loss_weight=args.total_m_loss_weight).item()
                    if vb == vb and vb != float("inf") and vb != float("-inf"):
                        v_loss += vb; v_n += 1
                    else:
                        v_bad += 1
            val_mse = v_loss / max(v_n, 1) if v_n > 0 else float("inf")
        else:
            v_bad = 0
            val_mse = float("inf")
        val_mse = _broadcast_float(val_mse, device, is_ddp, src=0)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        log(f"Epoch {epoch:3d}/{args.epochs} | train MSE {train_mse:.6f} | "
            f"val MSE {val_mse:.6f} | lr {lr:.2e} | "
            f"bad: loss={n_bad_loss} grad={n_bad_grad} val={v_bad} | "
            f"{elapsed:.1f}s", flush=True)

        # Regression reset: decision must be identical across ranks so all reload.
        local_should_reset = (
            best_val != float("inf") and
            val_mse > args.regression_thresh * best_val and
            Path(out_dir / "best_model.pt").is_file()
        )
        should_reset = _any_rank_true(local_should_reset, device, is_ddp)
        if should_reset:
            n_regression_resets += 1
            log(f"  ! REGRESSION: val {val_mse:.4f} > {args.regression_thresh}x "
                f"best {best_val:.4f}. Reloading best_model.pt.")
            ckpt = torch.load(out_dir / "best_model.pt",
                              weights_only=True, map_location=device)
            base.load_state_dict(ckpt["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            continue

        if args.rollout_every > 0 and (epoch == 1 or epoch % args.rollout_every == 0):
            if is_rank0:
                ro = evaluate_full_rollout(model, val_log, val_cond, norm_stats, device, n=200)
                for h, m in sorted(ro.items()):
                    log(f"  Rollout {h}h: RMSE(norm)={m['rmse_norm']:.4f} "
                        f"RMSE(log)={m['rmse_log']:.4f}", flush=True)
            if is_ddp:
                dist.barrier()

        log_rows.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "lr": lr})

        if val_mse < best_val:
            best_val = val_mse
            if is_rank0:
                save_best(epoch, val_mse)
            log(f"  -> new best ({val_mse:.6f})")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            if is_rank0:
                save_full_checkpoint(epoch, val_mse, out_dir / "latest_checkpoint.pt")
        if is_ddp:
            dist.barrier()

    if is_rank0:
        torch.save({"epoch": args.epochs,
                    "model_state_dict": base.state_dict(),
                    "val_mse": val_mse,
                    "args": vars(args)},
                   out_dir / "final_model.pt")
        if log_rows:
            np.savez(out_dir / "train_log.npz", **{
                k: np.array([r[k] for r in log_rows]) for k in log_rows[0]
            })
    log(f"\nDone. Best val MSE: {best_val:.6f}  "
        f"(regression resets: {n_regression_resets})\nOutputs -> {out_dir}")
    _ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
