"""Δt-conditioned TOMAS emulator trained with the PROVEN p7 ROLLOUT CURRICULUM.

Motivation (diagnosis 2026-05-28, /tmp/diag_dtc.py): the earlier Δt-consistency
model had EXCELLENT single-step accuracy (~1.7% teacher-forced at every Δt) but
43-58% at 6h rollout — i.e. the failure was EXPOSURE BIAS / compounding, because
that trainer only ever did a depth-2 composition and no multi-step rollout. The
consistency loss was inert (~1% residual). The fix is the same multi-step
rollout/pushforward curriculum that made p7 work, applied per-Δt.

This file forks train_v4_conserved_v2.py (the p7 trainer) and adds:
  - Δt conditioning via Fourier features (imported from train_dt_consistency so
    eval_dt_consistency.py works unchanged on this run's checkpoints).
  - A Δt-rollout dataset: each sample picks a step size d from a DENSE set
    (fixes the off-lattice Δt=5 OOD hole) and returns a d-spaced sub-trajectory
    of (chunk_len + pushforward_steps + 1) points, so the rollout loss trains on
    the exact d-step dynamics the model will see at inference.
  - p7 warm-start with exact zero-init of the 8 new Δt input columns, so the
    model == p7 at init and only learns to USE Δt from there.

Conservation is OFF for this run (base TomasMultiHeadMLP, no projection wrapper):
the v2 conservation projection assumes a 1-min H2SO4 forcing and would be wrong
for coarse d (needs interval-integrated forcing — a dataset change, deferred).
This matches the coarse-30min run, and isolates the rollout-curriculum effect.

Reuses the DDP fixes from the p7/consistency trainers: broadcast_buffers=False,
precomputed val tensors (no val DataLoader), no DistributedSampler + global-min
batch cap, synced bad-loss skip.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from dataset_v4 import (
    load_v4_batches, build_state_array_v4, transform_cond,
    compute_norm_stats_v4, NormStatsV4, N_STATE, N_COND,
)
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2
from dt_features import fourier_dt_feat, N_DT_FEAT, N_COND_BASE, N_COND_TOTAL
try:  # experimental alternative architectures, not included in this release
    from model_bin_transformer import TomasBinTransformer
    from model_overlap_head import TomasOverlapHeadMLP
    from model_bin_conv import TomasBinConvMLP
    from model_bin_cnn import TomasBinCNN
    from model_ponder import TomasIterRefine
except ImportError:
    pass

DEFAULT_DT_LIST = "1,2,3,4,5,6,8,10,12,15,20,24,30"


def _ddp_setup():
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_world_size(), dist.get_rank(), local_rank, True


def _any_rank_true(flag, device, is_ddp):
    if not is_ddp:
        return flag
    t = torch.tensor(int(flag), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def _broadcast_float(value, device, is_ddp, src=0):
    if not is_ddp:
        return value
    t = torch.tensor(float(value), device=device)
    dist.broadcast(t, src=src)
    return t.item()


def _grads_finite(m):
    for p in m.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def build_bin_weights(gc_weight=1.0, edge_weight=1.0, mk_weight=1.0,
                      nuc_weight=1.0, coarse_weight=1.0, bin25_weight=1.0, device=None):
    w = torch.ones(81, device=device)
    if edge_weight != 1.0:                 # legacy combined edge (nuc+coarse)
        w[0:6] = edge_weight; w[40:46] = edge_weight
        w[26:40] = edge_weight; w[66:80] = edge_weight
    if nuc_weight != 1.0:                  # nucleation bins (real error lives here)
        w[0:6] = nuc_weight; w[40:46] = nuc_weight
    if coarse_weight != 1.0:               # coarse bins (already most accurate)
        w[26:40] = coarse_weight; w[66:80] = coarse_weight
    if mk_weight != 1.0:
        w[40:80] = w[40:80] * mk_weight    # global moment weight on all Mk bins
    if gc_weight != 1.0:
        w[80] = gc_weight
    if bin25_weight != 1.0:                # bin-25 seam: Nk idx 25, Mk idx 65
        w[25] = w[25] * bin25_weight; w[65] = w[65] * bin25_weight
    return w / w.mean()


class TomasDtRolloutDataset(Dataset):
    """Each item: a d-spaced sub-trajectory of (L+1) points + the step size d.

    L = chunk_len + pushforward_steps (total rollout steps). d is drawn from
    dt_list, filtered to those with L*d <= n_steps-1 so the strided window fits.
    All items in a batch have the SAME L (uniform collation); d varies per row.
    """

    def __init__(self, log_states, cond, norm_stats, n_chunk_steps, dt_list,
                 samples_per_scenario=8):
        self.S = norm_stats.normalize_state(torch.as_tensor(log_states, dtype=torch.float32))
        self.C = norm_stats.normalize_cond(torch.as_tensor(cond, dtype=torch.float32))
        self.n_scen, self.n_steps, _ = self.S.shape
        self.L = n_chunk_steps
        self.spp = samples_per_scenario
        self.dt_list = [d for d in dt_list if self.L * d <= self.n_steps - 1]
        if not self.dt_list:
            raise ValueError(
                f"No valid d: all L*d > n_steps-1 (L={self.L}, n_steps={self.n_steps})")

    def __len__(self):
        return self.n_scen * self.spp

    def __getitem__(self, idx):
        scn = idx % self.n_scen
        rng = np.random.default_rng((idx * 2654435761 + os.getpid()
                                     + int(time.time() * 1e3)) & 0x7FFFFFFF)
        d = int(rng.choice(self.dt_list))
        max_t0 = self.n_steps - 1 - self.L * d
        t0 = int(rng.integers(0, max_t0 + 1))
        idxs = t0 + d * np.arange(self.L + 1)
        return (self.S[scn, idxs], self.C[scn, idxs], torch.tensor(float(d)))


def rollout_loss_dt(model, state_chunk, cond_chunk, d,
                    norm_stats, device,
                    noise_std=0.0, gc_weight=1.0, edge_weight=1.0,
                    state_clamp=20.0, grad_steps=None, pushforward_steps=0,
                    pushforward_weight=1.0, relerr_weight=0.0,
                    relerr_clip_max=10.0, phys_floor=None,
                    loss_type="hybrid", huber_beta=0.2, aux_mse_weight=0.1,
                    mk_weight=1.0, nuc_weight=1.0, coarse_weight=1.0, bias_weight=0.0, realerr_weight=0.0,
                    bin25_weight=1.0):
    """Roll forward over a d-spaced chunk, Δt-conditioned. Mirrors p7's
    rollout_loss + hybrid step_loss, with the Fourier-Δt feature concatenated
    to the conditioning at every step."""
    n_total_steps = state_chunk.shape[1] - 1
    if grad_steps is None:
        grad_steps = n_total_steps - pushforward_steps
    assert grad_steps + pushforward_steps == n_total_steps

    dt_feat = fourier_dt_feat(d)  # (B, N_DT_FEAT); constant across the rollout

    current = state_chunk[:, 0, :]

    use_w = (gc_weight != 1.0) or (edge_weight != 1.0) or (mk_weight != 1.0) \
        or (nuc_weight != 1.0) or (coarse_weight != 1.0) or (bin25_weight != 1.0)
    w = build_bin_weights(gc_weight, edge_weight, mk_weight=mk_weight,
                          nuc_weight=nuc_weight, coarse_weight=coarse_weight,
                          bin25_weight=bin25_weight, device=device) if use_w else None

    def step_loss(pred, gt):
        if loss_type == "logmae":
            # Factor-error loss: Δ = pred_log - gt_log = log(pred_phys/gt_phys),
            # the multiplicative/relative error in physical log space. Huber (smooth-L1)
            # => Jensen-free, bin-uniform, bounded, and L1-tail => median-aligned
            # (our eval metric is the MEDIAN factor error). Small normalized-log-MSE
            # aux term keeps early-training gradients smooth.
            pred_log = norm_stats.denormalize_state(pred).clamp(min=-60.0, max=60.0)
            gt_log = norm_stats.denormalize_state(gt).clamp(min=-60.0, max=60.0)
            delta = pred_log - gt_log
            absd = delta.abs()
            b = huber_beta
            huber = torch.where(absd < b, 0.5 * delta * delta / b, absd - 0.5 * b)
            if w is not None:
                huber = w * huber
            loss = huber.mean()
            if aux_mse_weight > 0:
                aux = (pred - gt) ** 2
                if w is not None:
                    aux = w * aux
                loss = loss + aux_mse_weight * aux.mean()
            if bias_weight > 0:                # penalize SIGNED per-bin batch-mean error => de-bias
                bias = (pred_log - gt_log).mean(0).abs()
                if w is not None: bias = w * bias
                loss = loss + bias_weight * bias.mean()
            if realerr_weight > 0 and phys_floor is not None:   # REAL-SPACE rel-err (anneal up to sharpen)
                pred_phys = torch.exp(pred_log); gt_phys = torch.exp(gt_log)
                denom = torch.clamp(gt_phys, min=phys_floor)
                rel = torch.clamp((pred_phys - gt_phys).abs() / denom, max=relerr_clip_max)
                if w is not None: rel = w * rel
                loss = loss + realerr_weight * rel.mean()
            return loss
        # --- hybrid (p7-style: normalized log-MSE + clipped physical rel-err) ---
        err_sq = (pred - gt) ** 2
        if w is not None:
            err_sq = w * err_sq
        log_mse = err_sq.mean()
        if not (relerr_weight > 0 and phys_floor is not None):
            return log_mse
        pred_log = norm_stats.denormalize_state(pred).clamp(min=-60.0, max=60.0)
        gt_log = norm_stats.denormalize_state(gt).clamp(min=-60.0, max=60.0)
        pred_phys = torch.exp(pred_log)
        gt_phys = torch.exp(gt_log)
        denom = torch.clamp(gt_phys, min=phys_floor)
        rel_err = torch.clamp((pred_phys - gt_phys).abs() / denom, max=relerr_clip_max)
        if w is not None:
            rel_err = w * rel_err
        return (1.0 - relerr_weight) * log_mse + relerr_weight * rel_err.mean()

    total = 0.0
    for t in range(grad_steps):
        inp = current
        if noise_std > 0:                      # per-step input noise => learn to CORRECT perturbations
            inp = inp + torch.randn_like(inp) * noise_std
        x = torch.cat([inp, cond_chunk[:, t, :], dt_feat], dim=-1)
        current = model(x).clamp(-state_clamp, state_clamp)
        total = total + step_loss(current, state_chunk[:, t + 1, :])
    for k in range(pushforward_steps):
        t = grad_steps + k
        current_in = current.detach()
        x = torch.cat([current_in, cond_chunk[:, t, :], dt_feat], dim=-1)
        current = model(x).clamp(-state_clamp, state_clamp)
        total = total + pushforward_weight * step_loss(current, state_chunk[:, t + 1, :])
    return total / n_total_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/tomas-jax-output/v4_repr_1M")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--nuc-head-hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--core-end-excl", type=int, default=26)
    p.add_argument("--seam-hidden-dim", type=int, default=128,
                   help="hidden width of the overlap seam head (model-type overlap)")
    p.add_argument("--conv-channels", type=int, default=64,
                   help="hidden channels of the bin-local conv branch (model-type binconv)")
    p.add_argument("--conv-kernel", type=int, default=5,
                   help="kernel size (bin receptive field) of the binconv branch")
    p.add_argument("--conv-layers", type=int, default=2,
                   help="number of conv layers in the binconv branch")
    p.add_argument("--conv-pos-emb", type=int, default=0,
                   help="learned per-bin positional-embedding channels for binconv "
                        "(>0 lets the conv apply position-specific corrections)")
    p.add_argument("--train-only-seam", action="store_true",
                   help="freeze everything except the seam head (overlap fine-tune)")
    p.add_argument("--model-type", choices=["mlp", "transformer", "cnn", "iterrefine", "overlap", "binconv"], default="mlp",
                   help="mlp = TomasMultiHeadMLP (p7 arch); transformer = "
                        "TomasBinTransformer (GNN over bins, all-pairs attention).")
    p.add_argument("--n-heads", type=int, default=8, help="transformer attention heads")
    p.add_argument("--refine-iters", type=int, default=6, help="iterrefine refinement passes (K)")
    p.add_argument("--ffn-mult", type=int, default=4, help="transformer FFN width multiplier")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--chunk-len", type=int, default=12,
                   help="Grad rollout horizon (number of d-steps with gradient).")
    p.add_argument("--pushforward-steps", type=int, default=0,
                   help="Extra detached-input rollout steps appended per chunk.")
    p.add_argument("--pushforward-weight", type=float, default=1.0)
    p.add_argument("--samples-per-scenario", type=int, default=8)
    p.add_argument("--val-max-batches", type=int, default=80)
    p.add_argument("--dt-list", type=str, default=DEFAULT_DT_LIST,
                   help="Comma-separated step sizes d (min). Filtered per phase "
                        "to those with chunk_len*d <= n_steps-1.")
    p.add_argument("--gc-weight", type=float, default=20.0)
    p.add_argument("--edge-weight", type=float, default=3.0)
    p.add_argument("--mk-weight", type=float, default=1.0,
                   help="Global loss weight on all Mk bins (40-79). >1 recovers Mk "
                        "accuracy when the bin-uniform logmae loss equalizes Nk/Mk.")
    p.add_argument("--nuc-weight", type=float, default=1.0,
                   help="Loss weight on nucleation bins (Nk 0-5, Mk 40-45) — where "
                        "the real error lives. Use instead of edge-weight.")
    p.add_argument("--coarse-weight", type=float, default=1.0,
                   help="Loss weight on coarse bins (Nk 26-39, Mk 66-79) — already "
                        "the most accurate; keep ~1 (edge-weight over-weighted these).")
    p.add_argument("--bin25-weight", type=float, default=1.0,
                   help="Loss weight on the bin-25 seam (Nk idx 25, Mk idx 65). "
                        ">1 forces gradient pressure on the head-seam artifact bin.")
    p.add_argument("--relerr-weight", type=float, default=0.4)
    p.add_argument("--relerr-clip-max", type=float, default=5.0)
    p.add_argument("--loss-type", choices=["hybrid", "logmae"], default="hybrid",
                   help="hybrid = p7's 0.6*logMSE + 0.4*clipped-relerr. logmae = "
                        "Huber(pred_log - gt_log): Jensen-free, bin-uniform, "
                        "median-aligned factor-error loss.")
    p.add_argument("--huber-beta", type=float, default=0.2,
                   help="logmae Huber crossover (log units; 0.2 ~ 22%% rel-err).")
    p.add_argument("--realerr-weight", type=float, default=0.0,
                   help="REAL-space rel-err weight ADDED to logmae (anneal up to sharpen toward the metric).")
    p.add_argument("--bias-weight", type=float, default=0.0,
                   help="De-bias: penalty on signed per-bin batch-mean log-error (0=off).")
    p.add_argument("--aux-mse-weight", type=float, default=0.1,
                   help="logmae: weight on the normalized-log-MSE aux term.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default="auto")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--skip-loss-thresh", type=float, default=float("inf"))
    p.add_argument("--state-clamp", type=float, default=20.0)
    p.add_argument("--proj-n-iter", type=int, default=0,
                   help="Conservation projection iterations (ConservativeMultiHeadMLPV2). "
                        "0 = OFF (bare base model). 3 = p7's setting. Uses the step's "
                        "input-cond forcing (single-step), which is ~3%% accurate even "
                        "at d=30 since the forcing is smooth.")
    p.add_argument("--log-factor-clamp", type=float, default=2.302585,
                   help="Clamp per-iter additive log-factor in the projection (ln10).")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA decay on base weights (e.g. 0.9999 ~ avg last ~5 "
                        "epochs). Saves ema_model.pt. 0 disables. Reduces variance "
                        "-> lowers the p90 tail.")
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start base weights. p7's 86-col input layer is "
                        "expanded to 94 cols with the 8 Δt columns zero-init'd "
                        "(model == init at start).")
    args = p.parse_args()

    dt_list = [int(x) for x in args.dt_list.split(",")]
    world_size, rank, local_rank, is_ddp = _ddp_setup()
    is_rank0 = (rank == 0)

    def log(*a, **k):
        if is_rank0:
            print(*a, **k, flush=True)

    out_dir = Path(args.out_dir)
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    device = torch.device(f"cuda:{local_rank}") if is_ddp else torch.device("cuda")

    log(f"Δt-ROLLOUT trainer (conservation OFF). DDP ws={world_size}. "
        f"chunk_len={args.chunk_len} pf={args.pushforward_steps} dt_list={dt_list}")
    log(f"n_cond={N_COND_TOTAL} (={N_COND_BASE} cond + {N_DT_FEAT} Fourier-Δt)")

    rng = np.random.default_rng(args.seed)
    Nk, Mk, Gc, cond = load_v4_batches(args.data_dir, max_scenarios=args.max_scenarios,
                                       rank=rank, world_size=world_size, verbose=is_rank0)
    n_total = Nk.shape[0]
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_frac))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    log(f"Rank shard: {n_total:,}. Train {len(train_idx):,} Val {len(val_idx):,}")

    log_states = build_state_array_v4(Nk, Mk, Gc)
    # This model uses the 5 original forcing features; drop any extras
    # dataset_v4 has since grown (e.g. fion) so cond matches N_COND_BASE.
    cond_t = transform_cond(cond)[..., :N_COND_BASE]
    del Nk, Mk, Gc, cond
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
    if is_rank0:
        # Reuse norm_stats if present (resume, or pre-seeded from the p7 init so
        # the warm-start is byte-identical to p7). Else compute from train data.
        if saved_ns.is_file():
            norm_stats = NormStatsV4.load(saved_ns)
            log(f"Loaded existing norm_stats from {saved_ns}")
        else:
            log("Computing norm stats...")
            norm_stats = compute_norm_stats_v4(train_log, train_cond)
            norm_stats.save(saved_ns)
    if is_ddp:
        dist.barrier()
    if not is_rank0:
        norm_stats = NormStatsV4.load(saved_ns)

    phys_floor = None
    if args.relerr_weight > 0.0 or args.realerr_weight > 0.0:
        cap = min(100_000, train_log.shape[0])
        sample = train_log[:cap].reshape(-1, train_log.shape[-1])
        phys = np.exp(sample.astype(np.float32))
        phys_floor_np = np.maximum(np.percentile(phys, 1.0, axis=0), 1e-30).astype(np.float32)
        phys_floor = torch.as_tensor(phys_floor_np, device=device)
        log(f"rel-err loss ON: weight={args.relerr_weight} clip={args.relerr_clip_max}")

    L = args.chunk_len + args.pushforward_steps
    train_ds = TomasDtRolloutDataset(train_log, train_cond, norm_stats, L, dt_list,
                                     samples_per_scenario=args.samples_per_scenario)
    val_ds = TomasDtRolloutDataset(val_log, val_cond, norm_stats, L, dt_list,
                                   samples_per_scenario=args.samples_per_scenario)
    log(f"Effective dt_list (L={L}): {train_ds.dt_list}")
    log(f"Train items: {len(train_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True,
                              persistent_workers=(args.workers > 0))
    n_batches_local = len(train_loader)
    if is_ddp:
        t = torch.tensor(n_batches_local, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_batches_per_epoch = int(t.item())
    else:
        n_batches_per_epoch = n_batches_local
    log(f"Batches/rank local={n_batches_local} -> global min={n_batches_per_epoch}")

    # Precomputed val tensors (no val DataLoader — avoids the rank-0 fork hang).
    val_tensors = None
    if is_rank0:
        n_val_samples = min(args.val_max_batches * args.batch_size, len(val_ds))
        cols = [[], [], []]
        for j in range(n_val_samples):
            item = val_ds[j]
            for c, x in zip(cols, item):
                c.append(x)
        val_tensors = tuple(torch.stack(c) for c in cols)
        log(f"Precomputed val: {val_tensors[0].shape[0]} samples")

    if args.model_type == "iterrefine":
        base = TomasIterRefine(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                               n_cond=N_COND_TOTAL, refine_iters=args.refine_iters,
                               dropout=args.dropout).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: iter-refine d={args.hidden_dim} L={args.n_layers} K={args.refine_iters}, params {n_params:,}")
    elif args.model_type == "cnn":
        base = TomasBinCNN(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                           dropout=args.dropout, n_cond=N_COND_TOTAL).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: bin-CNN d={args.hidden_dim} L={args.n_layers}, n_cond={N_COND_TOTAL}, params {n_params:,}")
    elif args.model_type == "transformer":
        base = TomasBinTransformer(
            hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads,
            ffn_mult=args.ffn_mult, dropout=args.dropout, n_cond=N_COND_TOTAL,
        ).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: bin-Transformer d={args.hidden_dim} L={args.n_layers} "
            f"heads={args.n_heads} ffn={args.ffn_mult}x, n_cond={N_COND_TOTAL}, params {n_params:,}")
    elif args.model_type == "overlap":
        base = TomasOverlapHeadMLP(
            hidden_dim=args.hidden_dim, n_backbone_layers=args.n_layers,
            head_hidden_dim=args.head_hidden_dim, nuc_head_hidden_dim=args.nuc_head_hidden_dim,
            dropout=args.dropout, n_cond=N_COND_TOTAL, core_end_excl=args.core_end_excl,
            seam_hidden_dim=args.seam_hidden_dim,
        ).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: OVERLAP-head backbone {args.n_layers}x{args.hidden_dim}, "
            f"seam_hidden={args.seam_hidden_dim}, params {n_params:,}")
    elif args.model_type == "binconv":
        base = TomasBinConvMLP(
            hidden_dim=args.hidden_dim, n_backbone_layers=args.n_layers,
            head_hidden_dim=args.head_hidden_dim, nuc_head_hidden_dim=args.nuc_head_hidden_dim,
            dropout=args.dropout, n_cond=N_COND_TOTAL, core_end_excl=args.core_end_excl,
            conv_channels=args.conv_channels, conv_kernel=args.conv_kernel,
            conv_layers=args.conv_layers, conv_pos_emb=args.conv_pos_emb,
        ).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: BINCONV backbone {args.n_layers}x{args.hidden_dim} + conv "
            f"{args.conv_layers}L x{args.conv_channels}ch k{args.conv_kernel}, params {n_params:,}")
    else:
        base = TomasMultiHeadMLP(
            hidden_dim=args.hidden_dim, n_backbone_layers=args.n_layers,
            head_hidden_dim=args.head_hidden_dim, nuc_head_hidden_dim=args.nuc_head_hidden_dim,
            dropout=args.dropout, n_cond=N_COND_TOTAL, core_end_excl=args.core_end_excl,
        ).to(device)
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        log(f"Model: backbone {args.n_layers}x{args.hidden_dim}, n_cond={N_COND_TOTAL}, params {n_params:,}")

    if args.init_from and not resume_path:
        ck = torch.load(args.init_from, weights_only=False, map_location=device)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        sd = {k.replace("base.", "", 1) if k.startswith("base.") else k: v
              for k, v in sd.items()}
        own = base.state_dict()
        IDX_BUF = {"nuc_idx", "core_idx", "coarse_idx", "gc_idx", "reassemble"}
        filtered = {}
        expanded = []
        skipped = []
        for k, v in sd.items():
            if k in IDX_BUF or k not in own:
                continue
            if own[k].shape == v.shape:
                filtered[k] = v
            elif (k == "backbone.0.weight" and v.dim() == 2
                  and v.shape[0] == own[k].shape[0]
                  and v.shape[1] == 81 + N_COND
                  and own[k].shape[1] == 81 + N_COND_TOTAL):
                # Expand input layer: copy [state|cond] cols, zero the Δt cols
                # so the model is IDENTICAL to the init checkpoint at start.
                neww = torch.zeros_like(own[k])
                neww[:, :81 + N_COND] = v
                filtered[k] = neww
                expanded.append(k)
            else:
                skipped.append((k, tuple(v.shape), tuple(own[k].shape)))
        base.load_state_dict(filtered, strict=False)
        log(f"Warm-start from {args.init_from}: loaded {len(filtered)} "
            f"(expanded {expanded}), skipped_shape {len(skipped)}")
        for k, so, sn in skipped:
            log(f"  skipped (shape {so}->{sn}): {k}")

    # Optionally wrap base in the conservation projection (sulfur balance +
    # per-bin geometry clamp). The geometry clamp is Δt-independent; the sulfur
    # balance uses the step's input-cond forcing (single-step, ~3% accurate at
    # d=30 — see /tmp/test_cons.py; the summed-forcing variant is WRONG, p90~0.97).
    if args.train_only_seam:
        n_tr = 0
        for name, p in base.named_parameters():
            if "head_seam" not in name:
                p.requires_grad_(False)
            else:
                n_tr += p.numel()
        log(f"FROZEN all but seam head: {n_tr:,} trainable params")

    net = base
    if args.proj_n_iter > 0:
        lfc = args.log_factor_clamp if args.log_factor_clamp > 0 else None
        net = ConservativeMultiHeadMLPV2(base, norm_stats, log_factor_clamp=lfc,
                                         n_iter=args.proj_n_iter).to(device)
        log(f"Conservation ON: proj_n_iter={args.proj_n_iter} log_factor_clamp={lfc}")
    else:
        log("Conservation OFF (bare base model)")
    model = DDP(net, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, broadcast_buffers=False) if is_ddp else net

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac)

    start_epoch = 1
    best_val = float("inf")
    if resume_path:
        ck = torch.load(resume_path, weights_only=False, map_location=device)
        base.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        log(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")

    # EMA of base weights (rank 0 only; DDP keeps params synced across ranks so
    # the rank-0 EMA is correct). Reduces variance -> typically pulls in the p90
    # tail and a bit of the median. ema_decay=0 disables.
    ema = None
    if is_rank0 and args.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in base.state_dict().items()}
        log(f"EMA ON: decay={args.ema_decay}")

    def update_ema():
        if ema is None:
            return
        with torch.no_grad():
            sd = base.state_dict()
            for k, v in ema.items():
                if v.dtype.is_floating_point:
                    v.mul_(args.ema_decay).add_(sd[k].detach(), alpha=1.0 - args.ema_decay)
                else:
                    v.copy_(sd[k])

    def save_ckpt(epoch, val, path, full=True):
        d = {"epoch": epoch, "model_state_dict": base.state_dict(),
             "val_mse": val, "best_val": best_val, "args": vars(args)}
        if full:
            d["optimizer_state_dict"] = optimizer.state_dict()
            d["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(d, path)

    def save_ema(path):
        if ema is None:
            return
        torch.save({"model_state_dict": {k: v.clone() for k, v in ema.items()},
                    "args": vars(args), "is_ema": True}, path)

    def loss_for(batch):
        s, c, d = (x.to(device, non_blocking=True) for x in batch)
        return rollout_loss_dt(
            model if model is base else model, s, c, d, norm_stats, device,
            noise_std=args.noise_std, gc_weight=args.gc_weight, edge_weight=args.edge_weight,
            mk_weight=args.mk_weight, nuc_weight=args.nuc_weight, coarse_weight=args.coarse_weight,
            state_clamp=args.state_clamp, grad_steps=args.chunk_len,
            pushforward_steps=args.pushforward_steps, pushforward_weight=args.pushforward_weight,
            relerr_weight=args.relerr_weight, relerr_clip_max=args.relerr_clip_max,
            phys_floor=phys_floor, loss_type=args.loss_type, huber_beta=args.huber_beta,
            aux_mse_weight=args.aux_mse_weight, bias_weight=args.bias_weight, realerr_weight=args.realerr_weight,
            bin25_weight=args.bin25_weight)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0.0; nb = 0; n_bad = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= n_batches_per_epoch:
                break
            loss = loss_for(batch)
            v = loss.item()
            local_bad = (not (v == v and abs(v) != float("inf"))) or v > args.skip_loss_thresh
            if _any_rank_true(local_bad, device, is_ddp):
                n_bad += int(local_bad)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not _grads_finite(model):
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            update_ema()
            ep_loss += v; nb += 1
        scheduler.step()
        train_loss = ep_loss / max(nb, 1)

        if is_rank0:
            net.eval()
            bs = args.batch_size
            vs, vn = 0.0, 0
            with torch.no_grad():
                N = val_tensors[0].shape[0]
                for start in range(0, N, bs):
                    end = min(start + bs, N)
                    vbatch = tuple(t[start:end] for t in val_tensors)
                    vloss = rollout_loss_dt(
                        net, vbatch[0].to(device), vbatch[1].to(device), vbatch[2].to(device),
                        norm_stats, device, noise_std=0.0,
                        gc_weight=args.gc_weight, edge_weight=args.edge_weight, mk_weight=args.mk_weight,
                        nuc_weight=args.nuc_weight, coarse_weight=args.coarse_weight,
                        state_clamp=args.state_clamp, grad_steps=args.chunk_len,
                        pushforward_steps=args.pushforward_steps,
                        pushforward_weight=args.pushforward_weight,
                        relerr_weight=args.relerr_weight, relerr_clip_max=args.relerr_clip_max,
                        phys_floor=phys_floor, loss_type=args.loss_type,
                        huber_beta=args.huber_beta, aux_mse_weight=args.aux_mse_weight, bias_weight=args.bias_weight, realerr_weight=args.realerr_weight,
                        bin25_weight=args.bin25_weight).item()
                    if vloss == vloss and abs(vloss) != float("inf"):
                        vs += vloss; vn += 1
            val_mse = vs / max(vn, 1)
            net.train()
        else:
            val_mse = float("inf")
        val_mse = _broadcast_float(val_mse, device, is_ddp, src=0)

        lr = scheduler.get_last_lr()[0]
        log(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.6f} | "
            f"val {val_mse:.6f} | lr {lr:.2e} | bad {n_bad} | {time.time()-t0:.1f}s")

        if val_mse < best_val:
            best_val = val_mse
            if is_rank0:
                save_ckpt(epoch, val_mse, out_dir / "best_model.pt", full=False)
            log(f"  -> new best ({val_mse:.6f})")
        if is_rank0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            save_ckpt(epoch, val_mse, out_dir / "latest_checkpoint.pt", full=True)
            save_ema(out_dir / "ema_model.pt")
        if is_ddp:
            dist.barrier()

    if is_rank0:
        save_ckpt(args.epochs, val_mse, out_dir / "final_model.pt", full=False)
        save_ema(out_dir / "ema_model.pt")
    log(f"Done. Best val {best_val:.6f}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
