"""Train the non-autoregressive AXIAL trajectory operator (model_axial_traj).
seq2seq: state0 + full forcing -> 12 Δt=30 states in ONE forward (no AR). logmae+realerr,
deep-supervised on all 12 GT states (the t=1 output supervises per-step physics directly).
DDP, equal per-rank caps (load_v4_batches gives disjoint UNEQUAL shards)."""
import math
import os, sys, time, argparse
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from dataset_v4 import (load_v4_batches, build_state_array_v4, transform_cond,
                        NormStatsV4, compute_norm_stats_v4, N_STATE)
from model_axial_traj import AxialTrajOperator, N_OUT, DT_MIN
from burst_feats import dunne_logJ
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2

TP = list(range(0, 361, DT_MIN))   # [0,30,...,360]

# Fourier Δt features used by the (optional, --distill-weight > 0) distillation
# champion, a Δt-conditioned MLP: 4 octaves x {sin, cos} = 8 dims + 5 base cond.
FOURIER_OCTAVES = [-2.0, -1.0, 0.0, 1.0]
N_DT_FEAT = 2 * len(FOURIER_OCTAVES)
N_COND_TOTAL = 5 + N_DT_FEAT


def fourier_dt_feat(d_minutes):
    """d_minutes: (B,) tensor in minutes -> (B, N_DT_FEAT) log-scale Fourier features."""
    log_d = torch.log(d_minutes.float().clamp(min=1e-6))
    octaves = torch.tensor(FOURIER_OCTAVES, dtype=torch.float32, device=d_minutes.device)
    scaled = (math.tau * log_d.unsqueeze(-1) * (2.0 ** octaves))
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


def build_bin_weights(gc, nuc, coarse, mk, device, burst=1.0):
    w = torch.ones(N_STATE, device=device)
    if nuc != 1.0: w[0:6] = nuc; w[40:46] = nuc
    if coarse != 1.0: w[26:40] = coarse; w[66:80] = coarse
    if burst != 1.0: w[0:8] = burst; w[40:48] = burst   # tail loss: upweight burst bins 0-7
    if mk != 1.0: w[40:80] = w[40:80] * mk
    if gc != 1.0: w[80] = gc
    return w / w.mean()


class TrajDataset(Dataset):
    def __init__(self, s0, cs, tg, ns, cs_extra=None):
        self.s0 = ns.normalize_state(torch.as_tensor(s0, dtype=torch.float32))
        cn = ns.normalize_cond(torch.as_tensor(cs, dtype=torch.float32))
        if cs_extra is not None:
            cn = torch.cat([cn, torch.as_tensor(cs_extra, dtype=torch.float32)], dim=-1)
        self.cs = cn
        self.tg = ns.normalize_state(torch.as_tensor(tg, dtype=torch.float32))
    def __len__(self): return self.s0.shape[0]
    def __getitem__(self, i): return self.s0[i], self.cs[i], self.tg[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/v4_repr_1M")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-scenarios", type=int, default=140000)
    p.add_argument("--per-rank-train", type=int, default=120000)
    p.add_argument("--per-rank-val", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--ff-experts", type=int, default=1, help="MoE-gated FFN with this many experts (1=plain)")
    p.add_argument("--distill-weight", type=float, default=0.0, help="low-weight champion semigroup-consistency aux")
    p.add_argument("--champion", type=str, default="runs/mlp_champion/best_model.pt",
                   help="Δt-conditioned MLP champion checkpoint (only needed with --distill-weight > 0; not shipped)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--time-grid", type=str, default="", help="comma minutes for output grid (dense-early); empty=uniform Δt=30")
    p.add_argument("--burst-feats", action="store_true", help="append analytic Dunne logJ as a per-timestep cond feature")
    p.add_argument("--burst-weight", type=float, default=1.0, help="tail loss: upweight burst bins 0-7")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--huber-beta", type=float, default=0.2)
    p.add_argument("--aux-mse-weight", type=float, default=0.1)
    p.add_argument("--realerr-weight", type=float, default=0.2)
    p.add_argument("--relerr-clip-max", type=float, default=5.0)
    p.add_argument("--gc-weight", type=float, default=4.0)
    p.add_argument("--nuc-weight", type=float, default=2.5)
    p.add_argument("--coarse-weight", type=float, default=1.0)
    p.add_argument("--mk-weight", type=float, default=1.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default="auto")
    p.add_argument("--warm-start", type=str, default="", help="init weights from this checkpoint (fresh optimizer) if no resume ckpt")
    p.add_argument("--extra-dirs", type=str, default="", help="comma-sep augmentation roots MIXED into TRAIN only (val stays pure CESM)")
    p.add_argument("--extra-cap", type=int, default=0, help="max scenarios per extra dir (per rank, 0=all)")
    args = p.parse_args()

    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0)); world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}" if is_ddp else "cuda"
    torch.cuda.set_device(local_rank if is_ddp else 0)
    is_rank0 = rank == 0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    def log(*a):
        if is_rank0: print(" ".join(str(x) for x in a), flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    log(f"AXIAL traj operator (non-AR). DDP ws={world_size}. d={args.d_model} blocks={args.num_blocks}")

    GRID = [int(x) for x in args.time_grid.split(",")] if args.time_grid else list(range(DT_MIN, 361, DT_MIN))
    TPg = [0] + GRID; n_out = len(GRID)
    rng = np.random.default_rng(args.seed + rank)
    Nk, Mk, Gc, cond = load_v4_batches(args.data_dir, max_scenarios=args.max_scenarios,
                                       rank=rank, world_size=world_size, verbose=is_rank0)
    log_states = build_state_array_v4(Nk, Mk, Gc); cond_t = transform_cond(cond)
    burst_lj = dunne_logJ(cond[:, TPg, :]) if args.burst_feats else None   # from RAW cond (n, T+1)
    del Nk, Mk, Gc, cond
    state0 = log_states[:, 0, :]; targets = log_states[:, TPg[1:], :]; cond13 = cond_t[:, TPg, :]
    del log_states, cond_t

    n_total = state0.shape[0]; perm = rng.permutation(n_total)
    n_val = min(args.per_rank_val, max(1, int(n_total * 0.05))); val_sel = perm[:n_val]
    local_avail = n_total - n_val
    if is_ddp:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        t = torch.tensor([local_avail], device=device); dist.all_reduce(t, op=dist.ReduceOp.MIN)
        gmin = int(t.item())
    else:
        gmin = local_avail
    eff = min(args.per_rank_train, gmin); train_sel = perm[n_val:n_val + eff]
    print(f"  rank {rank}: shard {n_total:,} -> train {len(train_sel):,} (eff {eff:,}) val {len(val_sel):,}", flush=True)

    # ---- TRAIN pool = CESM train + augmentation (mixed). VAL stays pure CESM (no-regression signal). ----
    tr_s0, tr_c13, tr_tg = state0[train_sel], cond13[train_sel], targets[train_sel]
    n_cesm_tr = tr_s0.shape[0]
    if args.extra_dirs:
        for ed in [e for e in args.extra_dirs.split(",") if e]:
            eNk, eMk, eGc, ec = load_v4_batches(ed, max_scenarios=(args.extra_cap or None),
                                                rank=rank, world_size=world_size, verbose=is_rank0)
            els = build_state_array_v4(eNk, eMk, eGc); ect = transform_cond(ec)
            tr_s0 = np.concatenate([tr_s0, els[:, 0, :]])
            tr_tg = np.concatenate([tr_tg, els[:, TPg[1:], :]])
            tr_c13 = np.concatenate([tr_c13, ect[:, TPg, :]])
            del eNk, eMk, eGc, ec, els, ect
        o = rng.permutation(tr_s0.shape[0]); tr_s0, tr_c13, tr_tg = tr_s0[o], tr_c13[o], tr_tg[o]
        if is_ddp:   # equalize combined train count across ranks (avoid DDP desync on unequal #batches)
            tt = torch.tensor([tr_s0.shape[0]], device=device); dist.all_reduce(tt, op=dist.ReduceOp.MIN)
            cap = int(tt.item()); tr_s0, tr_c13, tr_tg = tr_s0[:cap], tr_c13[:cap], tr_tg[:cap]
        log(f"  rank {rank}: MIXED train {tr_s0.shape[0]:,} (CESM {n_cesm_tr:,} + aug {tr_s0.shape[0]-n_cesm_tr:,})")

    saved_ns = out_dir / "norm_stats.pt"
    if saved_ns.is_file():
        norm_stats = NormStatsV4.load(saved_ns); log(f"Loaded norm_stats {saved_ns}")
    elif args.extra_dirs:   # MIXED distribution -> recompute norm on the combined train pool
        norm_stats = compute_norm_stats_v4(np.stack([tr_s0, tr_tg[:, 0, :]], 1), tr_c13)
        if is_rank0: norm_stats.save(saved_ns); log("recomputed norm_stats on MIXED train pool")
    else:
        # pure-CESM training: reuse the shipped stats so fine-tunes stay in the
        # released model's normalization (recomputing shifts state_mean and
        # breaks warm-started weights -- see README on norm-stat freezing)
        champ_ns = Path("checkpoints/axial_aug5/norm_stats.pt")
        if champ_ns.is_file():
            norm_stats = NormStatsV4.load(champ_ns)
            if is_rank0:
                import shutil; shutil.copy(champ_ns, saved_ns)
        else:
            norm_stats = compute_norm_stats_v4(np.stack([state0[train_sel], targets[train_sel,0,:]],1), cond13[train_sel])
            if is_rank0: norm_stats.save(saved_ns)

    phys = np.exp(tr_tg.reshape(-1, N_STATE))
    phys_floor = torch.as_tensor(np.maximum(np.percentile(phys, 1.0, axis=0), 1e-30).astype(np.float32), device=device)
    del phys

    ex_tr = ex_v = None
    if args.burst_feats:
        lj_mean = float(burst_lj[train_sel].mean()); lj_std = float(burst_lj[train_sel].std() + 1e-6)
        args.lj_mean = lj_mean; args.lj_std = lj_std   # saved in args for eval-time standardization
        ex_tr = ((burst_lj[train_sel] - lj_mean) / lj_std)[..., None]
        ex_v = ((burst_lj[val_sel] - lj_mean) / lj_std)[..., None]
        log(f"burst-feats ON: logJ appended (mean {lj_mean:.2f} std {lj_std:.2f})")
    train_ds = TrajDataset(tr_s0, tr_c13, tr_tg, norm_stats, ex_tr)
    val_ds = TrajDataset(state0[val_sel], cond13[val_sel], targets[val_sel], norm_stats, ex_v)
    del tr_s0, tr_c13, tr_tg
    del state0, cond13, targets
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    n_cond = 6 + (1 if args.burst_feats else 0)
    model = AxialTrajOperator(d_model=args.d_model, nhead=args.nhead, num_blocks=args.num_blocks,
                              dim_ff=args.dim_ff, dropout=args.dropout, ff_experts=args.ff_experts, n_out=n_out, n_cond=n_cond).to(device)
    log(f"params: {sum(p.numel() for p in model.parameters()):,} (ff_experts={args.ff_experts})")
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=(args.ff_experts > 1))

    champion = None
    if args.distill_weight > 0:
        cck = torch.load(args.champion, weights_only=False, map_location=device); ca = cck["args"]
        cbase = TomasMultiHeadMLP(hidden_dim=ca["hidden_dim"], n_backbone_layers=ca.get("n_layers", 5),
                                  head_hidden_dim=ca.get("head_hidden_dim", 256), nuc_head_hidden_dim=ca.get("nuc_head_hidden_dim", 512),
                                  dropout=0.0, n_cond=N_COND_TOTAL, core_end_excl=ca.get("core_end_excl", 26)).to(device)
        cbase.load_state_dict(cck["model_state_dict"])
        champion = ConservativeMultiHeadMLPV2(cbase, norm_stats, log_factor_clamp=ca.get("log_factor_clamp", 2.302585),
                                              n_iter=int(ca.get("proj_n_iter", 3))).to(device).eval()
        for q in champion.parameters(): q.requires_grad_(False)
        feat30 = fourier_dt_feat(torch.tensor([30.0], device=device))
        log(f"distillation ON: champion semigroup-consistency aux weight={args.distill_weight}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac)
    w = build_bin_weights(args.gc_weight, args.nuc_weight, args.coarse_weight, args.mk_weight, device, burst=args.burst_weight)

    def loss_fn(pred, tgt):
        pl = norm_stats.denormalize_state(pred).clamp(-60, 60); gl = norm_stats.denormalize_state(tgt).clamp(-60, 60)
        dd = pl - gl; ad = dd.abs(); b = args.huber_beta
        huber = torch.where(ad < b, 0.5 * dd * dd / b, ad - 0.5 * b)
        loss = (w * huber).mean()
        if args.aux_mse_weight > 0: loss = loss + args.aux_mse_weight * (w * (pred - tgt) ** 2).mean()
        if args.realerr_weight > 0:
            pp = torch.exp(pl); gg = torch.exp(gl)
            rel = torch.clamp((pp - gg).abs() / torch.clamp(gg, min=phys_floor), max=args.relerr_clip_max)
            loss = loss + args.realerr_weight * (w * rel).mean()
        return loss

    start_epoch = 0; best_val = float("inf"); ckpt_path = out_dir / "latest_checkpoint.pt"
    if args.resume == "auto" and ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        (model.module if is_ddp else model).load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"]); sched.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1; best_val = ck.get("best_val", float("inf")); log(f"resumed ep {ck['epoch']}")
    elif args.warm_start and Path(args.warm_start).is_file():
        wc = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd_w = wc["model_state_dict"]; cw = sd_w.get("cond_in.weight")
        tgt = (model.module if is_ddp else model).cond_in.weight
        if cw is not None and cw.shape[1] < tgt.shape[1]:
            pad = torch.zeros(tgt.shape[0], tgt.shape[1]-cw.shape[1], dtype=cw.dtype, device=cw.device)
            sd_w["cond_in.weight"] = torch.cat([cw, pad], dim=1)
            log(f"WARM-START: padded cond_in {cw.shape[1]}->{tgt.shape[1]} (fion zero-init)")
        missing, unexpected = (model.module if is_ddp else model).load_state_dict(sd_w, strict=False)
        log(f"WARM-START from {args.warm_start} (ep {wc.get('epoch')}); missing={len(missing)} unexpected={len(unexpected)}")
    args_d = vars(args).copy()
    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); tr = 0.0; nb = 0
        for s0, cs, tg in train_dl:
            s0 = s0.to(device, non_blocking=True); cs = cs.to(device, non_blocking=True); tg = tg.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(s0, cs)
            loss = loss_fn(pred, tg)
            if champion is not None:
                feat = feat30.expand(pred.shape[0], -1)
                prev = s0; dl = 0.0
                for k in range(N_OUT):
                    ch = champion(torch.cat([prev, cs[:, k, :], feat], dim=-1)).clamp(-20, 20)  # champion next-from-prev
                    dl = dl + (ch - pred[:, k, :]).abs().mean()
                    prev = pred[:, k, :]
                loss = loss + args.distill_weight * dl / N_OUT
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
            tr += loss.item(); nb += 1
        sched.step()
        model.eval(); vl = 0.0; vb = 0
        with torch.no_grad():
            for s0, cs, tg in val_dl:
                s0 = s0.to(device); cs = cs.to(device); tg = tg.to(device)
                vl += loss_fn(model(s0, cs), tg).item(); vb += 1
        tr /= max(nb, 1); vl /= max(vb, 1)
        if is_ddp:
            t = torch.tensor([vl], device=device); dist.all_reduce(t); vl = (t / world_size).item()
        log(f"Epoch {epoch:3d}/{args.epochs} | train {tr:.5f} | val {vl:.5f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")
        if is_rank0:
            sd = (model.module if is_ddp else model).state_dict()
            torch.save({"epoch": epoch, "model_state_dict": sd, "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(), "best_val": best_val, "val_mse": vl, "args": args_d}, ckpt_path)
            if vl < best_val:
                best_val = vl
                torch.save({"epoch": epoch, "model_state_dict": sd, "val_mse": vl, "args": args_d}, out_dir / "best_model.pt")
                log(f"  -> new best ({vl:.5f})")
    log("done.");
    if is_ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
