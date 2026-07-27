"""Fine-tune the axial trajectory operator for LONG-RANGE stability + physical consistency.

Measured motivation (eval_axial_chain.py on runs_axial_aug5, 24h TOMAS truth,
chain started from a TOMAS-consistent state at t=60):

    horizon   self-chained Nk p50/p90      GT-restarted Nk p50/p90
      4h          0.026 / 0.105                0.026 / 0.105
     10h          0.050 / 0.217                0.026 / 0.108
     19h          0.075 / 0.345                0.028 / 0.135

Per-window skill is FLAT out to 19h; essentially all long-range error is exposure
bias -- the model has never seen its own output as an input. aug5 also violates
sulfur monotonicity (sum(Mk) must be non-decreasing) in 59% of chained steps.

So this fine-tune adds exactly three things to the aug5 recipe:
  1. PUSHFORWARD over two consecutive 6h windows drawn from the 24h dataset:
     window 2 starts from the model's own (detached) window-1 output, supervised
     against TOMAS truth. This is the direct fix for the measured failure.
  2. INTERIOR STARTS: windows begin anywhere in [start_min, 24h-12h], so the
     initial-state distribution includes burst-relaxed, TOMAS-evolved states
     rather than only t=0.
  3. PHYSICAL CONSTRAINTS: the monotonicity + bin-geometry projection
     (model_conserved_axial, mode="mono") active in the forward pass, plus a soft
     one-sided sulfur-budget penalty. Training WITH the projection is what let the
     old AR champion cooperate with its constraint instead of fighting it.

6h CESM replay keeps the in-distribution 6h metric from regressing, and
norm_stats are copied from the warm-start run, never recomputed (recomputing on
a mixed pool is the known catastrophic-forgetting trap -- see aug3/aug5).

Launch: see orchestrator_axial_stable.sh
"""
import os, sys, time, argparse, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from dataset_v4 import (load_v4_batches, build_state_array_v4, transform_cond,
                        NormStatsV4, N_STATE, LOG_EPS)
from dataset_long import load_long_batches
from model_axial_traj import AxialTrajOperator, N_OUT, DT_MIN
from model_conserved_axial import ConservativeAxialTraj, sulfur_budget_penalty

WIN_STEPS = N_OUT            # 12 outputs per window
WIN_GRID = WIN_STEPS + 1     # 13 cond anchors


def build_bin_weights(gc, nuc, coarse, mk, device, burst=1.0):
    """Identical to train_axial_aug.build_bin_weights (kept local to avoid importing a script)."""
    w = torch.ones(N_STATE, device=device)
    if nuc != 1.0: w[0:6] = nuc; w[40:46] = nuc
    if coarse != 1.0: w[26:40] = coarse; w[66:80] = coarse
    if burst != 1.0: w[0:8] = burst; w[40:48] = burst
    if mk != 1.0: w[40:80] = w[40:80] * mk
    if gc != 1.0: w[80] = gc
    return w / w.mean()


class TwoWindowDataset(Dataset):
    """Two consecutive 6h windows on the 30-min grid, from the 24h trajectories.

    state/cond are pre-subsampled to the 30-min grid, so index g means minute 30*g.
    Item g yields window 1 = [g, g+12] and window 2 = [g+12, g+24].
    """
    def __init__(self, state_g, cond_g, ns, start_g, rng_pairs):
        self.s = ns.normalize_state(torch.as_tensor(state_g, dtype=torch.float32))   # (N,G,81)
        self.c = ns.normalize_cond(torch.as_tensor(cond_g, dtype=torch.float32))     # (N,G,C)
        G = self.s.shape[1]
        self.pairs = [(i, g) for i in range(self.s.shape[0])
                      for g in range(start_g, G - 2 * WIN_STEPS)]
        if rng_pairs is not None and rng_pairs < len(self.pairs):
            idx = np.random.default_rng(0).permutation(len(self.pairs))[:rng_pairs]
            self.pairs = [self.pairs[j] for j in idx]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, g = self.pairs[k]
        s0 = self.s[i, g]
        c1 = self.c[i, g:g + WIN_GRID]
        t1 = self.s[i, g + 1:g + WIN_GRID]
        c2 = self.c[i, g + WIN_STEPS:g + WIN_STEPS + WIN_GRID]
        t2 = self.s[i, g + WIN_STEPS + 1:g + WIN_STEPS + WIN_GRID]
        s0_w2 = self.s[i, g + WIN_STEPS]
        return s0, c1, t1, s0_w2, c2, t2


class ReplayDataset(Dataset):
    """Standard 6h CESM windows (no-regression anchor)."""
    def __init__(self, s0, cs, tg, ns):
        self.s0 = ns.normalize_state(torch.as_tensor(s0, dtype=torch.float32))
        self.cs = ns.normalize_cond(torch.as_tensor(cs, dtype=torch.float32))
        self.tg = ns.normalize_state(torch.as_tensor(tg, dtype=torch.float32))

    def __len__(self): return self.s0.shape[0]
    def __getitem__(self, i): return self.s0[i], self.cs[i], self.tg[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--warm-start", default="checkpoints/axial_aug5/best_model.pt")
    p.add_argument("--norm-from", default="checkpoints/axial_aug5/norm_stats.pt")
    p.add_argument("--long-dir", default="data/v4_long24h")
    p.add_argument("--replay-dir", default="data/v4_repr_1M")
    p.add_argument("--holdout", type=int, default=800, help="first N long scenarios reserved for eval")
    p.add_argument("--long-pairs", type=int, default=60000, help="cap on (scenario,start) training pairs per rank")
    p.add_argument("--replay-max", type=int, default=20000)
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--start-min", type=int, default=60,
                   help="earliest window start; t=0 states are bin-mass-inconsistent")
    p.add_argument("--pushforward-p", type=float, default=0.75,
                   help="probability window 2 starts from the model's own detached output")
    p.add_argument("--w2-weight", type=float, default=1.0)
    p.add_argument("--budget-weight", type=float, default=0.05)
    p.add_argument("--proj-mode", default="mono", choices=["none", "geom", "mono"])
    p.add_argument("--proj-iter", type=int, default=2)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
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
    p.add_argument("--resume", default="auto")
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
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    if is_ddp:
        dist.init_process_group("nccl")

    # ---- norm stats: ALWAYS inherited, never recomputed ----
    saved_ns = out_dir / "norm_stats.pt"
    if not saved_ns.is_file() and is_rank0:
        import shutil; shutil.copy(args.norm_from, saved_ns)
    if is_ddp: dist.barrier()
    ns = NormStatsV4.load(saved_ns)
    log(f"norm_stats inherited from {args.norm_from}")

    # ---- 24h data, subsampled to the 30-min grid ----
    Nk, Mk, Gc, cond, n_steps = load_long_batches(args.long_dir, rank=rank,
                                                  world_size=world_size, verbose=is_rank0)
    if world_size == 1:
        Nk, Mk, Gc, cond = Nk[args.holdout:], Mk[args.holdout:], Gc[args.holdout:], cond[args.holdout:]
    else:
        # holdout lives in the first file(s); rank 0 must drop its share
        if rank == 0:
            k = args.holdout // world_size + args.holdout % world_size
            Nk, Mk, Gc, cond = Nk[k:], Mk[k:], Gc[k:], cond[k:]
        else:
            k = args.holdout // world_size
            Nk, Mk, Gc, cond = Nk[k:], Mk[k:], Gc[k:], cond[k:]
    grid = np.arange(0, n_steps + 1, DT_MIN)
    state_g = np.log(np.maximum(
        np.concatenate([Nk[:, grid], Mk[:, grid], Gc[:, grid][..., None]], -1), LOG_EPS)).astype(np.float32)
    cond_g = transform_cond(cond[:, grid])
    del Nk, Mk, Gc, cond
    start_g = args.start_min // DT_MIN
    n_lval = max(1, len(state_g) // 20)
    long_val_ds = TwoWindowDataset(state_g[:n_lval], cond_g[:n_lval], ns, start_g, 600)
    long_ds = TwoWindowDataset(state_g[n_lval:], cond_g[n_lval:], ns, start_g, args.long_pairs)
    del state_g, cond_g
    log(f"long: {len(long_ds):,} (scenario,start) pairs per rank, starts from t={args.start_min}min")

    # ---- 6h CESM replay ----
    rNk, rMk, rGc, rcond = load_v4_batches(args.replay_dir, max_scenarios=args.replay_max,
                                           rank=rank, world_size=world_size, verbose=False)
    rls = build_state_array_v4(rNk, rMk, rGc); rct = transform_cond(rcond)
    del rNk, rMk, rGc, rcond
    TPg = list(range(0, 361, DT_MIN))
    n_val = min(1000, len(rls) // 5)
    perm = np.random.default_rng(0).permutation(len(rls))
    v_sel, t_sel = perm[:n_val], perm[n_val:]
    rep_tg = rls[t_sel][:, TPg[1:], :]
    replay_ds = ReplayDataset(rls[t_sel, 0, :], rct[t_sel][:, TPg, :], rep_tg, ns)
    val_ds = ReplayDataset(rls[v_sel, 0, :], rct[v_sel][:, TPg, :], rls[v_sel][:, TPg[1:], :], ns)
    del rls, rct
    log(f"replay: {len(replay_ds):,} train / {len(val_ds):,} val (pure CESM 6h) per rank")

    # equalize step counts across ranks so DDP never desyncs (unequal shards -> hang)
    n_long = torch.tensor([len(long_ds)], device=device)
    if is_ddp: dist.all_reduce(n_long, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(n_long.item()) // args.batch_size

    long_dl = DataLoader(long_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)
    replay_dl = DataLoader(replay_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    long_val_dl = DataLoader(long_val_ds, batch_size=64, shuffle=False, num_workers=2, drop_last=False)

    # ---- model ----
    ck = torch.load(args.warm_start, weights_only=False, map_location="cpu"); ma = ck["args"]
    n_cond = ck["model_state_dict"]["cond_in.weight"].shape[1]
    base = AxialTrajOperator(d_model=ma["d_model"], nhead=ma["nhead"], num_blocks=ma["num_blocks"],
                             dim_ff=ma["dim_ff"], dropout=ma.get("dropout", 0.0),
                             ff_experts=ma.get("ff_experts", 1), n_out=N_OUT, n_cond=n_cond).to(device)
    base.load_state_dict(ck["model_state_dict"])
    log(f"warm-start {args.warm_start} (ep {ck.get('epoch')}, val {ck.get('val_mse')}) "
        f"params {sum(q.numel() for q in base.parameters()):,}")
    net = base if args.proj_mode == "none" else ConservativeAxialTraj(
        base, ns, mode=args.proj_mode, n_iter=args.proj_iter).to(device)
    log(f"constraint projection: {args.proj_mode} (n_iter={args.proj_iter})")
    model = net
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=(ma.get("ff_experts", 1) > 1))

    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * args.lr_min_frac)
    w = build_bin_weights(args.gc_weight, args.nuc_weight, args.coarse_weight, args.mk_weight, device)
    _ph = np.exp(rep_tg.reshape(-1, N_STATE))
    phys_floor = torch.as_tensor(
        np.maximum(np.percentile(_ph, 1.0, axis=0), 1e-30).astype(np.float32), device=device)
    del _ph
    log_h2so4_const = float(np.log(1e-6*1e6*98e-3/6.022e23/1.380649e-23))
    smean = ns.state_mean.to(device); sstd = ns.state_std.to(device)

    def loss_fn(pred, tgt):
        pl = (pred * sstd + smean).clamp(-60, 60)
        gl = (tgt * sstd + smean).clamp(-60, 60)
        dd = pl - gl; ad = dd.abs(); b = args.huber_beta
        huber = torch.where(ad < b, 0.5*dd*dd/b, ad - 0.5*b)
        loss = (w * huber).mean()
        if args.aux_mse_weight > 0:
            loss = loss + args.aux_mse_weight * (w * (pred - tgt) ** 2).mean()
        if args.realerr_weight > 0:
            pp = torch.exp(pl); gg = torch.exp(gl)
            rel = torch.clamp((pp - gg).abs() / torch.clamp(gg, min=phys_floor),
                              max=args.relerr_clip_max)
            loss = loss + args.realerr_weight * (w * rel).mean()
        return loss

    start_epoch = 0; best = float("inf"); ckpt_path = out_dir / "latest_checkpoint.pt"
    if args.resume == "auto" and ckpt_path.is_file():
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        base.load_state_dict(c["model_state_dict"]); opt.load_state_dict(c["optimizer_state_dict"])
        sched.load_state_dict(c["scheduler_state_dict"])
        start_epoch = c["epoch"] + 1; best = c.get("best_val", float("inf"))
        log(f"resumed epoch {c['epoch']}")

    args_d = vars(args).copy(); args_d.update(
        {k: ma[k] for k in ("d_model", "nhead", "num_blocks", "dim_ff", "ff_experts", "dropout") if k in ma})
    args_d["time_grid"] = ""

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time()
        tr = pf_used = nb = 0
        rep_it = itertools.cycle(replay_dl)
        for bi, (s0, c1, t1, s0w2, c2, t2) in enumerate(long_dl):
            if bi >= steps_per_epoch: break
            s0 = s0.to(device, non_blocking=True); c1 = c1.to(device, non_blocking=True)
            t1 = t1.to(device, non_blocking=True); c2 = c2.to(device, non_blocking=True)
            t2 = t2.to(device, non_blocking=True); s0w2 = s0w2.to(device, non_blocking=True)
            opt.zero_grad()

            p1 = model(s0, c1)
            loss = loss_fn(p1, t1)
            loss = loss + args.budget_weight * sulfur_budget_penalty(
                p1, s0, c1, ns, log_h2so4_const)

            # --- pushforward: window 2 starts from the model's OWN output ---
            if np.random.rand() < args.pushforward_p:
                start2 = p1[:, -1, :].detach()
                pf_used += 1
            else:
                start2 = s0w2
            p2 = model(start2, c2)
            loss = loss + args.w2_weight * (
                loss_fn(p2, t2)
                + args.budget_weight * sulfur_budget_penalty(p2, start2, c2, ns, log_h2so4_const))

            # --- 6h CESM replay (no-regression anchor) ---
            if args.replay_weight > 0:
                rs0, rc, rt = next(rep_it)
                rs0 = rs0.to(device, non_blocking=True); rc = rc.to(device, non_blocking=True)
                rt = rt.to(device, non_blocking=True)
                loss = loss + args.replay_weight * loss_fn(model(rs0, rc), rt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), args.grad_clip)
            opt.step()
            tr += loss.item(); nb += 1
        sched.step()

        # ---- validation: PURE 6h CESM, the no-regression metric ----
        model.eval(); vl = 0.0; vb = 0
        with torch.no_grad():
            for rs0, rc, rt in val_dl:
                rs0 = rs0.to(device); rc = rc.to(device); rt = rt.to(device)
                vl += loss_fn(model(rs0, rc), rt).item(); vb += 1
        v = vl / max(vb, 1)
        # chained val: window 2 ALWAYS started from the model's own output (deployment mode)
        cl = 0.0; cb = 0
        with torch.no_grad():
            for s0, c1, t1, _s2, c2, t2 in long_val_dl:
                s0 = s0.to(device); c1 = c1.to(device); t1 = t1.to(device)
                c2 = c2.to(device); t2 = t2.to(device)
                q1 = model(s0, c1)
                q2 = model(q1[:, -1, :], c2)
                cl += (loss_fn(q1, t1) + loss_fn(q2, t2)).item() / 2.0; cb += 1
        vc = cl / max(cb, 1)
        if is_ddp:
            tv = torch.tensor([v, vc], device=device); dist.all_reduce(tv, op=dist.ReduceOp.SUM)
            v, vc = float(tv[0].item()) / world_size, float(tv[1].item()) / world_size
        sel = v + vc   # selection criterion: 6h skill AND chained skill must both hold
        log(f"Epoch {epoch:3d}/{args.epochs} | train {tr/max(nb,1):.5f} | val6h {v:.5f} | "
            f"valchain {vc:.5f} | pf {pf_used}/{nb} | lr {sched.get_last_lr()[0]:.2e} | "
            f"{time.time()-t0:.0f}s" + ("  -> new best" if sel < best else ""))
        if is_rank0:
            sd = base.state_dict()
            torch.save({"epoch": epoch, "model_state_dict": sd,
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "best_val": min(best, sel), "val_mse": v, "val_chain": vc,
                        "args": args_d}, ckpt_path)
            if sel < best:
                torch.save({"epoch": epoch, "model_state_dict": sd, "val_mse": v,
                            "val_chain": vc, "args": args_d}, out_dir / "best_model.pt")
        best = min(best, sel)
    log("done.")
    if is_ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
