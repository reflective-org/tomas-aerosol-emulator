"""Rollout evaluation for the TOMAS emulator.

Loads a trained checkpoint, autoregressively rolls it out over the full 6-hour
horizon on the held-out validation scenarios, and reports per-channel median
relative error (and pair counts) at 1 h / 3 h / 6 h.

The error metric masks bins that sit at the empty-bin floor for most of the
horizon (those carry no physical signal — see docs/DATA.md and the README's
"Why the floor matters" note), so the reported median reflects bins that are
actually populated.

Example
-------
    python evaluate.py \
        --run-dir checkpoints \
        --data-dir data/v4_repr_1M \
        --n-scenarios 500
"""

import argparse

import numpy as np
import torch

from dataset_v4 import (
    load_v4_batches, build_state_array_v4, transform_cond, NormStatsV4, N_BINS, LOG_EPS,
)
from model_multihead import TomasMultiHeadMLP
from model_conserved_v2 import ConservativeMultiHeadMLPV2
from metrics import log_state_to_physical, split_train_val

HORIZON_LABELS = {60: "1h", 180: "3h", 360: "6h"}


def build_model(run_dir, device):
    """Reconstruct the conservation-projected model from a checkpoint's saved args."""
    norm_stats = NormStatsV4.load(f"{run_dir}/norm_stats.pt")
    ckpt = torch.load(f"{run_dir}/best_model.pt", weights_only=True, map_location=device)
    ma = ckpt["args"]
    base = TomasMultiHeadMLP(
        hidden_dim=ma["hidden_dim"], n_backbone_layers=ma.get("n_layers", 5),
        head_hidden_dim=ma.get("head_hidden_dim", 256),
        nuc_head_hidden_dim=ma.get("nuc_head_hidden_dim"),
        dropout=ma.get("dropout", 0.0), n_cond=5,
        core_end_excl=ma.get("core_end_excl", 26),
    ).to(device)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()

    lfc = ma.get("log_factor_clamp", 2.302585)
    lfc = lfc if lfc and lfc > 0 else None
    n_iter = ma.get("proj_n_iter", 3)
    tn_clamp = ma.get("total_n_log_clamp", 0.0)
    tn_clamp = tn_clamp if tn_clamp and tn_clamp > 0 else None
    model = ConservativeMultiHeadMLPV2(
        base, norm_stats, log_factor_clamp=lfc, n_iter=n_iter,
        total_n_log_clamp=tn_clamp,
    ).to(device)
    model.eval()
    return model, norm_stats, ckpt, ma


def channel_median_relerr(pred, gt, floor_tol, empty_frac):
    """Median relative error over bins that are populated for most of the horizon."""
    all_rel = []
    sat = 0
    for b in range(N_BINS):
        g = gt[:, b]
        p = pred[:, b]
        pos = g[g > 10 * LOG_EPS]
        if pos.size < 10:
            continue
        phys_floor = pos.min()
        near_floor = g <= floor_tol * phys_floor
        if near_floor.mean() > empty_frac:   # bin spends most of the horizon at the floor
            sat += 1
            continue
        keep = ~near_floor
        if keep.sum() < 10:
            continue
        all_rel.append(np.abs(p[keep] - g[keep]) / g[keep])
    all_rel = np.concatenate(all_rel) if all_rel else np.array([])
    finite = all_rel[np.isfinite(all_rel)] if all_rel.size else all_rel
    med = float(np.median(finite)) if finite.size else float("nan")
    return med, int(all_rel.size), sat


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="checkpoints",
                   help="dir with best_model.pt + norm_stats.pt")
    p.add_argument("--data-dir", default="data/v4_repr_1M",
                   help="root of v4 NPZ batches (t*/batch_*.npz)")
    p.add_argument("--n-scenarios", type=int, default=500,
                   help="number of held-out scenarios to evaluate")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="must match the value used at training time")
    p.add_argument("--seed", type=int, default=0,
                   help="must match the value used at training time")
    p.add_argument("--state-clamp", type=float, default=20.0)
    p.add_argument("--floor-tol", type=float, default=1.5)
    p.add_argument("--empty-frac", type=float, default=0.5)
    p.add_argument("--horizons", type=int, nargs="+", default=[60, 180, 360])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device

    print("Loading data...")
    Nk, Mk, Gc, cond = load_v4_batches(args.data_dir, verbose=True)
    n_total = Nk.shape[0]
    _, val_idx = split_train_val(n_total, val_frac=args.val_frac, seed=args.seed)
    eval_idx = val_idx[:args.n_scenarios]

    log_states = build_state_array_v4(Nk[eval_idx], Mk[eval_idx], Gc[eval_idx])
    cond_t = transform_cond(cond[eval_idx])

    model, norm_stats, ckpt, ma = build_model(args.run_dir, device)
    print(f"Model: epoch-{ckpt['epoch']}, val_mse {ckpt.get('val_mse'):.6f}, "
          f"chunk_len={ma.get('chunk_len')}, pf={ma.get('pushforward_steps')}, "
          f"relerr_weight={ma.get('relerr_weight')}")

    states_norm = norm_stats.normalize_state(
        torch.as_tensor(log_states, dtype=torch.float32)).to(device)
    cond_norm = norm_stats.normalize_cond(
        torch.as_tensor(cond_t, dtype=torch.float32)).to(device)

    horizons = sorted(args.horizons)
    print(f"Rolling out to step {horizons[-1]}...")
    current = states_norm[:, 0, :]
    pred_at = {}
    with torch.no_grad():
        for t in range(horizons[-1]):
            x = torch.cat([current, cond_norm[:, t, :]], dim=-1)
            current = model(x).clamp(-args.state_clamp, args.state_clamp)
            if (t + 1) in horizons:
                pred_at[t + 1] = current.clone()

    def relerr_at(pred_log_norm, gt_log_norm):
        pred_log = norm_stats.denormalize_state(pred_log_norm).cpu()
        gt_log = norm_stats.denormalize_state(gt_log_norm).cpu()
        pred_Nk, pred_Mk, _ = log_state_to_physical(pred_log)
        gt_Nk, gt_Mk, _ = log_state_to_physical(gt_log)
        nk = channel_median_relerr(pred_Nk.numpy(), gt_Nk.numpy(), args.floor_tol, args.empty_frac)
        mk = channel_median_relerr(pred_Mk.numpy(), gt_Mk.numpy(), args.floor_tol, args.empty_frac)
        return nk, mk

    print()
    print(f"{'horizon':>8s}  {'Nk_med':>8s} {'Nk_n':>8s} {'Nk_sat':>7s}   "
          f"{'Mk_med':>8s} {'Mk_n':>8s} {'Mk_sat':>7s}")
    for h in horizons:
        (nk_med, nk_n, nk_sat), (mk_med, mk_n, mk_sat) = relerr_at(pred_at[h], states_norm[:, h, :])
        label = HORIZON_LABELS.get(h, f"{h}s")
        print(f"{label:>8s}  {nk_med:8.4f} {nk_n:>8,d} {nk_sat:>7d}   "
              f"{mk_med:8.4f} {mk_n:>8,d} {mk_sat:>7d}")


if __name__ == "__main__":
    main()
