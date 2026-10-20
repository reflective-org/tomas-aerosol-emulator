"""Load a shipped checkpoint and run it: one 6-hour block, or a chained multi-day rollout.

    from inference import load_model, rollout
    model = load_model("checkpoints/bintf_d384_l6")          # fp16 autocast + torch.compile (CUDA graphs)
    traj  = rollout(model, state0_log, cond_log)              # (B, 12*K+1, 82) log-states every 30 min

Precision: fp16 autocast reproduces fp32 within rounding at every horizon (inputs are
normalized log-states, so range is not an issue). bf16 is NOT lossless (14-day tails widen)
and is refused. Batch size must be constant for a compiled graph; `rollout` pads the last batch.
Measured on one H200, bs 4096: d384 2.9 us/cell/6h, d256 1.1 us/cell/6h (fp16 compiled);
eager fp32 is 16 / 6 us.
"""
import os
import numpy as np
import torch
from bintf_model import BinTF, CHECKPOINTS, N_STATE, N_COND, T
from normalization import NormStats

HERE = os.path.dirname(os.path.abspath(__file__))


class Emulator:
    def __init__(self, model, norm, device, dtype, batch):
        self.model, self.norm, self.device, self.dtype, self.batch = model, norm, device, dtype, batch

    @torch.no_grad()
    def block(self, s_norm, cond_norm):
        """One 6-hour block on normalized inputs. s_norm (B,82), cond_norm (B,13,6) -> (B,12,82) normalized."""
        with torch.autocast(self.device, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            out = self.model(s_norm, cond_norm)
        return torch.nan_to_num(out.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)


def load_model(ckpt_dir, device="cuda", dtype="fp16", compile_model=True, batch=16384):
    """ckpt_dir: a directory under checkpoints/ (or its name). Returns an Emulator."""
    name = os.path.basename(os.path.normpath(ckpt_dir))
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}; shipped: {list(CHECKPOINTS)}")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = os.path.join(HERE, "checkpoints", name)
    d, nl = CHECKPOINTS[name]
    m = BinTF(d, nl).to(device).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, "avg_top3.pt"), map_location=device)["model"])
    if dtype not in ("fp16", "fp32"):
        raise ValueError("dtype must be fp16 or fp32 (bf16 widens the 14-day tails; see README)")
    dt = torch.float16 if (dtype == "fp16" and device == "cuda") else torch.float32
    fwd = torch.compile(m, mode="reduce-overhead") if (compile_model and device == "cuda") else m
    norm = NormStats.load(os.path.join(HERE, "checkpoints", "norm_stats.npz"))
    return Emulator(fwd, norm, device, dt, batch)


@torch.no_grad()
def rollout(emu, state0_log, cond_log, n_blocks=None):
    """state0_log: (B, 82) log-state at t=0 (normalization.state_to_log).
    cond_log: (B, 12*K+1, 6) forcing on the 30-min grid for K blocks (normalization.cond_to_log).
    Returns (B, 12*K+1, 82) log-states including the initial state; every 30 min, 6-h blocks chained."""
    s = torch.as_tensor(np.asarray(state0_log, dtype=np.float32), device=emu.device)
    c = torch.as_tensor(np.asarray(cond_log, dtype=np.float32), device=emu.device)
    K = n_blocks or (c.shape[1] - 1) // T
    assert c.shape[1] >= T * K + 1, f"need {T*K+1} forcing points for {K} blocks, got {c.shape[1]}"
    B = s.shape[0]; bs = emu.batch
    cur = emu.norm.norm_state(s); cn = emu.norm.norm_cond(c); outs = []
    for b in range(K):
        cb = cn[:, b * T:b * T + T + 1]; chunks = []
        for a in range(0, B, bs):                                       # fixed batch for the compiled graph
            sa, ca = cur[a:a + bs], cb[a:a + bs]; n = sa.shape[0]
            if n < bs and bs <= B:
                sa = torch.cat([sa, sa[:1].expand(bs - n, -1)]); ca = torch.cat([ca, ca[:1].expand(bs - n, -1, -1)])
            chunks.append(emu.block(sa, ca)[:n])
        o = torch.cat(chunks, 0); outs.append(o); cur = o[:, -1]
    traj = emu.norm.denorm_state(torch.cat(outs, 1))
    return torch.cat([s.unsqueeze(1), traj], 1).cpu().numpy()
