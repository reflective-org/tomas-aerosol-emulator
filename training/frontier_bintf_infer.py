"""Deployment inference path for the bin-token transformers: fp16 autocast + torch.compile(reduce-overhead) (CUDA graphs).
Measured (FRONTIER_RESULTS §15): accuracy identical to fp32 within rounding at 6 h and over 14-day rollouts; speed 2.4x over eager bf16.
    from frontier_bintf_infer import load_bintf_fast
    step6h = load_bintf_fast(f"{EMU}/frontier_bintf_kchain_big_rebal/avg_top3.pt", 384, 6)   # d384 rebalanced, ~2.9 us/cell/6h
    s_next = step6h(s_norm, cond_norm)   # s_norm (B,82) normalized log-state, cond_norm (B,13,6) 30-min forcing incl. both endpoints
Batch size must be constant for a given compiled graph (pad the last batch); use bs 4096..16384. Returns the normalized 6 h state (B,82) in fp32.
Do NOT use bf16: it widens the 14-day tails (d256: day-14 totN p90 71 -> 84). TF32 is also lossless but gives no speed-up here."""
import torch, torch.nn as nn
from frontier_common import *
_tf = open("/data/tomas-emulator/frontier_transformers.py").read(); _ns = {"torch": torch, "nn": nn, "N_BINS": N_BINS, "P": N_BINS + 2, "T": 12}
exec(_tf[_tf.index("def enc("):_tf.index("torch.manual_seed(0)")], _ns); BinTF = _ns["BinTF"]

def load_bintf_fast(ckpt, d, nl, dtype=torch.float16, compile_mode="reduce-overhead"):
    m = BinTF(d, nl).to(dev).eval(); m.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    fwd = torch.compile(m, mode=compile_mode) if compile_mode else m
    @torch.no_grad()
    def step6h(s_norm, cond_norm):
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)): out = fwd(s_norm, cond_norm)[:, -1]
        return torch.nan_to_num(out.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)
    return step6h

if __name__ == "__main__":   # self-test: 1.5 M cells, one 6 h step, wall time
    import time
    D_ = int(os.environ.get("D", "384")); NL_ = int(os.environ.get("NL", "6")); CK = os.environ.get("CKPT", f"{EMU}/frontier_bintf_kchain_big_rebal/avg_top3.pt")
    f = load_bintf_fast(CK, D_, NL_); bs = int(os.environ.get("BS", "16384")); ncell = 1_500_000
    s = torch.randn(bs, 82, device=dev) * 0.1; c = torch.randn(bs, 13, 6, device=dev) * 0.1
    for _ in range(5): f(s, c)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(ncell // bs + 1): f(s, c)
    torch.cuda.synchronize(); dt = time.time() - t0
    log(f"{CK}: {ncell} cells x one 6 h step in {dt:.2f} s wall (bs {bs}; {dt / ncell * 1e6:.2f} us/cell) -- includes padding of the last batch")
