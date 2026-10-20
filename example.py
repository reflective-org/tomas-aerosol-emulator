"""Run one box through both shipped checkpoints: a 10 ppbv SO2 pulse into a dense stratospheric layer, 48 h.

    python example.py [--ckpt bintf_d384_l6|bintf_d256_l4] [--hours 48] [--cpu]
"""
import argparse, time
from math import erf
import numpy as np
import torch
from normalization import (state_to_log, log_to_state, cond_to_log, BOX, M_MID, D_EDGE_M, D_MID_UM, N_BINS)
from inference import load_model, rollout

KB, AV = 1.380649e-23, 6.02214076e23


def lognormal_bins(n_cm3, dg_nm, gsd):
    z = (np.log(D_EDGE_M) - np.log(dg_nm * 1e-9)) / (np.sqrt(2) * np.log(gsd))
    cdf = 0.5 * (1 + np.vectorize(erf)(z))
    return np.maximum(n_cm3 * BOX * np.diff(cdf), 1e-20)


def gas_kg(vmr, P, T, mw):
    return vmr * (P / (KB * T) * 1e-6) * BOX * (mw / 1000.0) / AV


def scenario(hours, T=205.0, P=5500.0, RH=0.05, oh_peak=3e6, n_cm3=300, dg_nm=80, gsd=1.6, so2_bg_ppt=20, so2_pulse_ppb=10):
    Nk = lognormal_bins(n_cm3, dg_nm, gsd); Mk = Nk * M_MID
    s0 = state_to_log(Nk, Mk, gas_kg(0.05e-12, P, T, 98.0), gas_kg((so2_bg_ppt * 1e-12 + so2_pulse_ppb * 1e-9), P, T, 64.0))
    t_h = np.arange(hours * 2 + 1) * 0.5
    oh = oh_peak * np.maximum(0, np.cos(2 * np.pi * t_h / 24.0))             # diurnal OH, peak at t = 0
    cond = cond_to_log(P, T, RH, oh)                                            # (13K, 6); no source, no dilution
    return s0[None], cond[None]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="bintf_d384_l6"); ap.add_argument("--hours", type=int, default=48); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    emu = load_model(a.ckpt, device=dev, dtype="fp16" if dev == "cuda" else "fp32", compile_model=False, batch=1)
    s0, cond = scenario(a.hours // 6 * 6)
    rollout(emu, s0, cond)                                                     # warm-up
    t0 = time.time(); traj = rollout(emu, s0, cond); ms = (time.time() - t0) * 1e3
    st = log_to_state(traj[0])
    N = st["Nk"].sum(1) / BOX; M = st["Mk"].sum(1) * 1e9
    print(f"{a.ckpt} on {dev}: {a.hours} h in {ms:.1f} ms")
    print(f"{'t [h]':>6} {'N [cm-3]':>10} {'mass [ug m-3]':>14} {'SO2 [ug/box]':>13} {'H2SO4 [ug/box]':>15}")
    for i in range(0, traj.shape[1], 12):
        print(f"{i*0.5:6.1f} {N[i]:10.3e} {M[i]:14.3f} {st['so2_kg'][i]*1e9:13.3f} {st['h2so4_kg'][i]*1e9:15.3e}")
    k = int(np.argmax(st["Nk"][-1])); print(f"most populated bin at the end: {k} (d = {D_MID_UM[k]*1e3:.0f} nm)")
