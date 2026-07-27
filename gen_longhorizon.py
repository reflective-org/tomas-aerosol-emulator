"""Generate 24h CESM-forced TOMAS ground truth in the v4 convention.

Everything in the repo stops at 6h, so long-horizon (chained) accuracy has
never been measurable. This produces the reference data for that.

Physics recipe is identical to the validated v4 wrapper (gen_idealized_heldout.py,
which reproduces v4 GT one-step to median 1e-6):
  - grid xk0=(pi/6)(1.7e-9)^3*1770, mass-doubling, 40 bins
  - make_step(["nucleation","coagulation","condensation"], cond_method="ppm_jit",
              nucl_scheme="ricco_dunne") at dt=60s
  - Gc[SRTSO4] RESET each minute to h2so4_kg(Hvmr(t),P,T); org_conc=0

Differences from gen_idealized_heldout.py, on purpose:
  - CESM-forced (real MAM4 ICs + time-varying T/RH/H2SO4), not idealized lognormals
  - 24h (1441 stored minutes, 25 hourly anchors) instead of 6h
  - stores the TRUE post-step Gc every minute. gen_idealized_heldout.py:113 writes
    a CONSTANT Gc0, which silently breaks the sulfur invariant on that data; this
    dataset is meant for conservation work so the real residual gas is required.

IC convention keeps the v4 sigma triple (1.6,1.8,1.6). That is the known-wrong
MAM4 value (correct is (1.8,1.6,1.8)) but it is what the emulator was trained on,
so using it keeps these ICs IN-DISTRIBUTION. Fixing it belongs to a v5 effort.

Run (GPU 3, driver libs via GPU-Operator path):
  CESM_HOURLY=/path/to/cesm/tseries/hour_1 python gen_longhorizon.py --n 4000 --out data/v4_long24h
"""
import os, glob, time, argparse
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import netCDF4 as nc
import jax, jax.numpy as jnp
from tomas_jax.core.config import ICOMP, SRTSO4
from tomas_jax.solvers.condensation import make_step

CESM_H1 = os.environ.get("CESM_HOURLY", "data/cesm_hourly")   # CESM h1 tseries dir
CESM_D1 = CESM_H1.replace("hour_1", "day_1")

RHO_SO4 = 1770.0; KB = 1.380649e-23; AVOGADRO = 6.02214076e23
MW_H2SO4 = 98.0; BOXVOL = 1e6
N_BINS = 40; NATIVE_DT = 60.0
SIGMA_TRAIN = (1.6, 1.8, 1.6)          # v4 training convention (see docstring)
STRATA = {"BL": range(61, 70), "FT": range(48, 61), "UT": range(36, 48)}
RH_MAX = 0.8                            # v4 metadata rh_max_filter

xk0 = (np.pi / 6.0) * (1.7e-9) ** 3 * RHO_SO4
XK = jnp.array(xk0 * (2.0 ** np.arange(N_BINS + 1, dtype=np.float64)))
_XKn = np.asarray(XK)

STEP = make_step(["nucleation", "coagulation", "condensation"],
                 cond_method="ppm_jit", nucl_scheme="ricco_dunne")


def h2so4_kg(vmr, P, T):
    return vmr * (P / (KB * T) * 1e-6) * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO


def _one_step(Nk, Mk_so4, P, nh3, T, RH, Hvmr, fion):
    """One 60s coupled step with gas reset. Returns next Nk, Mk_so4, and the
    POST-STEP residual gas (what the sulfur invariant needs)."""
    Mk = jnp.zeros((N_BINS, ICOMP)).at[:, SRTSO4].set(Mk_so4)
    Gc = jnp.zeros((ICOMP,)).at[SRTSO4].set(h2so4_kg(Hvmr, P, T))
    Nk2, Mk2, Gc2 = STEP(Nk, Mk, Gc, XK, T, P, BOXVOL, RH, 1.0, NATIVE_DT,
                         org_conc=0.0, nh3_conc=nh3, fion=fion)
    return Nk2, Mk2[:, SRTSO4], Gc2[SRTSO4]


_step_b = jax.jit(jax.vmap(_one_step, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))


def find_file(var, base=CESM_H1):
    return sorted(glob.glob(f"{base}/*.{var}.*.nc"))[0]


def xk_bounds(n=N_BINS):
    return xk0 * (2.0 ** np.arange(n + 1))


def distribute(Ntot, Mtot, dgn, sigma, xk):
    """MAM4 mode -> TOMAS sectional bins (verbatim from chain_axial_7day.py)."""
    dp = np.sqrt((6*xk[:-1]/(np.pi*RHO_SO4))**(1/3) * (6*xk[1:]/(np.pi*RHO_SO4))**(1/3))
    pdf = np.exp(-0.5*((np.log(dp)[None, :] - np.log(np.maximum(dgn, 1e-12))[:, None])/np.log(sigma))**2)
    pdf /= np.maximum(pdf.sum(1, keepdims=True), 1e-30)
    return Ntot[:, None]*pdf, Mtot[:, None]*pdf


def read_block(var, t0, n_h):
    with nc.Dataset(find_file(var)) as f:
        return np.asarray(f.variables[var][t0:t0+n_h+1], dtype=np.float32)   # (n_h+1,lev,lat,lon)


def sample_cells(rng, t0, n_want, n_hours):
    """Stratified BL/FT/UT cells whose 24h RH stays within the v4 training range."""
    over = 4
    cells = []
    for band, levs in STRATA.items():
        levs = list(levs)
        per = int(np.ceil(n_want / 3 * over))
        lv = rng.choice(levs, per); la = rng.integers(0, 192, per); lo = rng.integers(0, 288, per)
        cells += list(zip(lv, la, lo, [band]*per))
    lev = np.array([c[0] for c in cells]); lat = np.array([c[1] for c in cells]); lon = np.array([c[2] for c in cells])
    rh = np.clip(read_block('RELHUM', t0, n_hours)[:, lev, lat, lon].T / 100., 0.01, 0.99)  # (B,n_h+1)
    keep = rh.max(1) <= RH_MAX
    idx = np.where(keep)[0]
    rng.shuffle(idx)
    idx = idx[:n_want]
    return [cells[i] for i in idx], idx.size


def build_batch(rng, t0, cells, n_hours):
    """CESM ICs + per-minute forcing for the chosen cells."""
    B = len(cells)
    lev = np.array([c[0] for c in cells]); lat = np.array([c[1] for c in cells]); lon = np.array([c[2] for c in cells])
    bands = [c[3] for c in cells]

    with nc.Dataset(find_file('T')) as f:
        hyam = np.array(f.variables['hyam'][:]); hybm = np.array(f.variables['hybm'][:]); P0 = float(f.variables['P0'][:])
    with nc.Dataset(find_file('PS', CESM_D1)) as f:
        nday = f.variables['PS'].shape[0]; PS = np.array(f.variables['PS'][min(t0//24, nday-1)])
    P = hyam[lev]*P0 + hybm[lev]*PS[lat, lon]                                 # (B,)

    Th = read_block('T', t0, n_hours)[:, lev, lat, lon].T.astype(np.float64)          # (B,n_h+1)
    RHh = np.clip(read_block('RELHUM', t0, n_hours)[:, lev, lat, lon].T / 100., 0.01, RH_MAX).astype(np.float64)
    H2h = read_block('H2SO4', t0, n_hours)[:, lev, lat, lon].T.astype(np.float64)

    n_steps = n_hours * 60
    t_src = np.arange(n_hours + 1, dtype=np.float64)
    t_min = np.arange(n_steps + 1) / 60.0
    interp = lambda h: np.stack([np.interp(t_min, t_src, h[b]) for b in range(B)])
    T_m, RH_m, H2_m = interp(Th), interp(RHh), interp(H2h)                    # (B,n_steps+1)

    mamv = ['num_a1','num_a2','num_a3','so4_a1','so4_a2','so4_a3',
            'pom_a1','pom_a4','bc_a1','bc_a4','dgnumwet1','dgnumwet2','dgnumwet3']
    m = {}
    for v in mamv:
        with nc.Dataset(find_file(v)) as f:
            m[v] = np.asarray(f.variables[v][t0], dtype=np.float64)[lev, lat, lon]
    T00 = Th[:, 0]; kg_air = (P/(287.058*T00))*BOXVOL*1e-6
    Ma1 = (m['so4_a1']+m['pom_a1']+m['bc_a1']+m['pom_a4']+m['bc_a4'])*kg_air
    Ma2 = m['so4_a2']*kg_air; Ma3 = m['so4_a3']*kg_air
    Na1 = m['num_a1']*kg_air; Na2 = m['num_a2']*kg_air; Na3 = m['num_a3']*kg_air
    xk = xk_bounds()
    Nk1, Mk1 = distribute(Na1, Ma1, m['dgnumwet1'], SIGMA_TRAIN[0], xk)
    Nk2, Mk2 = distribute(Na2, Ma2, m['dgnumwet2'], SIGMA_TRAIN[1], xk)
    Nk3, Mk3 = distribute(Na3, Ma3, m['dgnumwet3'], SIGMA_TRAIN[2], xk)
    Nk0 = np.maximum(Nk1+Nk2+Nk3, 1e-20); Mk0 = np.maximum(Mk1+Mk2+Mk3, 1e-30)
    nh3 = 10.0**rng.uniform(-1, 2, size=B)          # v4 climatology span
    fion = np.full(B, 3.0)                           # CESM default (v4 convention)
    return dict(B=B, P=P, T_m=T_m, RH_m=RH_m, H2_m=H2_m, Nk0=Nk0, Mk0=Mk0,
                nh3=nh3, fion=fion, lev=lev, lat=lat, lon=lon, bands=bands,
                Th=Th, RHh=RHh, H2h=H2h)


def roll(s, n_steps, log_every=120):
    """Roll the batch n_steps minutes, storing TRUE post-step Gc."""
    B = s['B']
    Nk_traj = np.empty((B, n_steps+1, N_BINS), np.float32)
    Mk_traj = np.empty((B, n_steps+1, N_BINS), np.float32)
    Gc_traj = np.empty((B, n_steps+1), np.float32)
    curN = jnp.array(s['Nk0']); curM = jnp.array(s['Mk0'])
    Pj = jnp.array(s['P']); nh3j = jnp.array(s['nh3']); fionj = jnp.array(s['fion'])
    Nk_traj[:, 0] = np.asarray(curN); Mk_traj[:, 0] = np.asarray(curM)
    # t=0 gas is the reset value (no step has run yet) -- matches v4 convention
    Gc_traj[:, 0] = h2so4_kg(s['H2_m'][:, 0], s['P'], s['T_m'][:, 0])
    t0 = time.time()
    for t in range(n_steps):
        curN, curM, gc = _step_b(curN, curM, Pj, nh3j,
                                 jnp.array(s['T_m'][:, t]), jnp.array(s['RH_m'][:, t]),
                                 jnp.array(s['H2_m'][:, t]), fionj)
        Nk_traj[:, t+1] = np.asarray(curN); Mk_traj[:, t+1] = np.asarray(curM)
        Gc_traj[:, t+1] = np.asarray(gc)
        if (t+1) % log_every == 0:
            print(f"    minute {t+1}/{n_steps}  ({time.time()-t0:.0f}s)", flush=True)
    return Nk_traj, Mk_traj, Gc_traj


def invariant_check(Nk, Mk, Gc, s, n_check=60):
    """Per-minute sulfur closure on the generated data: should be ~1e-8."""
    Ms = Mk.astype(np.float64).sum(-1)
    H = h2so4_kg(s['H2_m'], s['P'][:, None], s['T_m'])
    r = []
    for t in range(min(n_check, Ms.shape[1]-1)):
        lhs = Ms[:, t+1] + Gc[:, t+1]
        rhs = Ms[:, t] + H[:, t]
        r.append(np.abs(lhs-rhs)/np.maximum(np.abs(rhs), 1e-30))
    a = np.concatenate(r); a = a[np.isfinite(a)]
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.percentile(a, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v4_long24h")
    ap.add_argument("--n", type=int, default=4000, help="total scenarios")
    ap.add_argument("--per-file", type=int, default=500)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--t-starts", type=int, nargs="+",
                    default=[131832, 133292, 134752, 136212, 137672, 139132],
                    help="CESM hour indices; defaults span the v4 training window")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--smoke", action="store_true", help="1 small batch + invariant check, no write")
    a = ap.parse_args()
    print("jax devices:", jax.devices(), flush=True)
    rng = np.random.default_rng(a.seed)
    n_steps = a.hours * 60

    if a.smoke:
        cells, got = sample_cells(rng, a.t_starts[0], 100, a.hours)
        print(f"[smoke] cells kept {got}", flush=True)
        s = build_batch(rng, a.t_starts[0], cells, a.hours)
        t0 = time.time()
        Nk, Mk, Gc = roll(s, n_steps)
        dt = time.time()-t0
        fin = np.isfinite(Nk.reshape(s['B'], -1)).all(1) & np.isfinite(Mk.reshape(s['B'], -1)).all(1)
        inv = invariant_check(Nk, Mk, Gc, s, n_check=n_steps)
        print(f"[smoke] B={s['B']} {a.hours}h in {dt:.0f}s ({dt/max(s['B'],1)*1000:.1f} ms/scen)")
        print(f"[smoke] finite {fin.sum()}/{s['B']}")
        print(f"[smoke] per-minute sulfur invariant: p50={inv[0]:.2e} p90={inv[1]:.2e} p99={inv[2]:.2e}")
        print(f"[smoke] GATE {'PASS' if inv[0] < 1e-6 else 'FAIL'} (p50 < 1e-6)")
        print(f"[smoke] projected wall time for {a.n} scen: {dt/max(s['B'],1)*a.n/60:.1f} min")
        return

    os.makedirs(a.out, exist_ok=True)
    per_start = int(np.ceil(a.n / len(a.t_starts)))
    total = 0
    for ti, t0h in enumerate(a.t_starts):
        remain = per_start
        fi = 0
        while remain > 0:
            nb = min(a.per_file, remain)
            cells, got = sample_cells(rng, t0h, nb, a.hours)
            if got == 0:
                print(f"  t={t0h}: no cells passed RH filter, skipping", flush=True); break
            s = build_batch(rng, t0h, cells, a.hours)
            print(f"[gen] t={t0h} file {fi}: rolling {s['B']} scenarios x {n_steps} min ...", flush=True)
            Nk, Mk, Gc = roll(s, n_steps)
            fin = (np.isfinite(Nk.reshape(s['B'], -1)).all(1)
                   & np.isfinite(Mk.reshape(s['B'], -1)).all(1) & np.isfinite(Gc).all(1))
            inv = invariant_check(Nk, Mk, Gc, s)
            print(f"  finite {fin.sum()}/{s['B']}  invariant p50={inv[0]:.2e} p90={inv[1]:.2e}", flush=True)
            k = np.where(fin)[0]
            outdir = os.path.join(a.out, f"t{t0h:08d}")
            os.makedirs(outdir, exist_ok=True)
            fn = os.path.join(outdir, f"batch_s{ti:02d}_b{fi:04d}.npz")
            hourly = lambda h: h[k].T.astype(np.float32)     # (n_h+1, B)
            np.savez(fn,
                     Nk=Nk[k], Mk_SO4=Mk[k], Gc_H2SO4=Gc[k],
                     pres=s['P'][k].astype(np.float32), nh3_conc=s['nh3'][k].astype(np.float32),
                     fion=s['fion'][k].astype(np.float32),
                     T_hourly=hourly(s['Th']), RH_hourly=hourly(s['RHh']),
                     H2SO4_vmr_hourly=hourly(s['H2h']),
                     lev_idx=s['lev'][k].astype(np.int16), lat_idx=s['lat'][k].astype(np.int16),
                     lon_idx=s['lon'][k].astype(np.int16), t_idx=np.int32(t0h),
                     hours=np.int32(a.hours))
            print(f"[gen] wrote {k.size} -> {fn}", flush=True)
            total += k.size; remain -= nb; fi += 1
    print(f"[gen] TOTAL {total} scenarios -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
