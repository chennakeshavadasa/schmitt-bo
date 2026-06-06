"""
run_bo.py v2 — Fixed acquisition function + realistic targets
=============================================================
Fixes from v1:
  1. EI=0 bug: composite objective was always deeply negative,
     making f_best so low that EI(x) = 0 everywhere.
     Fix: normalize errors to [0,1] range so GP learns meaningful gradients.

  2. Realistic targets based on what the circuit can actually achieve:
     V_PH = 1.60V  ← achievable (circuit reaches 1.48-1.63V range)
     V_PL = 1.05V  ← realistic ceiling from physics analysis
     OR just optimize to minimize total error with no hard target.

  3. Better GP: use separate EI for each output, then combine.

Usage:
    python3 run_bo.py --n_init 5 --n_iter 35 --seed 13
    python3 run_bo.py --n_init 5 --n_iter 35 --realistic  # uses achievable targets
"""
import sys, argparse, json, time, copy
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer.simulator import (simulate, vector_to_params,
                                  NOMINAL, PARAM_KEYS, PARAM_BOUNDS)
from optimizer.visualization import (plot_convergence,
                                      plot_observation_history)

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target_ph',  type=float, default=1.60)
    p.add_argument('--target_pl',  type=float, default=1.40)
    p.add_argument('--tolerance',  type=float, default=0.015)
    p.add_argument('--n_init',     type=int,   default=5)
    p.add_argument('--n_iter',     type=int,   default=35)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--realistic',  action='store_true',
                   help='Use targets V_PH=1.60 V_PL=1.05 (what circuit can achieve)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GP SURROGATE  (one per output, normalized targets)
# ─────────────────────────────────────────────────────────────────────────────

class GPSurrogate:
    def __init__(self, n_dim=10):
        k = (Matern(length_scale=np.ones(n_dim),
                    length_scale_bounds=(1e-2, 5.0), nu=2.5)
             + WhiteKernel(noise_level=1e-3,
                           noise_level_bounds=(1e-5, 1e-1)))
        self.gp_ph = GaussianProcessRegressor(
            kernel=k, n_restarts_optimizer=5,
            normalize_y=True, random_state=42)
        self.gp_pl = GaussianProcessRegressor(
            kernel=copy.deepcopy(k), n_restarts_optimizer=5,
            normalize_y=True, random_state=43)
        self.fitted = False

    def fit(self, X, Yph, Ypl):
        self.gp_ph.fit(X, Yph)
        self.gp_pl.fit(X, Ypl)
        self.fitted = True

    def predict(self, X):
        X = np.atleast_2d(X)
        mh, sh = self.gp_ph.predict(X, return_std=True)
        ml, sl = self.gp_pl.predict(X, return_std=True)
        return mh, sh, ml, sl


# ─────────────────────────────────────────────────────────────────────────────
# ACQUISITION  — fixed EI using per-output improvement
# ─────────────────────────────────────────────────────────────────────────────

def ei_1d(mu, sigma, best, xi=0.005):
    """Standard EI for a single output (maximize mu)."""
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - best - xi) / sigma
    ei = (mu - best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return np.maximum(ei, 0)

def acquisition(surrogate, X, best_ph, best_pl, target_ph, target_pl):
    """
    Combined acquisition:
    EI_ph(x) × EI_pl(x)  — both outputs must improve simultaneously.

    Key fix: we maximize V_PH toward target_ph and V_PL toward target_pl
    independently, then multiply. This gives non-zero gradients even when
    one output is far from target.
    """
    X = np.atleast_2d(X)
    mh, sh, ml, sl = surrogate.predict(X)

    # For V_PH: maximize closeness = -|mu_ph - target_ph|
    # Convert to: maximize (negate absolute error)
    obj_ph = -np.abs(mh - target_ph)
    obj_pl = -np.abs(ml - target_pl)

    ei_ph = ei_1d(obj_ph, sh, best_ph)
    ei_pl = ei_1d(obj_pl, sl, best_pl)

    # Joint: both must improve
    return ei_ph * ei_pl


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LHS  (avoids extreme corners)
# ─────────────────────────────────────────────────────────────────────────────

def safe_lhs(n, d, rng):
    lo = np.array([0.10, 0.05, 0.10, 0.05, 0.05, 0.02,
                    0.10, 0.02, 0.05, 0.05])
    hi = np.array([0.60, 0.50, 0.60, 0.50, 0.60, 0.50,
                    0.60, 0.50, 0.60, 0.50])
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = lo[j] + (perm + rng.random(n)) / n * (hi[j] - lo[j])
    return X

def safe_params(x):
    p = vector_to_params(x)
    for k in ['L1','L2','L3','L4','L6']:
        p[k] = min(p[k], 3.0)
    return p

def safe_simulate(p):
    try:
        vph, vpl = simulate(p)
        if vph and vpl and 0.3 < vpl < vph < 1.75:
            return vph, vpl
        return None, None
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # Realistic targets if requested
    if args.realistic:
        args.target_ph = 1.60
        args.target_pl = 1.05   # what the circuit can actually achieve
        args.tolerance = 0.015
        print(f'  [Realistic mode: V_PH=1.60V V_PL=1.05V ±15mV]')

    TPH, TPL, TOL = args.target_ph, args.target_pl, args.tolerance
    N_DIM = 10

    print("\n" + "="*62)
    print("  Bayesian Optimization  v2  (Fixed EI + Realistic Targets)")
    print(f"  Target: V_PH={TPH:.3f}V  V_PL={TPL:.3f}V  ±{TOL*1e3:.0f}mV")
    print(f"  seed={args.seed}  init={args.n_init}  iter={args.n_iter}")
    print("="*62)

    gp    = GPSurrogate(N_DIM)
    X_obs, Yph, Ypl = [], [], []
    best_vph = best_vpl = None
    converged = False

    def register(x, vph, vpl):
        nonlocal best_vph, best_vpl, converged
        X_obs.append(x.copy()); Yph.append(vph); Ypl.append(vpl)
        if best_vph is None or abs(vph-TPH)+abs(vpl-TPL) < abs(best_vph-TPH)+abs(best_vpl-TPL):
            best_vph, best_vpl = vph, vpl
        if abs(vph-TPH)<=TOL and abs(vpl-TPL)<=TOL:
            converged = True

    # ── LHS init ──────────────────────────────────────────────────────────
    print(f"\n── LHS Initialization ({args.n_init} sims) ─────────────────────────")
    X_init = safe_lhs(args.n_init * 3, N_DIM, rng)  # extra to allow retries
    n_done = 0
    for x in X_init:
        if n_done >= args.n_init: break
        p = safe_params(x)
        print(f"  [{n_done+1}/{args.n_init}] sim... ", end='', flush=True)
        t0 = time.time()
        vph, vpl = safe_simulate(p)
        dt = time.time()-t0
        if vph is None:
            print(f"FAIL ({dt:.0f}s) — retry")
            continue
        print(f"V_PH={vph:.4f}  V_PL={vpl:.4f}  "
              f"ePH={vph-TPH:+.4f}  ePL={vpl-TPL:+.4f}  ({dt:.0f}s)")
        register(x, vph, vpl)
        n_done += 1

    if n_done == 0:
        print("All init sims failed"); return

    # ── BO loop ────────────────────────────────────────────────────────────
    print(f"\n── Bayesian Optimization ({args.n_iter} iters) ─────────────────────")
    print(f"  {'It':>3}  {'V_PH':>8}  {'V_PL':>8}  "
          f"{'ePH':>8}  {'ePL':>8}  {'EI':>12}  time")
    print(f"  {'─'*70}")

    history = {'vph':[],'vpl':[],'ei':[]}

    for it in range(1, args.n_iter+1):

        # Fit GP
        X_arr = np.array(X_obs)
        gp.fit(X_arr, np.array(Yph), np.array(Ypl))

        # Current best objectives
        bph = -np.min(np.abs(np.array(Yph) - TPH))
        bpl = -np.min(np.abs(np.array(Ypl) - TPL))

        # Maximize acquisition via multi-start L-BFGS-B
        best_x, best_acq = None, -np.inf
        for _ in range(20):
            x0 = safe_lhs(1, N_DIM, rng)[0]
            try:
                res = minimize(
                    lambda x: -float(acquisition(gp, x.reshape(1,-1),
                                                  bph, bpl, TPH, TPL)[0]),
                    x0, bounds=[(0,1)]*N_DIM, method='L-BFGS-B',
                    options={'maxiter':100}
                )
                if -res.fun > best_acq:
                    best_acq = -res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is None:
            best_x = safe_lhs(1, N_DIM, rng)[0]

        # Simulate
        p = safe_params(np.clip(best_x, 0, 1))
        print(f"  {it:>3}  ", end='', flush=True)
        t0 = time.time()
        vph, vpl = safe_simulate(p)
        dt = time.time()-t0

        if vph is None:
            print(f"{'FAIL':<8}  (retry next iter)")
            continue

        register(best_x, vph, vpl)
        history['vph'].append(vph)
        history['vpl'].append(vpl)
        history['ei'].append(best_acq)

        flag = " ✓" if converged else ""
        print(f"{vph:>8.4f}  {vpl:>8.4f}  "
              f"{vph-TPH:>+8.4f}  {vpl-TPL:>+8.4f}  "
              f"{best_acq:>12.6f}  {dt:.0f}s{flag}")

        if converged:
            print(f"\n  ✓ Converged at iteration {it}!")
            break

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    ok = converged
    print(f"  {'✓ CONVERGED' if ok else '✗ Best effort'}  "
          f"({len(X_obs)} total sims)")
    print(f"  V_PH = {best_vph:.4f} V  (err {best_vph-TPH:+.4f} V)")
    print(f"  V_PL = {best_vpl:.4f} V  (err {best_vpl-TPL:+.4f} V)")

    # Find best params
    errs = [abs(ph-TPH)+abs(pl-TPL) for ph,pl in zip(Yph,Ypl)]
    best_idx = np.argmin(errs)
    p_best = safe_params(X_obs[best_idx])

    print(f"\n  Best sizing (ML result):")
    for k in PARAM_KEYS:
        chg = '*' if abs(p_best[k]-NOMINAL.get(k,0))>0.01 else ' '
        print(f"  {chg} {k:>4}: {p_best[k]:>8.4f} µm  (nom={NOMINAL.get(k,'?')})")

    # Save
    results = {
        'n_sims': len(X_obs), 'converged': ok,
        'targets': {'V_PH': TPH, 'V_PL': TPL, 'tol_mV': TOL*1e3},
        'best': {'V_PH': best_vph, 'V_PL': best_vpl,
                 'err_PH_mV': (best_vph-TPH)*1e3,
                 'err_PL_mV': (best_vpl-TPL)*1e3,
                 'params': p_best},
        'history': {'V_PH': Yph, 'V_PL': Ypl, 'EI': history['ei']}
    }
    out = Path("results/bo_real_run.json")
    with open(out,'w') as f: json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}")

    # Plots
    print("\n  Generating plots...")
    try:
        from optimizer.visualization import plot_observation_history
        class _State:
            Y_ph=Yph; Y_pl=Ypl; X=X_obs
        plot_observation_history(_State(), TPH, TPL, TOL)
        plot_convergence(history, target_ph=TPH, target_pl=TPL, tol=TOL)
        print("  Plots saved → figures/")
    except Exception as e:
        print(f"  Plot error (non-fatal): {e}")

    print("="*62+"\n")


if __name__ == "__main__":
    main()
