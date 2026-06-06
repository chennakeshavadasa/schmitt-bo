"""
run_bo.py
=========
Main Bayesian Optimization script for SKY130 Schmitt Trigger sizing.

Uses real ngspice simulations when available; exits with a clear error
message if ngspice is not found.

Usage
-----
    # Standard run (ambient targets)
    python experiments/run_bo.py --n_init 5 --n_iter 40 --seed 13

    # Realistic targets (what the 6T topology can actually achieve)
    python experiments/run_bo.py --n_init 5 --n_iter 40 --seed 13 --realistic

    # Custom targets
    python experiments/run_bo.py --target_ph 1.60 --target_pl 1.05

Expected runtime: ~35 s/simulation (BSIM4 model load dominates).
Full run: 5 init + 40 iter = 45 sims ~ 26 minutes.

If ngspice hangs on a bad sizing combination, kill with:
    pkill -f run_bo; pkill -f ngspice
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer.simulator import (
    simulate, vector_to_params, params_to_vector,
    NOMINAL, PARAM_KEYS, PARAM_BOUNDS,
)
from optimizer.bayesian_opt import (
    BayesianOptimizer, annealed_xi, smart_acquisition,
    TARGET_PH, TARGET_PL_DESIRED, TARGET_PL_REALISTIC, TOL,
)
from optimizer.visualization import (
    plot_convergence, plot_observation_history,
)

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bayesian Optimization for SKY130 Schmitt Trigger sizing"
    )
    p.add_argument("--target_ph",  type=float, default=1.60,
                   help="V_PH target voltage (default: 1.60)")
    p.add_argument("--target_pl",  type=float, default=1.40,
                   help="V_PL target voltage (default: 1.40)")
    p.add_argument("--tolerance",  type=float, default=0.015,
                   help="Convergence tolerance in V (default: 0.015)")
    p.add_argument("--n_init",     type=int,   default=5,
                   help="LHS initialization simulations (default: 5)")
    p.add_argument("--n_iter",     type=int,   default=40,
                   help="BO iterations after init (default: 40)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--realistic",  action="store_true",
                   help="Use V_PH=1.60 V_PL=1.05 (what the topology achieves)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Safe LHS init (avoids extreme parameter corners that cause BSIM4 hangs)
# ---------------------------------------------------------------------------

def safe_lhs(n, n_dim, rng):
    """
    Latin Hypercube Sampling restricted to the central 10-60% of each dim.
    Extreme corners (L4 < 0.15+eps, W4 > 30 etc.) often cause DC non-convergence.
    """
    lo = np.full(n_dim, 0.05)
    hi = np.full(n_dim, 0.60)
    X = np.zeros((n, n_dim))
    for j in range(n_dim):
        perm = rng.permutation(n)
        X[:, j] = lo[j] + (perm + rng.random(n)) / n * (hi[j] - lo[j])
    return X


def safe_simulate(p):
    """Run simulate() with basic sanity filtering."""
    try:
        vph, vpl = simulate(p)
        if vph and vpl and 0.3 < vpl < vph < 1.75:
            return vph, vpl
        return None, None
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    if args.realistic:
        args.target_ph = 1.60
        args.target_pl = 1.05
        args.tolerance = 0.015
        print("  [Realistic mode: V_PH=1.60 V V_PL=1.05 V +-15 mV]")

    TPH  = args.target_ph
    TPL  = args.target_pl
    TOL_ = args.tolerance
    N    = 10  # n_dim

    print("\n" + "=" * 65)
    print("  Bayesian Optimization -- SKY130 6T CMOS Schmitt Trigger")
    print(f"  Target: V_PH={TPH:.3f} V  V_PL={TPL:.3f} V  +-{TOL_*1e3:.0f} mV")
    print(f"  seed={args.seed}  n_init={args.n_init}  n_iter={args.n_iter}")
    print("=" * 65)

    opt = BayesianOptimizer(n_dim=N, tol=TOL_,
                             n_restarts_acq=20, n_restarts_gp=10)
    history = {"vph": [], "vpl": [], "ei": [], "xi": []}

    # -----------------------------------------------------------------------
    # LHS initialization
    # -----------------------------------------------------------------------
    print(f"\n-- LHS Initialization ({args.n_init} simulations) " + "-" * 30)
    X_init = safe_lhs(args.n_init * 4, N, rng)   # pool with retries
    n_done = 0

    for x in X_init:
        if n_done >= args.n_init:
            break
        p = vector_to_params(x)
        print(f"  [{n_done + 1:02d}/{args.n_init}] simulating... ", end="", flush=True)
        t0 = time.time()
        vph, vpl = safe_simulate(p)
        dt = time.time() - t0

        if vph is None:
            print(f"FAILED ({dt:.0f}s) -- retry")
            continue

        opt.register(x, vph, vpl)
        ph_ok = "V" if abs(vph - TPH) <= TOL_ else " "
        print(f"V_PH={vph:.4f}{ph_ok} V_PL={vpl:.4f}"
              f"  ePH={vph-TPH:+.4f}  ({dt:.0f}s)")
        n_done += 1

    if n_done == 0:
        print("All init simulations failed. Check ngspice + PDK installation.")
        return

    # -----------------------------------------------------------------------
    # BO loop
    # -----------------------------------------------------------------------
    print(f"\n-- Bayesian Optimization ({args.n_iter} iterations) " + "-" * 25)
    print(f"  {'It':>3}  {'V_PH':>8}  {'V_PL':>8}"
          f"  {'ePH':>8}  {'ePL_des':>9}  {'time':>6}  phase")
    print("  " + "-" * 62)

    for it in range(1, args.n_iter + 1):
        xi_now = annealed_xi(it)
        x_next = opt.suggest_next()

        p = vector_to_params(x_next)
        print(f"  {it:>3}  simulating... ", end="", flush=True)
        t0 = time.time()
        vph, vpl = safe_simulate(p)
        dt = time.time() - t0

        if vph is None:
            print(f"FAILED ({dt:.0f}s) -- skipping")
            continue

        # Log EI at accepted point
        if opt.surrogate.fitted:
            mu_ph, sig_ph, mu_pl, sig_pl = opt.surrogate.predict(
                x_next.reshape(1, -1))
            acq = smart_acquisition(
                mu_ph, sig_ph, mu_pl, sig_pl,
                opt._f_best_ph(), opt._f_best_pl(),
                xi_now, ph_solved=opt.ph_solved,
            )
            ei_val = float(acq[0])
        else:
            ei_val = float("nan")

        opt.register(x_next, vph, vpl)
        history["vph"].append(vph)
        history["vpl"].append(vpl)
        history["ei"].append(ei_val)
        history["xi"].append(xi_now)

        ph_ok = "V" if abs(vph - TPH) <= TOL_ else " "
        pl_ok = "V" if abs(vpl - TPL) <= TOL_ else " "
        phase = "PL-focus" if opt.ph_solved else "joint"
        print(f"{vph:>8.4f}{ph_ok} {vpl:>8.4f}{pl_ok}"
              f"  {vph-TPH:>+8.4f}  {vpl-TARGET_PL_DESIRED:>+9.4f}"
              f"  {dt:>5.0f}s  {phase}")

        if opt.converged:
            print(f"\n  CONVERGED at iteration {it}!")
            break

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    x_best, vph_best, vpl_best = opt.best_result()
    from optimizer.simulator import vector_to_params as v2p
    p_best = v2p(x_best)

    print("\n" + "=" * 65)
    print(f"  {'CONVERGED' if opt.converged else 'Best effort'}  "
          f"({len(opt.X)} total simulations)")
    print(f"  V_PH = {vph_best:.4f} V  (err {vph_best-TPH:+.4f} V)"
          f"  {'OK' if abs(vph_best-TPH)<=TOL_ else '--'}")
    print(f"  V_PL = {vpl_best:.4f} V  (err {vpl_best-TPL:+.4f} V)"
          f"  {'OK' if abs(vpl_best-TPL)<=TOL_ else '--'}")
    print(f"  Best V_PL reached: {max(opt.Ypl):.4f} V  (ceiling ~1.38 V)")

    print(f"\n  {'Device':<10} {'Param':<6} {'Optimal':>10} {'Nominal':>10} {'Delta':>8}")
    print("  " + "-" * 52)
    dev_map = {"W1":"XM1","L1":"XM1","W2":"XM2","L2":"XM2","W3":"XM3",
               "L3":"XM3","W4":"XM4/5","L4":"XM4/5","W6":"XM6","L6":"XM6"}
    for k in PARAM_KEYS:
        ov, nv = p_best[k], NOMINAL[k]
        span = PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]
        changed = "<-- changed" if abs(ov - nv) > 0.10 * span else ""
        print(f"  {dev_map[k]:<10} {k:<6} {ov:>10.4f} um"
              f" {nv:>10.3f} um  {ov-nv:>+7.3f}  {changed}")

    print("\n  GP-learned parameter relevance:")
    opt.print_learned_relevance(PARAM_KEYS)

    # Save JSON
    results = {
        "n_sims":    len(opt.X),
        "converged": opt.converged,
        "targets":   {"V_PH": TPH, "V_PL": TPL, "tol_mV": TOL_ * 1e3},
        "best": {
            "V_PH":       vph_best,
            "V_PL":       vpl_best,
            "err_PH_mV":  round((vph_best - TPH) * 1e3, 2),
            "err_PL_mV":  round((vpl_best - TPL) * 1e3, 2),
            "params":     p_best,
        },
        "history": {
            "V_PH": opt.Yph,
            "V_PL": opt.Ypl,
            "EI":   history["ei"],
        },
        "physics_note": (
            "V_PL ceiling ~1.38 V on 6T SKY130: VDD - |Vtp| - Vov. "
            "V_PL > 1.38 V requires topology change."
        ),
    }
    out = Path("results/bo_real_run.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out}")

    # Plots
    print("  Generating figures...")
    try:
        plot_observation_history(
            opt.Yph, opt.Ypl, history, args.n_init,
            target_ph=TPH, target_pl_realistic=TPL,
        )
        plot_convergence(history, target_ph=TPH, target_pl=TPL, tol=TOL_)
        if opt.surrogate.fitted:
            from optimizer.visualization import plot_surrogate_surface
            plot_surrogate_surface(
                opt.surrogate, x_best, PARAM_KEYS, PARAM_BOUNDS,
            )
        print("  Figures saved -> figures/")
    except Exception as e:
        print(f"  Figure error (non-fatal): {e}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
