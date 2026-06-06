"""
run_baselines.py
================
Compare Bayesian Optimization against three classical search strategies,
using the calibrated physics surrogate (no ngspice needed).

Baselines
---------
- Random search      : uniform random sampling
- Latin Hypercube    : stratified space-filling sampling
- Grid search        : 2D grid over L4 x W4 (the two most important params),
                       all others fixed at nominal

All baselines are run independently with no access to BO results (no data
leakage). Grid search starts from NOMINAL, not from any BO-derived point.

Usage
-----
    python experiments/run_baselines.py

Outputs: results/baseline_comparison.json, figures/convergence.png
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer.simulator import PARAM_KEYS, PARAM_BOUNDS, NOMINAL, vector_to_params
from optimizer.bayesian_opt import (
    BayesianOptimizer, annealed_xi, smart_acquisition,
    TARGET_PH, TARGET_PL_DESIRED, TARGET_PL_REALISTIC, TOL,
)
from optimizer.visualization import plot_convergence

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

N_BUDGET = 60   # simulation budget per method
N_DIM    = 10
VPL_MAX  = 1.38


# ---------------------------------------------------------------------------
# Calibrated physics surrogate
# ---------------------------------------------------------------------------
# Coefficients from real ngspice sensitivity scan (v8, 58 sims, tt corner).
# These corrected values produce realistic V_PL responses:
#   dV_PL/dW4 = +0.045 V/um  (not 0.0036 -- old error was 12x too small)
#   dV_PL/dL6 = +0.078 V/um  (not 0.008  -- old error was 10x too small)
#   dV_PL/dW6 = -0.084 V/um  (not -0.008 -- old error was 10x too small)

def _soft_ceiling(x, ceiling=VPL_MAX, transition=0.12):
    """Smooth logistic ceiling -- avoids GP-breaking gradient discontinuity."""
    midpoint = ceiling - transition / 2.0
    k = 6.0 / transition
    above = 1.0 / (1.0 + np.exp(-k * (np.asarray(x) - midpoint)))
    return x * (1.0 - above) + ceiling * above


def physics_simulator(p, noise_std=0.006, seed=None):
    """
    Physics-calibrated Schmitt Trigger model (no ngspice required).
    Matches real ngspice at nominal to <1 mV; sensitivity coefficients
    match to ~98% across the parameter space.
    """
    W1, L1 = p["W1"], p["L1"]
    W2, L2 = p["W2"], p["L2"]
    W3, L3 = p["W3"], p["L3"]
    W4, L4 = p["W4"], p["L4"]
    W6, L6 = p["W6"], p["L6"]

    vph = (
        1.615
        - 0.598 * (L4 - 0.15)           # dominant: -0.598 V/um
        + 0.0016 * (W4 - 16.0)
        - 0.081 * (W3 - 6.5)
        + 0.015 * (L1 - 2.5)
        + 0.021 * (1.0 / max(W1, 0.1) - 1.0)
        - 0.004 * (W2 - 5.0)
    )

    vpl_raw = (
        1.058
        + 0.045 * (W4 - 16.0)           # CORRECTED: was 0.0036
        + 0.078 * (L6 - 0.15)           # CORRECTED: was 0.008
        + 0.006 * (L1 - 2.5)
        + 0.006 * (L2 - 2.5)
        - 0.084 * (W6 - 1.0)            # CORRECTED: was -0.008
    )
    vpl = _soft_ceiling(vpl_raw)

    if noise_std > 0:
        _seed = seed if seed is not None else (
            int(abs(hash(tuple(round(p[k], 4) for k in PARAM_KEYS)))) % (2**31)
        )
        rng = np.random.default_rng(_seed)
        vph += noise_std * rng.standard_normal()
        vpl += noise_std * rng.standard_normal()

    return float(np.clip(vph, 0.3, 1.76)), float(np.clip(vpl, 0.3, vph - 0.05))


def simulate(x_norm):
    """Simulate from a normalized [0,1]^10 vector."""
    return physics_simulator(vector_to_params(x_norm))


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def run_random_search(n=N_BUDGET, seed=1):
    np.random.seed(seed)
    hist = {"vph": [], "vpl": []}
    for _ in range(n):
        x = np.random.rand(N_DIM)
        vph, vpl = simulate(x)
        hist["vph"].append(vph)
        hist["vpl"].append(vpl)
    return hist


def run_lhs(n=N_BUDGET, seed=1):
    np.random.seed(seed)
    X = np.zeros((n, N_DIM))
    for j in range(N_DIM):
        perm = np.random.permutation(n)
        X[:, j] = (perm + np.random.rand(n)) / n
    hist = {"vph": [], "vpl": []}
    for x in X:
        vph, vpl = simulate(x)
        hist["vph"].append(vph)
        hist["vpl"].append(vpl)
    return hist


def run_grid_search(n=N_BUDGET):
    """
    Grid over L4 x W4 from NOMINAL baseline.
    No BO data used -- fresh independent start.
    """
    side = int(np.sqrt(n))
    hist = {"vph": [], "vpl": []}
    p_base = dict(NOMINAL)
    for l4 in np.linspace(*PARAM_BOUNDS["L4"], side):
        for w4 in np.linspace(*PARAM_BOUNDS["W4"], side):
            p_base["L4"], p_base["W4"] = l4, w4
            vph, vpl = physics_simulator(p_base)
            hist["vph"].append(vph)
            hist["vpl"].append(vpl)
            if len(hist["vph"]) >= n:
                break
        if len(hist["vph"]) >= n:
            break
    return hist


def run_bo(n=N_BUDGET, seed=42):
    np.random.seed(seed)
    opt = BayesianOptimizer(n_dim=N_DIM, tol=TOL,
                             n_restarts_acq=20, n_restarts_gp=10)
    hist = {"vph": [], "vpl": []}

    # LHS init
    lhs = np.zeros((6, N_DIM))
    for j in range(N_DIM):
        perm = np.random.permutation(6)
        lhs[:, j] = (perm + np.random.rand(6)) / 6
    for x in lhs:
        vph, vpl = simulate(x)
        opt.register(x, vph, vpl)
        hist["vph"].append(vph)
        hist["vpl"].append(vpl)

    for it in range(1, n - 6 + 1):
        x_next = opt.suggest_next()
        vph, vpl = simulate(x_next)
        opt.register(x_next, vph, vpl)
        hist["vph"].append(vph)
        hist["vpl"].append(vpl)
        if len(hist["vph"]) >= n:
            break

    return hist, opt


# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------

print("Running baseline comparison (offline -- no ngspice needed)")
print(f"  Budget: {N_BUDGET} simulations per method\n")

rand_hist  = run_random_search()
print("  Random search       -- done")
lhs_hist   = run_lhs()
print("  Latin Hypercube     -- done")
grid_hist  = run_grid_search()
print("  Grid search (L4xW4) -- done")
bo_hist, bo_opt = run_bo()
print("  Bayesian Opt        -- done\n")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

all_results = {
    "Random Search":   rand_hist,
    "Latin Hypercube": lhs_hist,
    "Grid (L4xW4)":    grid_hist,
    "Bayesian Opt":    bo_hist,
}

print(f"  {'Method':<20}  {'N sims':>7}  {'Best ePH (mV)':>14}"
      f"  {'Best ePL (mV)':>14}  {'Max V_PL':>10}")
print("  " + "-" * 75)

for name, hist in all_results.items():
    n = len(hist["vph"])
    best_ph = min(abs(v - TARGET_PH) * 1e3 for v in hist["vph"])
    best_pl = min(abs(v - TARGET_PL_DESIRED) * 1e3 for v in hist["vpl"])
    max_vpl = max(hist["vpl"])
    star = "*" if "Bayesian" in name else " "
    print(f"  {star}{name:<19}  {n:>7}  {best_ph:>14.1f}"
          f"  {best_pl:>14.1f}  {max_vpl:>10.4f} V")

# ---------------------------------------------------------------------------
# Save and plot
# ---------------------------------------------------------------------------

results_json = {
    "random": rand_hist,
    "lhs":    lhs_hist,
    "grid":   grid_hist,
    "bo":     bo_hist,
}
with open("results/baseline_comparison.json", "w") as f:
    json.dump(results_json, f, indent=2)
print("\n  Results saved -> results/baseline_comparison.json")

bo_main = {"vph": bo_hist["vph"], "vpl": bo_hist["vpl"]}
baselines = {
    "random": rand_hist,
    "lhs":    lhs_hist,
    "grid":   grid_hist,
}
plot_convergence(bo_main, baselines=baselines,
                 target_ph=TARGET_PH, target_pl=TARGET_PL_DESIRED)
print("  Convergence plot saved -> figures/convergence.png")
