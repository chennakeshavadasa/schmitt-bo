"""
run_baselines.py
================
Compare Bayesian Optimization against classical search strategies.
Runs offline (no ngspice needed) using synthetic physics model.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer.bayesian_opt import BayesianOptimizer, BOConfig
from optimizer.simulator import vector_to_params, NOMINAL, PARAM_KEYS
from optimizer.visualization import plot_convergence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

TARGET_PH, TARGET_PL, TOL = 1.60, 1.40, 0.015
N_BUDGET = 60   # total simulation budget for all methods

# ── Synthetic oracle (physics-inspired, no ngspice) ───────────────────────────
def oracle(x):
    """Approximate V_PH and V_PL from normalized parameter vector."""
    np.random.seed(None)  # non-deterministic noise
    vph = 1.72 - 0.18*x[7] + 0.04*x[6] - 0.10*x[1] + 0.02*np.random.randn()
    vpl = 0.85 + 0.45*x[6] + 0.08*x[9] - 0.05*x[3] + 0.02*np.random.randn()
    return float(np.clip(vph,0.5,1.75)), float(np.clip(vpl,0.3,1.70))

def error(vph, vpl):
    return abs(vph-TARGET_PH) + abs(vpl-TARGET_PL)

# ── Baseline 1: Random Search ─────────────────────────────────────────────────
def run_random(n=N_BUDGET, seed=0):
    rng = np.random.default_rng(seed)
    best_err = np.inf; hist = {'vph':[],'vpl':[]}
    for _ in range(n):
        x = rng.random(10)
        vph, vpl = oracle(x)
        hist['vph'].append(vph); hist['vpl'].append(vpl)
        best_err = min(best_err, error(vph,vpl))
    return hist

# ── Baseline 2: Latin Hypercube Sampling ─────────────────────────────────────
def run_lhs(n=N_BUDGET, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n,10))
    for j in range(10):
        perm = rng.permutation(n)
        X[:,j] = (perm + rng.random(n))/n
    hist = {'vph':[],'vpl':[]}
    for x in X:
        vph,vpl = oracle(x)
        hist['vph'].append(vph); hist['vpl'].append(vpl)
    return hist

# ── Baseline 3: Grid Search (2 key params) ───────────────────────────────────
def run_grid(n=N_BUDGET, seed=0):
    side = int(np.sqrt(n))
    g = np.linspace(0,1,side)
    hist = {'vph':[],'vpl':[]}
    for v1 in g:
        for v2 in g:
            x = np.full(10, 0.5)
            x[6]=v1; x[7]=v2   # W4, L4 — most important
            vph,vpl = oracle(x)
            hist['vph'].append(vph); hist['vpl'].append(vpl)
            if len(hist['vph'])>=n: break
        if len(hist['vph'])>=n: break
    return hist

# ── Bayesian Optimization ─────────────────────────────────────────────────────
def run_bo(n=N_BUDGET, seed=0):
    np.random.seed(seed)
    config = BOConfig(target_ph=TARGET_PH, target_pl=TARGET_PL, tolerance=TOL)
    opt = BayesianOptimizer(config)
    hist = {'vph':[],'vpl':[]}
    for i in range(n):
        x = opt.suggest_next()
        vph,vpl = oracle(x)
        opt.register(x,vph,vpl)
        hist['vph'].append(vph); hist['vpl'].append(vpl)
        if opt.state.converged: break
    return hist, opt

# ── Run all & plot ────────────────────────────────────────────────────────────
print("Running baseline comparison (offline, no ngspice)...")
print(f"  Budget: {N_BUDGET} simulations per method\n")

results = {}

print("  Random search..."); results['random'] = run_random()
print("  Latin Hypercube..."); results['lhs']    = run_lhs()
print("  Grid search...");    results['grid']   = run_grid()
print("  Bayesian Opt...");
bo_hist, bo_opt = run_bo()
results['bo'] = bo_hist

# Summary table
print(f"\n  {'Method':<20}  {'Sims':>5}  {'Best ePH':>10}  {'Best ePL':>10}  {'Converged':>10}")
print(f"  {'─'*62}")
for name, hist in results.items():
    n = len(hist['vph'])
    best_ph_err = min(abs(v-TARGET_PH) for v in hist['vph'])
    best_pl_err = min(abs(v-TARGET_PL) for v in hist['vpl'])
    conv_at = next((i+1 for i,(ph,pl) in enumerate(zip(hist['vph'],hist['vpl']))
                    if abs(ph-TARGET_PH)<=TOL and abs(pl-TARGET_PL)<=TOL), None)
    conv_str = f"sim #{conv_at}" if conv_at else "No"
    label = "Bayesian Opt ★" if name=='bo' else name.replace('_',' ').title()
    print(f"  {label:<20}  {n:>5}  {best_ph_err*1e3:>9.1f}mV  "
          f"{best_pl_err*1e3:>9.1f}mV  {conv_str:>10}")

# Save results
with open("results/baseline_comparison.json",'w') as f:
    json.dump(results, f, indent=2)

# Convergence plot
bo_main = results.pop('bo')
plot_convergence(bo_main, baselines=results)
print("\n  Comparison plot saved → figures/convergence.png")
