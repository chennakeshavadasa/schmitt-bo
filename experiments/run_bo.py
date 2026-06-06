"""
run_bo.py — Main Bayesian Optimization script.
Usage:
    python experiments/run_bo.py              # live ngspice
    python experiments/run_bo.py --offline    # no ngspice needed
    python experiments/run_bo.py --target_ph 1.60 --target_pl 1.40 --n_iter 50
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizer.bayesian_opt import BayesianOptimizer, BOConfig
from optimizer.simulator import (simulate, params_to_vector,
                                  vector_to_params, NOMINAL, PARAM_KEYS)
from optimizer.visualization import (plot_convergence, plot_surrogate_surface,
                                      plot_observation_history)

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target_ph', type=float, default=1.60)
    p.add_argument('--target_pl', type=float, default=1.40)
    p.add_argument('--tolerance', type=float, default=0.015)
    p.add_argument('--n_init',    type=int,   default=5)
    p.add_argument('--n_iter',    type=int,   default=45)
    p.add_argument('--offline',   action='store_true')
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--kernel',    default='matern', choices=['matern','rbf'])
    return p.parse_args()

def make_offline_data(n=80):
    """Physics-inspired synthetic data — runs without ngspice."""
    np.random.seed(42)
    data = []
    for _ in range(n):
        x = np.random.rand(10)
        # Approximate physics: L4(x[7]) dominates V_PH, W4(x[6]) lifts V_PL
        vph = 1.72 - 0.18*x[7] + 0.04*x[6] - 0.10*x[1] + 0.03*np.random.randn()
        vpl = 0.85 + 0.45*x[6] + 0.08*x[9] - 0.05*x[3] + 0.03*np.random.randn()
        vph = float(np.clip(vph, 0.5, 1.75))
        vpl = float(np.clip(vpl, 0.3, min(vph-0.05, 1.75)))
        data.append({'x': x.tolist(), 'vph': vph, 'vpl': vpl})
    return data

def lhs_sample(n, d, rng):
    """Latin Hypercube Sampling: better space coverage than random."""
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    print("\n" + "="*62)
    print("  Bayesian Optimization for Analog Circuit Sizing")
    print(f"  Target: V_PH={args.target_ph:.3f}V  V_PL={args.target_pl:.3f}V  ±{args.tolerance*1e3:.0f}mV")
    print(f"  Kernel: {args.kernel}  init={args.n_init}  iter={args.n_iter}")
    print(f"  Mode: {'OFFLINE (synthetic data)' if args.offline else 'LIVE ngspice'}")
    print("="*62)

    config = BayesianOptimizer.__init__.__doc__  # just for ref
    config = BOConfig(target_ph=args.target_ph, target_pl=args.target_pl,
                      tolerance=args.tolerance, kernel_type=args.kernel)
    opt = BayesianOptimizer(config)
    offline_data = make_offline_data() if args.offline else None

    # ── Phase 1: LHS initialization ──────────────────────────────────────
    print(f"\n── LHS Initialization ({args.n_init} sims) ─────────────────────────")
    X_init = lhs_sample(args.n_init, config.n_dim, rng)
    for i, x in enumerate(X_init):
        p = vector_to_params(x)
        if args.offline:
            d = offline_data[i % len(offline_data)]
            vph, vpl = d['vph'], d['vpl']
        else:
            print(f"  [{i+1}/{args.n_init}] sim... ", end='', flush=True)
            vph, vpl = simulate(p)
        if vph is None: print("FAIL"); continue
        opt.register(x, vph, vpl)
        print(f"  [{i+1}/{args.n_init}] V_PH={vph:.4f}  V_PL={vpl:.4f}  "
              f"ePH={vph-args.target_ph:+.4f}  ePL={vpl-args.target_pl:+.4f}")

    # ── Phase 2: BO loop ─────────────────────────────────────────────────
    print(f"\n── Bayesian Optimization Loop ──────────────────────────────────")
    print(f"  {'It':>3}  {'V_PH':>8}  {'V_PL':>8}  {'ePH':>8}  {'ePL':>8}  {'EI':>10}")
    print(f"  {'─'*55}")

    history = {'vph':[], 'vpl':[]}
    offline_idx = args.n_init

    for it in range(1, args.n_iter+1):
        x_next = opt.suggest_next()

        if opt.surrogate.fitted:
            ei = opt.acquisition.constrained_ei(x_next.reshape(1,-1),
                                                  opt._f_best())[0]
        else:
            ei = float('nan')

        p_next = vector_to_params(x_next)

        if args.offline:
            d = offline_data[offline_idx % len(offline_data)]
            vph, vpl = d['vph'], d['vpl']
            offline_idx += 1
        else:
            vph, vpl = simulate(p_next)

        if vph is None:
            print(f"  {it:>3}  FAIL"); continue

        opt.register(x_next, vph, vpl)
        history['vph'].append(vph); history['vpl'].append(vpl)

        flag = " ✓" if opt.state.converged else ""
        print(f"  {it:>3}  {vph:>8.4f}  {vpl:>8.4f}  "
              f"{vph-args.target_ph:>+8.4f}  {vpl-args.target_pl:>+8.4f}  "
              f"{ei:>10.6f}{flag}")

        if opt.state.converged:
            print(f"\n  ✓ Converged at iteration {it}!"); break

    # ── Results ───────────────────────────────────────────────────────────
    x_best, vph_b, vpl_b = opt.best_result()
    p_best = vector_to_params(x_best)
    ok = (abs(vph_b-args.target_ph)<=args.tolerance and
          abs(vpl_b-args.target_pl)<=args.tolerance)

    print("\n" + "="*62)
    print(f"  {'✓ CONVERGED' if ok else '✗ Best effort'}  "
          f"({len(opt.state.X)} simulations total)")
    print(f"  V_PH = {vph_b:.4f} V  (err {vph_b-args.target_ph:+.4f} V)")
    print(f"  V_PL = {vpl_b:.4f} V  (err {vpl_b-args.target_pl:+.4f} V)")
    print(f"\n  Best sizing:")
    for k in PARAM_KEYS:
        chg = '*' if abs(p_best[k]-NOMINAL.get(k,0))>0.01 else ' '
        print(f"    {chg}{k}: {p_best[k]:.4f}µm")

    with open("results/bo_run.json",'w') as f:
        json.dump({'n_sims':len(opt.state.X),'converged':ok,
                   'best':{'vph':vph_b,'vpl':vpl_b,'params':p_best},
                   'history':history}, f, indent=2)

    print("\n  Generating plots...")
    plot_observation_history(opt.state, args.target_ph, args.target_pl, args.tolerance)
    plot_convergence(history, target_ph=args.target_ph,
                     target_pl=args.target_pl, tol=args.tolerance)
    if opt.surrogate.fitted:
        plot_surrogate_surface(opt)
    print("  Plots saved to figures/")
    print("="*62+"\n")

if __name__ == "__main__":
    main()
