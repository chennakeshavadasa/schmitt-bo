"""
bayesian_opt.py
===============
Gaussian Process Bayesian Optimization engine for Schmitt Trigger sizing.

Algorithm
---------
1. Latin Hypercube Sampling (LHS) -- space-filling initialization
2. Dual GP surrogate (one per output: V_PH, V_PL)
   Kernel: ConstantKernel * Matern(nu=2.5, ARD) + WhiteKernel
3. Phase-aware acquisition function:
   Phase 1 (joint):    acq = EI_PH * EI_PL
   Phase 2 (PL-focus): acq = EI_PL * P(V_PH in band)
   EI_PL always uses the ambitious 1.40 V target to prevent EI collapse.
4. Annealed exploration: xi decays 0.05 -> 0.001 over iterations
5. Acquisition maximized via 20-start L-BFGS-B in [0,1]^10

Key design decisions documented inline.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_PH          = 1.60   # V -- upper switching threshold
TARGET_PL_DESIRED  = 1.40   # V -- used inside EI (ambitious; keeps EI informative)
TARGET_PL_CEILING  = 1.38   # V -- physics ceiling: VDD - |Vtp| - Vov
TARGET_PL_REALISTIC = 1.05  # V -- achievable on 6T topology at nominal sizing
TOL                = 0.015  # V -- +/- 15 mV convergence window
VDD                = 1.80   # V
VPL_MAX            = 1.38   # V -- hard ceiling from ngspice scan


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------

class GPSurrogate:
    """
    Dual Gaussian Process surrogate: one GP per output (V_PH, V_PL).

    Kernel: ConstantKernel(amplitude) * Matern(nu=2.5, ARD) + WhiteKernel

    ARD (Automatic Relevance Determination): one length-scale per parameter.
    After fitting, short length-scale on L4 confirms it as the dominant
    parameter; long length-scale on L3 means L3 is nearly irrelevant.
    The GP discovers this automatically from simulation data.

    Two separate GPs (not one joint GP) because V_PH and V_PL have different
    dependencies and smoothness -- a joint GP would conflate them.
    """

    def __init__(self, n_dim=10, n_restarts=10):
        self.n_dim = n_dim
        self.fitted = False
        self._build(n_restarts)

    def _make_kernel(self):
        return (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0))
            * Matern(
                length_scale=np.ones(self.n_dim),
                length_scale_bounds=(5e-3, 10.0),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=1e-4,
                noise_level_bounds=(1e-6, 1e-2),
            )
        )

    def _build(self, n_restarts):
        kw = dict(n_restarts_optimizer=n_restarts, normalize_y=True)
        self.gp_ph = GaussianProcessRegressor(
            kernel=self._make_kernel(), random_state=42, **kw
        )
        self.gp_pl = GaussianProcessRegressor(
            kernel=self._make_kernel(), random_state=43, **kw
        )

    def fit(self, X, Y_ph, Y_pl):
        """Train both GPs. X must be normalized to [0,1]^n_dim."""
        self.gp_ph.fit(X, Y_ph)
        self.gp_pl.fit(X, Y_pl)
        self.fitted = True

    def predict(self, X):
        """Return (mu_ph, sig_ph, mu_pl, sig_pl) at query points X."""
        X = np.atleast_2d(X)
        mu_ph, sig_ph = self.gp_ph.predict(X, return_std=True)
        mu_pl, sig_pl = self.gp_pl.predict(X, return_std=True)
        return (
            mu_ph,
            np.maximum(sig_ph, 1e-9),
            mu_pl,
            np.maximum(sig_pl, 1e-9),
        )

    def learned_length_scales(self, param_keys):
        """
        Return dict of {param: (ls_ph, ls_pl)} after fitting.
        Shorter length-scale = stronger influence on that output.
        """
        if not self.fitted:
            return None
        ls_ph = self.gp_ph.kernel_.k1.k2.length_scale
        ls_pl = self.gp_pl.kernel_.k1.k2.length_scale
        return {k: (ls_ph[i], ls_pl[i]) for i, k in enumerate(param_keys)}


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Standard Expected Improvement acquisition.

    EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
    Z      = (mu - f_best - xi) / sigma

    We maximize an objective where higher = better (obj = -|error|).
    f_best is the best (least-negative) objective seen so far.
    """
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return np.maximum(ei, 0.0)


def annealed_xi(iteration, xi_start=0.05, xi_end=0.001, decay=0.10):
    """
    Anneal exploration parameter over iterations.
    Early: xi_start=0.05  (broad exploration)
    Late:  xi_end=0.001   (exploit known good regions)
    """
    return xi_end + (xi_start - xi_end) * np.exp(-decay * iteration)


def smart_acquisition(mu_ph, sig_ph, mu_pl, sig_pl,
                      best_ph_obj, best_pl_obj, xi,
                      ph_solved=False):
    """
    Phase-aware acquisition function.

    CRITICAL: EI_PL always uses TARGET_PL_DESIRED (1.40 V).
    Reason: if TARGET_PL = 1.05 V and V_PL(nominal) = 1.058 V, then
    f_best_pl = -0.008 (already 8 mV from target). EI_PL collapses to
    near-zero everywhere and BO never pushes V_PL higher. Using the
    ambitious 1.40 V target keeps EI_PL informative across the full range.

    Phase 1 (V_PH not yet solved):
        acq = EI_PH * EI_PL  -- must improve both simultaneously

    Phase 2 (V_PH within tolerance):
        acq = EI_PL * P(V_PH within 3*TOL of target)
        A fixed weight (e.g. 0.1 * EI_PH) was tried but let V_PH escape
        to 1.76 V repeatedly. The probabilistic constraint naturally
        penalises points the GP predicts will push V_PH out of range,
        while a 5% floor ensures EI never fully collapses.
    """
    # V_PH acquisition: target = 1.60 V
    obj_ph = -(np.abs(mu_ph - TARGET_PH))
    ei_ph = expected_improvement(obj_ph, sig_ph, best_ph_obj, xi)

    # V_PL acquisition: ALWAYS use desired target (1.40 V) -- never realistic
    obj_pl = -(np.abs(mu_pl - TARGET_PL_DESIRED))
    ei_pl = expected_improvement(obj_pl, sig_pl, best_pl_obj, xi)

    if ph_solved:
        # V_PH is solved: maximize V_PL while keeping V_PH in a soft band
        band = 3.0 * TOL  # 45 mV -- relaxed so we don't strangle exploration
        pf_ph = (
            norm.cdf((TARGET_PH + band - mu_ph) / (sig_ph + 1e-9))
            - norm.cdf((TARGET_PH - band - mu_ph) / (sig_ph + 1e-9))
        )
        return (ei_pl * np.clip(pf_ph, 0.05, 1.0)).ravel()  # floor at 5%
    else:
        return (ei_ph * ei_pl).ravel()


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """
    Bayesian Optimizer for dual-output Schmitt Trigger sizing.

    Usage
    -----
    opt = BayesianOptimizer()
    for x in lhs_init:
        vph, vpl = simulate(vector_to_params(x))
        opt.register(x, vph, vpl)
    for it in range(n_iter):
        x_next = opt.suggest_next()
        vph, vpl = simulate(vector_to_params(x_next))
        opt.register(x_next, vph, vpl)
    x_best, vph_best, vpl_best = opt.best_result()
    """

    def __init__(self, n_dim=10, tol=TOL,
                 n_restarts_acq=20, n_restarts_gp=10):
        self.n_dim = n_dim
        self.tol = tol
        self.n_restarts_acq = n_restarts_acq
        self.surrogate = GPSurrogate(n_dim, n_restarts=n_restarts_gp)
        self.X   = []   # normalized [0,1]^n_dim observations
        self.Yph = []   # V_PH observations
        self.Ypl = []   # V_PL observations
        self.iteration = 0

    # ------------------------------------------------------------------
    # Data registration

    def register(self, x, vph, vpl):
        """Record one (x, V_PH, V_PL) observation."""
        self.X.append(np.asarray(x, dtype=float).copy())
        self.Yph.append(float(vph))
        self.Ypl.append(float(vpl))

    # ------------------------------------------------------------------
    # Internal best-objective tracking

    def _f_best_ph(self):
        return (
            -min(abs(v - TARGET_PH) for v in self.Yph)
            if self.Yph else -np.inf
        )

    def _f_best_pl(self):
        # Always compare against DESIRED target to keep EI informative
        return (
            -min(abs(v - TARGET_PL_DESIRED) for v in self.Ypl)
            if self.Ypl else -np.inf
        )

    # ------------------------------------------------------------------
    # Convergence properties

    @property
    def ph_solved(self):
        return any(abs(v - TARGET_PH) <= self.tol for v in self.Yph)

    @property
    def converged_ph(self):
        return self.ph_solved

    @property
    def converged_pl(self):
        """Convergence check uses the realistic target for reporting."""
        return any(abs(v - TARGET_PL_REALISTIC) <= self.tol for v in self.Ypl)

    @property
    def converged(self):
        return self.converged_ph and self.converged_pl

    # ------------------------------------------------------------------
    # Next-point suggestion

    def suggest_next(self):
        """
        Fit GPs and return the next normalized [0,1]^n_dim vector to simulate.
        Acquisition maximized via 20-start L-BFGS-B.
        """
        self.iteration += 1

        if len(self.X) < 3:
            return np.random.rand(self.n_dim)

        X_arr = np.array(self.X)
        self.surrogate.fit(X_arr, np.array(self.Yph), np.array(self.Ypl))

        xi      = annealed_xi(self.iteration)
        best_ph = self._f_best_ph()
        best_pl = self._f_best_pl()
        ph_done = self.ph_solved

        def neg_acq(x):
            mu_ph, sig_ph, mu_pl, sig_pl = self.surrogate.predict(x.reshape(1, -1))
            acq = smart_acquisition(
                mu_ph, sig_ph, mu_pl, sig_pl,
                best_ph, best_pl, xi,
                ph_solved=ph_done,
            )
            return -float(acq[0])

        best_x, best_val = None, np.inf
        bounds = [(0.0, 1.0)] * self.n_dim

        for _ in range(self.n_restarts_acq):
            x0 = np.random.rand(self.n_dim)
            try:
                res = minimize(
                    neg_acq, x0, bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-9},
                )
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x
            except Exception:
                pass

        if best_x is None:
            best_x = np.random.rand(self.n_dim)

        return np.clip(best_x, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Results

    def best_result(self):
        """
        Return (x, V_PH, V_PL) for the observation closest to both targets.
        Ranked by combined error against desired targets.
        """
        if not self.Yph:
            return None, None, None
        errs = [
            abs(ph - TARGET_PH) + abs(pl - TARGET_PL_DESIRED)
            for ph, pl in zip(self.Yph, self.Ypl)
        ]
        idx = int(np.argmin(errs))
        return self.X[idx], self.Yph[idx], self.Ypl[idx]

    def print_learned_relevance(self, param_keys):
        """
        Print GP-learned ARD length-scales after fitting.
        Shorter = that parameter has stronger influence.
        """
        scales = self.surrogate.learned_length_scales(param_keys)
        if not scales:
            print("  (GP not fitted yet)")
            return
        print(f"  {'Param':<6}  {'LS(V_PH)':>10}  {'LS(V_PL)':>10}"
              f"  (shorter = stronger influence)")
        print("  " + "-" * 50)
        for k, (lp, ll) in sorted(scales.items(), key=lambda kv: kv[1][0]):
            bar = "|" * max(1, int(5 / lp))
            print(f"  {k:<6}  {lp:>10.4f}  {ll:>10.4f}  {bar}")
