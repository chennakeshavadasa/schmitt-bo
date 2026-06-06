"""
bayesian_opt.py
===============
Gaussian Process Bayesian Optimization engine.

Core ML components:
  1. Gaussian Process Regression (surrogate model)
     - Learns the mapping: circuit params → [V_PH, V_PL]
     - Provides mean prediction + uncertainty at any point
     - Uses RBF kernel (smooth functions) + WhiteKernel (noise)

  2. Expected Improvement (acquisition function)
     - Answers: "where should I simulate next?"
     - EI(x) = E[max(f(x) - f_best, 0)]
     - Balances exploration (uncertain regions) vs exploitation (good regions)

  3. Multi-output: separate GP for V_PH and V_PL
     - Joint constraint: both must hit target ± tolerance

This is genuine ML: the GP is a probabilistic model trained on data,
and EI is the decision rule that makes it sample-efficient.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from dataclasses import dataclass, field
from typing import Optional
import copy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BOConfig:
    """Configuration for Bayesian Optimization."""
    target_ph: float = 1.60      # V_PH target
    target_pl: float = 1.40      # V_PL target
    tolerance: float = 0.015     # ±15mV
    n_dim: int = 10              # parameter space dimension
    n_restarts: int = 10         # acquisition function optimizer restarts
    xi: float = 0.01             # EI exploration parameter
    noise_level: float = 1e-4   # GP noise (measurement uncertainty)
    kernel_type: str = 'matern'  # 'rbf' or 'matern'


@dataclass
class BOState:
    """State of the optimization run."""
    X: list = field(default_factory=list)     # observed parameter vectors
    Y_ph: list = field(default_factory=list)  # observed V_PH values
    Y_pl: list = field(default_factory=list)  # observed V_PL values
    best_idx: Optional[int] = None
    n_iter: int = 0
    converged: bool = False

    def add_observation(self, x: np.ndarray, vph: float, vpl: float):
        self.X.append(x.copy())
        self.Y_ph.append(vph)
        self.Y_pl.append(vpl)
        self.n_iter += 1

    @property
    def X_array(self): return np.array(self.X)
    @property
    def Y_ph_array(self): return np.array(self.Y_ph)
    @property
    def Y_pl_array(self): return np.array(self.Y_pl)


class GaussianProcessSurrogate:
    """
    Dual GP surrogate model: one GP per output (V_PH, V_PL).

    Why GP?
    -------
    GPs are the gold standard surrogate for expensive black-box functions:
    - Non-parametric: no assumption on functional form
    - Uncertainty-aware: gives confidence intervals, not just point estimates
    - Sample efficient: learns from very few (10-50) observations
    - Kernel encodes prior: RBF = smooth, Matern = can handle kinks

    The kernel hyperparameters (length scale, noise) are optimized
    by maximizing the marginal log-likelihood — this is ML training.
    """

    def __init__(self, config: BOConfig):
        self.config = config

        if config.kernel_type == 'matern':
            # Matern 5/2 is standard for physical systems
            # (smoother than 3/2, less smooth than RBF)
            kernel = (Matern(length_scale=np.ones(config.n_dim),
                             length_scale_bounds=(1e-2, 10.0), nu=2.5)
                      + WhiteKernel(noise_level=config.noise_level,
                                    noise_level_bounds=(1e-6, 1e-1)))
        else:
            kernel = (RBF(length_scale=np.ones(config.n_dim),
                          length_scale_bounds=(1e-2, 10.0))
                      + WhiteKernel(noise_level=config.noise_level,
                                    noise_level_bounds=(1e-6, 1e-1)))

        # Two GPs: one per output
        self.gp_ph = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            normalize_y=True, random_state=42)
        self.gp_pl = GaussianProcessRegressor(
            kernel=copy.deepcopy(kernel),
            n_restarts_optimizer=5,
            normalize_y=True, random_state=43)
        self.fitted = False

    def fit(self, X: np.ndarray, Y_ph: np.ndarray, Y_pl: np.ndarray):
        """Train both GPs on observed data."""
        self.gp_ph.fit(X, Y_ph)
        self.gp_pl.fit(X, Y_pl)
        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict V_PH and V_PL with uncertainty.

        Returns
        -------
        mu_ph, sigma_ph : mean and std of V_PH prediction
        mu_pl, sigma_pl : mean and std of V_PL prediction
        """
        if not self.fitted:
            raise RuntimeError("GP not fitted yet")
        mu_ph, sigma_ph = self.gp_ph.predict(X, return_std=True)
        mu_pl, sigma_pl = self.gp_pl.predict(X, return_std=True)
        return mu_ph, sigma_ph, mu_pl, sigma_pl

    def log_marginal_likelihood(self):
        """GP training quality metric (higher = better fit)."""
        return (self.gp_ph.log_marginal_likelihood_value_,
                self.gp_pl.log_marginal_likelihood_value_)


class AcquisitionFunction:
    """
    Expected Improvement acquisition function.

    EI answers: "where in parameter space should we run the next simulation?"

    For constraint satisfaction (hit both V_PH and V_PL targets):
    We define a composite objective:
        f(x) = -[|V_PH(x) - target_ph| + |V_PL(x) - target_pl|]
    (maximize → minimize total error)

    EI(x) = E[max(f(x) - f_best, 0)]
           = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - f_best - xi) / sigma

    The xi parameter controls exploration-exploitation tradeoff:
        xi small → exploit (search near current best)
        xi large → explore (search uncertain regions)
    """

    def __init__(self, config: BOConfig, surrogate: GaussianProcessSurrogate):
        self.config = config
        self.surrogate = surrogate

    def _composite_objective(self, mu_ph, sigma_ph, mu_pl, sigma_pl):
        """
        Combine V_PH and V_PL predictions into single objective.
        Higher = closer to both targets simultaneously.
        """
        # Negative total absolute error
        err_ph = np.abs(mu_ph - self.config.target_ph)
        err_pl = np.abs(mu_pl - self.config.target_pl)
        return -(err_ph + err_pl)

    def expected_improvement(self, X: np.ndarray, f_best: float) -> np.ndarray:
        """
        Compute EI at query points X.

        Parameters
        ----------
        X : (n_points, n_dim) array of normalized parameter vectors
        f_best : best composite objective seen so far

        Returns
        -------
        ei : (n_points,) Expected Improvement values
        """
        X = np.atleast_2d(X)
        mu_ph, sigma_ph, mu_pl, sigma_pl = self.surrogate.predict(X)
        mu = self._composite_objective(mu_ph, sigma_ph, mu_pl, sigma_pl)
        sigma = np.sqrt(sigma_ph**2 + sigma_pl**2 + 1e-9)

        Z = (mu - f_best - self.config.xi) / sigma
        ei = (mu - f_best - self.config.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-10] = 0.0
        return ei

    def probability_of_feasibility(self, X: np.ndarray) -> np.ndarray:
        """
        P(V_PH ≈ target AND V_PL ≈ target) — joint constraint probability.
        Uses GP uncertainty to estimate how likely a point satisfies both specs.
        """
        X = np.atleast_2d(X)
        mu_ph, sigma_ph, mu_pl, sigma_pl = self.surrogate.predict(X)
        tol = self.config.tolerance

        # P(|V_PH - target| < tol) = P(target-tol < V_PH < target+tol)
        pf_ph = (norm.cdf((self.config.target_ph + tol - mu_ph) / (sigma_ph+1e-9)) -
                 norm.cdf((self.config.target_ph - tol - mu_ph) / (sigma_ph+1e-9)))
        pf_pl = (norm.cdf((self.config.target_pl + tol - mu_pl) / (sigma_pl+1e-9)) -
                 norm.cdf((self.config.target_pl - tol - mu_pl) / (sigma_pl+1e-9)))

        # Joint probability (assuming independence — approximation)
        return pf_ph * pf_pl

    def constrained_ei(self, X: np.ndarray, f_best: float) -> np.ndarray:
        """
        Constrained EI = EI × P(feasible)
        This is the main acquisition: find points that are both
        likely to be good AND likely to satisfy the constraint.
        """
        ei = self.expected_improvement(X, f_best)
        pf = self.probability_of_feasibility(X)
        return ei * pf


class BayesianOptimizer:
    """
    Main Bayesian Optimization loop.

    Algorithm:
        1. Evaluate n_init random points (warm-up)
        2. Fit GP surrogate to observations
        3. Maximize acquisition function to find next query point
        4. Evaluate simulator at that point
        5. Update surrogate and repeat

    This is the canonical BO loop from:
        Mockus (1978), Srinivas et al. (2010), Shahriari et al. (2016)
    """

    def __init__(self, config: BOConfig):
        self.config = config
        self.surrogate = GaussianProcessSurrogate(config)
        self.acquisition = AcquisitionFunction(config, self.surrogate)
        self.state = BOState()

    def _f_best(self) -> float:
        """Current best composite objective."""
        if not self.state.Y_ph:
            return -np.inf
        errs = [abs(ph - self.config.target_ph) + abs(pl - self.config.target_pl)
                for ph, pl in zip(self.state.Y_ph, self.state.Y_pl)]
        return -min(errs)

    def _optimize_acquisition(self) -> np.ndarray:
        """
        Find x* = argmax EI(x) by multi-start L-BFGS-B.

        This is an inner optimization problem (cheap — no simulator).
        We use gradient-based optimization on the differentiable EI function.
        Multiple random restarts avoid local optima.
        """
        f_best = self._f_best()
        best_x, best_acq = None, -np.inf

        def neg_acq(x):
            return -self.acquisition.constrained_ei(x.reshape(1,-1), f_best)[0]

        # Multi-start: random initial points
        n_restarts = self.config.n_restarts
        x0_candidates = np.random.rand(n_restarts, self.config.n_dim)

        for x0 in x0_candidates:
            try:
                result = minimize(neg_acq, x0,
                                  bounds=[(0,1)]*self.config.n_dim,
                                  method='L-BFGS-B')
                if -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x
            except Exception:
                pass

        if best_x is None:
            best_x = np.random.rand(self.config.n_dim)

        return np.clip(best_x, 0, 1)

    def suggest_next(self) -> np.ndarray:
        """
        Suggest the next point to evaluate.
        Returns normalized parameter vector in [0,1]^10.
        """
        if len(self.state.X) < 3:
            # Not enough data for GP — random explore
            return np.random.rand(self.config.n_dim)

        # Fit surrogate to current data
        X = self.state.X_array
        self.surrogate.fit(X, self.state.Y_ph_array, self.state.Y_pl_array)

        # Maximize acquisition
        return self._optimize_acquisition()

    def register(self, x: np.ndarray, vph: float, vpl: float):
        """Register a new observation."""
        self.state.add_observation(x, vph, vpl)

        # Check convergence
        if (abs(vph - self.config.target_ph) <= self.config.tolerance and
                abs(vpl - self.config.target_pl) <= self.config.tolerance):
            self.state.converged = True

    def best_result(self) -> tuple:
        """Return (x_best, vph_best, vpl_best) — closest to target."""
        if not self.state.Y_ph:
            return None, None, None
        errs = [abs(ph - self.config.target_ph) + abs(pl - self.config.target_pl)
                for ph, pl in zip(self.state.Y_ph, self.state.Y_pl)]
        idx = np.argmin(errs)
        return (self.state.X[idx],
                self.state.Y_ph[idx],
                self.state.Y_pl[idx])

    def surrogate_predict_grid(self, param_key1: str, param_key2: str,
                                n_grid: int = 30):
        """
        Predict surrogate surface for two parameters (others at best point).
        Used for visualization of the learned model.
        """
        from .simulator import PARAM_KEYS
        x_best, _, _ = self.best_result()
        if x_best is None:
            return None

        i1 = PARAM_KEYS.index(param_key1)
        i2 = PARAM_KEYS.index(param_key2)

        g = np.linspace(0, 1, n_grid)
        G1, G2 = np.meshgrid(g, g)
        X_grid = np.tile(x_best, (n_grid*n_grid, 1))
        X_grid[:, i1] = G1.ravel()
        X_grid[:, i2] = G2.ravel()

        mu_ph, sig_ph, mu_pl, sig_pl = self.surrogate.predict(X_grid)
        return G1, G2, mu_ph.reshape(n_grid,n_grid), mu_pl.reshape(n_grid,n_grid)
