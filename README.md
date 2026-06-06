# Bayesian Optimization for Analog IC Sizing
### ML-Guided CMOS Schmitt Trigger Threshold Optimization on SKY130 130nm PDK

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PDK](https://img.shields.io/badge/PDK-SKY130-green.svg)](https://github.com/google/skywater-pdk)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chennakeshavadasa/schmitt-bo/blob/master/notebooks/schmitt_bo_tutorial_v2.ipynb)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**Author:** Nithin Chennakeshava Dasa — IIT Gandhinagar

---

## What This Project Does

This project applies **Bayesian Optimization (BO)** — a sample-efficient machine learning technique — to automatically find MOSFET W/L sizes for a 6-transistor CMOS Schmitt Trigger on the Google SKY130 130nm open-source PDK, targeting precise switching thresholds.

The core problem: given 10 continuous sizing parameters (W and L for each of 5 devices), find the combination that simultaneously achieves:

```
V_PH = 1.60 V  ±15 mV   (positive-going switching threshold)
V_PL = 1.38 V  ±15 mV   (negative-going switching threshold — physics ceiling)
Hysteresis = 220 mV
```

Each evaluation requires two ngspice DC sweeps (~35 seconds on a modern machine). Brute-force enumeration is impossible:

```
10 parameters × 10 values each  →  10,000,000,000 simulations  (~1,000 years)
This approach                   →  ~50 simulations              (~30 minutes)
Speedup                         →  ~10^8×
```

---

## The Circuit

A 6-transistor CMOS Schmitt Trigger implemented in the Google SKY130 130nm open PDK. The circuit produces hysteretic switching — it switches high-to-low at V_PH and low-to-high at V_PL, with the difference (V_PH − V_PL) being the hysteresis window.

```
             VDD (1.8V)
              │
         ┌────┴────┐
         │  XM5   │   PMOS feedback
         │(W4,L4) │◄──────────────────┐
         └────┬────┘                  │
              │ net2                  │
         ┌────┴────┐                  │
    VIN──┤  XM4   │   PMOS main      │
         │(W4,L4) │                   │
         └────┬────┘                  │
              │ VOUT ─────────────────┘
         ┌────┴────┐
    VIN──┤  XM2   │   NMOS main
         │(W2,L2) │
         └────┬────┘
              │ net1
         ┌────┴────┐
         │  XM3   │◄── VOUT (NMOS feedback)
         │(W3,L3) │
         └────┬────┘
              │
         ┌────┴────┐
    VIN──┤  XM1   │   NMOS series
         │(W1,L1) │
         └────┴────┘
              │
             GND
```

**Device roles and parameter sensitivities** (from real ngspice scan, 58 simulations):

| Device | Type | W/L range (µm) | Primary effect | dV_PH/dX | dV_PL/dX |
|--------|------|----------------|----------------|----------|----------|
| XM1 | NMOS series | W: 0.42–4, L: 0.15–3 | Weak V_PH tuning | +0.021 | +0.012 |
| XM2 | NMOS main | W: 0.42–8, L: 0.15–3 | Minor V_PH | −0.004 | — |
| XM3 | NMOS feedback | W: 0.42–6.5, L: 0.15–2 | V_PH shift | −0.081 | — |
| **XM4/XM5** | **PMOS main** | **W: 4–32, L: 0.15–0.8** | **Dominant V_PH** | **−0.598 V/µm (L4)** | — |
| XM6 | PMOS feedback | W: 0.42–6, L: 0.15–8 | V_PL tuning | — | **+0.078 V/µm (L6)** |

**Key physics constraint:** V_PL is bounded by `VDD − |Vtp| − Vov ≈ 1.8 − 0.60 − 0.14 ≈ 1.06 V` at nominal sizing. By pushing W4 and L6 to their limits, the maximum achievable V_PL through W/L sizing alone is **~1.38 V**. Exceeding this requires a topology change (stacked PMOS, body biasing, or a cascode stage).

---

## The Machine Learning Approach

### Overview

Bayesian Optimization is the right tool here because:
- Each evaluation (one ngspice run) is expensive (~35 s)
- The objective is black-box — no analytical gradient exists
- The parameter space is continuous and 10-dimensional
- We need a joint constraint: *both* V_PH and V_PL must hit targets

The algorithm maintains a **probabilistic surrogate model** of the circuit and uses it to decide intelligently where to simulate next — always trading off between exploring uncertain regions and exploiting known-good ones.

### 1. Latin Hypercube Sampling — Initialization

Before BO begins, 6 points are drawn using Latin Hypercube Sampling (LHS). LHS divides each of the 10 parameter dimensions into equal strata and samples one point per stratum, guaranteeing better space coverage than pure random:

```python
for j in range(n_dim):
    perm = rng.permutation(n_init)
    X[:, j] = (perm + rng.random(n_init)) / n_init
```

This ensures the GP surrogate starts with observations spread across the full design space rather than clustered in one region.

### 2. Gaussian Process Surrogate — The Probabilistic Model

A GP learns the mapping `parameter_vector → (V_PH, V_PL)` from observed simulations. Two independent GPs are trained — one per output — because V_PH and V_PL depend on different subsets of parameters with different smoothness characteristics.

**Kernel:** `ConstantKernel × Matern(ν=5/2, ARD=True) + WhiteKernel`

- **ConstantKernel** — learns the signal variance amplitude
- **Matern ν=5/2** — twice-differentiable, appropriate for the smooth physical relationships in CMOS circuits
- **ARD (Automatic Relevance Determination)** — one length-scale per dimension. After fitting, short length-scale on L4 confirms it is the dominant parameter; long length-scale on L3 means L3 is nearly irrelevant. The GP discovers this from data.
- **WhiteKernel** — absorbs SPICE numerical noise (~6 mV RMS)

At any unsampled point x, the GP returns both a **mean prediction** and a **standard deviation** — it knows where it doesn't know:

```
μ(x) = best estimate of V_PH/V_PL at x
σ(x) = uncertainty (high in unexplored regions)
```

Kernel hyperparameters (length-scales, amplitude) are optimized by maximizing the marginal log-likelihood — this is the "ML training" step, run with 10 random restarts to avoid local optima.

### 3. Expected Improvement — The Acquisition Function

EI answers: **where should the next simulation be run?**

```
EI(x) = E[max(f(x) − f_best, 0)]
       = (μ(x) − f_best − ξ) · Φ(Z) + σ(x) · φ(Z)
       
where Z = (μ(x) − f_best − ξ) / σ(x)
```

The two terms encode the exploration-exploitation tradeoff:
- `(μ − f_best) · Φ(Z)` — **exploitation**: high where the mean prediction is better than what we've seen
- `σ · φ(Z)` — **exploration**: high where uncertainty is large, even if the mean prediction is mediocre

**Multi-output acquisition — product EI:** Since both V_PH and V_PL must improve, the joint acquisition is:

```
acq(x) = EI_PH(x) × EI_PL(x)
```

This is zero unless *both* GPs predict improvement — forcing the optimizer to find points that are likely good for both outputs simultaneously.

**Critical design choice — EI target for V_PL:** The EI acquisition for V_PL always uses the *desired* target (1.40 V), not the realistic ceiling (1.38 V). If the realistic target is used and V_PL(nominal) = 1.058 V is already near 1.05 V, then `f_best_pl = −0.008` and EI_PL collapses to near zero everywhere — BO would never push V_PL higher. Using the ambitious target keeps EI_PL informative across the full range and drives V_PL to the physics ceiling.

**Annealed exploration:** The exploration parameter ξ decays over iterations:

```
ξ(t) = ξ_end + (ξ_start − ξ_end) · exp(−λt)
      = 0.001 + 0.049 · exp(−0.10 · t)
```

Early iterations explore broadly; later ones exploit the now well-fitted surrogate.

### 4. Phase-Aware Switching

Once V_PH is within ±15 mV of target, the optimizer switches to **PL-focus mode**:

```
Phase 1 (joint):    acq = EI_PH × EI_PL
Phase 2 (PL-focus): acq = EI_PL × P(V_PH within ±45 mV of target)
```

The V_PH feasibility term is a GP-based probability estimate — it softly penalises points the surrogate predicts will push V_PH out of the acceptable band, without hard-zeroing the acquisition (which would stall optimization in uncertain regions).

### 5. Acquisition Maximization

The acquisition is maximized via 20-start L-BFGS-B over the normalized [0,1]¹⁰ parameter space. Each restart begins from a random initial point; the best result across all restarts is taken as the next simulation point. 20 restarts is the minimum recommended for a 10D space — fewer risks missing the global EI maximum.

---

## Results

### Best Sizing Found (Real ngspice, SKY130)

From the v9/v10 local runs on Fedora + ngspice 41 + SKY130 BSIM4:

| Device | Parameter | Optimal | Nominal | Change |
|--------|-----------|---------|---------|--------|
| XM1 | W / L | 0.42 µm / 8.0 µm | 1.0 / 2.5 | L increased 3.2× |
| XM2 | W / L | 0.42 µm / 8.0 µm | 5.0 / 2.5 | W reduced, L increased |
| XM3 | W / L | 6.5 µm / 0.15 µm | 6.5 / 0.15 | Unchanged |
| **XM4/XM5** | **W / L** | **16 µm / 0.15 µm** | 16 / 0.15 | Unchanged |
| XM6 | W / L | 1.0 µm / **10.0 µm** | 1.0 / 0.15 | **L increased 67×** |

```
Result:  V_PH = 1.739 V    V_PL = 1.388 V
         V_PH err = +139 mV (above target — L4 can be tuned to hit exactly)
         V_PL err = −12 mV  (within 15 mV of physics ceiling)
         Hysteresis = 351 mV
```

V_PL = 1.388 V is the maximum achievable on this 6T topology through W/L sizing. The physics ceiling is `VDD − |Vtp| − Vov = 1.8 − 0.60 − 0.14 ≈ 1.06 V` at nominal, rising to ~1.38 V at extreme sizing (L6 = 10 µm, W6 = 0.42 µm, W4 = 32 µm). Reaching V_PL > 1.38 V requires a topology change.

### Baseline Comparison (60 simulations each)

All methods run on the same physics-calibrated surrogate (coefficients from real ngspice sensitivity scan). Grid search varies only L4 and W4 from a neutral starting point — no BO data leakage.

| Method | Best V_PH error | Best V_PL error | Notes |
|--------|----------------|----------------|-------|
| Random Search | 3.5 mV | 73.6 mV | Unstructured |
| Latin Hypercube | 0.1 mV | 48.8 mV | Better coverage |
| Grid Search (L4×W4) | 0.9 mV | 70.4 mV | Only 2 of 10 params |
| **Bayesian Opt (ours)** | **0.2 mV** | **16.6 mV** | **Full 10D search** |

BO achieves **3–4× lower V_PL error** than the next best method (LHS) using the same simulation budget, and explores all 10 dimensions simultaneously rather than fixing 8 at nominal like grid search.

---

## Repository Structure

```
schmitt-bo/
├── README.md
├── requirements.txt
├── cmos_schmitt_trigger.spice      ← Original SKY130 netlist (nominal sizing)
│
├── optimizer/
│   ├── __init__.py
│   ├── bayesian_opt.py             ← GP surrogate + EI acquisition (core ML)
│   ├── simulator.py                ← ngspice DC sweep interface + threshold extraction
│   └── visualization.py           ← Convergence plots, surrogate surface, observation history
│
├── experiments/
│   ├── run_bo.py                   ← Main BO script (live ngspice)
│   ├── run_bo_v2.py                ← v2: fixed EI + phase-aware acquisition
│   └── run_baselines.py            ← Compare BO vs random / LHS / grid
│
├── notebooks/
│   ├── schmitt_bo_tutorial.ipynb   ← v1 tutorial (Colab, physics surrogate)
│   └── schmitt_bo_tutorial_v2.ipynb← v2 tutorial: real ngspice + surrogate fallback
│
├── results/
│   └── baseline_comparison.json   ← Saved baseline run data
│
└── figures/
    ├── convergence.png             ← BO vs baselines convergence curve
    └── convergence.pdf
```

### Key Files Explained

**`optimizer/simulator.py`** — The ngspice interface. Generates a parametrized SPICE netlist for each W/L combination, runs two DC sweeps (0→1.8 V for V_PH, 1.8→0 V for V_PL), parses the ASCII `.raw` output, and extracts thresholds by linear interpolation at V_OUT = 0.9 V. Handles timeout (45 s hard kill) for sizing combinations that cause the BSIM4 DC solver to not converge.

**`optimizer/bayesian_opt.py`** — The BO engine. `GaussianProcessSurrogate` wraps two scikit-learn `GaussianProcessRegressor` instances. `AcquisitionFunction` computes EI and the probability of feasibility. `BayesianOptimizer` runs the main loop: suggest → simulate → register → repeat.

**`experiments/run_bo.py`** — End-to-end runner. Parses CLI arguments, runs LHS initialization, then the BO loop. Saves full results to `results/bo_real_run.json` and calls `visualization.py` to generate figures. With `--realistic` flag, targets V_PL = 1.05 V (achievable on the 6T topology).

**`notebooks/schmitt_bo_tutorial_v2.ipynb`** — The self-contained tutorial. Automatically detects whether ngspice is available on PATH; uses real simulations if so, and falls back to the calibrated physics surrogate on Colab. Each cell is a standalone educational unit with inline commentary.

---

## Quickstart

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# ngspice (for real simulations — skip for Colab/surrogate mode)
# Fedora/RHEL:
sudo dnf install ngspice
# Ubuntu/Debian:
sudo apt install ngspice
```

For real ngspice runs you also need the SKY130 PDK. The simplest install:

```bash
pip install volare
volare enable --pdk sky130 $(volare ls-remote --pdk sky130 | head -1 | awk '{print $1}')
```

Or follow the [open_pdks install guide](https://github.com/RTimothyEdwards/open_pdks).

Update the PDK path in `optimizer/simulator.py`:

```python
LIB = "/path/to/sky130A/libs.tech/combined/sky130.lib.spice"
```

### Running the Optimizer

```bash
git clone https://github.com/chennakeshavadasa/schmitt-bo
cd schmitt-bo
pip install -r requirements.txt

# Standard run — targets V_PH=1.60V, V_PL=1.40V
python experiments/run_bo.py --n_init 5 --n_iter 40 --seed 13

# Realistic run — targets V_PH=1.60V, V_PL=1.05V (what the topology can hit)
python experiments/run_bo.py --n_init 5 --n_iter 35 --seed 13 --realistic

# Baseline comparison (no ngspice needed)
python experiments/run_baselines.py
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_init` | 5 | LHS initialization simulations |
| `--n_iter` | 35 | BO iterations after initialization |
| `--seed` | 42 | Random seed (change for different LHS draws) |
| `--realistic` | off | Sets V_PL target to 1.05 V (achievable) |
| `--target_ph` | 1.60 | Custom V_PH target (V) |
| `--target_pl` | 1.40 | Custom V_PL target (V) |
| `--tolerance` | 0.015 | Convergence tolerance (V) |

**Expected runtime:** ~35 s/simulation on a modern machine (SKY130 BSIM4 model load time dominates). A full 40-iteration run takes ~25 minutes.

### Colab Tutorial

No ngspice or PDK required. The notebook uses a physics surrogate calibrated to real ngspice data:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chennakeshavadasa/schmitt-bo/blob/master/notebooks/schmitt_bo_tutorial_v2.ipynb)

The notebook walks through every component step by step:

| Cell | Content |
|------|---------|
| 1 | Setup, dependency install, repo clone |
| 2 | Constants, targets, and the EI collapse problem explained |
| 3 | 10-parameter space with real sensitivity data |
| 4 | Physics simulator: corrected coefficients, smooth V_PL ceiling |
| 5 | GP surrogate: ARD kernel, dual GPs, uncertainty visualization |
| 6 | Acquisition functions: EI, annealed ξ, phase-aware switching |
| 7 | Full `BayesianOptimizer` class |
| 8 | Run 50 BO iterations with live log |
| 9 | Best sizing table + GP-learned parameter relevances |
| 10 | Convergence: BO vs random / LHS / grid |
| 11 | 2D GP surrogate surface (L4 × W4 slice) |
| 12 | Full observation history + EI/ξ trace |
| 13 | Save results JSON + figures |
| 14 | Push to GitHub |

---

## Implementation Notes

### Why the Physics Surrogate Needed Correction

The initial Colab surrogate had V_PL sensitivity coefficients that were 10–12× too small compared to the real ngspice scan data:

| Parameter | Initial coefficient | Correct (ngspice) | Error factor |
|-----------|--------------------|--------------------|-------------|
| dV_PL/dW4 | 0.0036 V/µm | **0.045 V/µm** | 12.5× too small |
| dV_PL/dL6 | 0.008 V/µm | **0.078 V/µm** | 9.75× too small |
| dV_PL/dW6 | −0.008 V/µm | **−0.084 V/µm** | 10.5× too small |

With the wrong coefficients, the GP learned that V_PL barely responds to any parameter. EI_PL collapsed to near-zero everywhere. Product EI ≈ 0. BO effectively only optimized V_PH and never pushed V_PL higher. After correcting the coefficients, BO drives V_PL from ~1.06 V to ~1.38 V (the physics ceiling).

### The EI Collapse Problem

If a realistic, achievable target is used for V_PL EI when V_PL is already near that target, the acquisition function collapses:

```
Nominal V_PL = 1.058 V
Realistic target = 1.05 V
f_best_pl = −|1.058 − 1.05| = −0.008 V

EI_PL(any point giving ~1.08 V) ≈ 0.000002
EI_PL(any point giving ~1.15 V) ≈ 0.000000
```

The fix: always use the ambitious target (1.40 V) inside EI, regardless of the physics ceiling. This keeps EI_PL large and informative, driving BO to explore high-V_PL regions. The realistic target is only used for convergence reporting.

### Handling Simulation Failures

Some W/L combinations cause the BSIM4 DC solver to not converge (infinite current, negative capacitance, etc.). The simulator handles this with:

1. A 45-second hard timeout via `subprocess.Popen` + `proc.kill()`
2. Return of `(None, None)` on any failure
3. The BO loop skips failed points and continues — a convergence failure is treated as a non-observation

If the optimizer gets stuck, kill hung processes: `pkill -f run_bo; pkill -f ngspice`

---

## Known Limitations and Future Work

**Current limitations:**
- V_PL = 1.40 V target is topology-limited. The 6T Schmitt Trigger cannot exceed ~1.38 V through W/L tuning alone due to the PMOS threshold voltage ceiling (`VDD − |Vtp| − Vov`). Achieving V_PL > 1.38 V requires a topology redesign.
- The physics surrogate used in Colab is calibrated to the tt corner at 27°C. It does not model process corners (ss/ff) or temperature variation.
- The BO implementation uses scikit-learn GPs which scale as O(n³) in the number of observations. For budgets above ~200 simulations, a sparse GP or BoTorch would be more appropriate.

**Potential extensions:**
- Apply the same framework to OTA or LDO sizing (15–20 parameter space)
- Extend to GF180MCU PDK
- Compare against industry-standard BoTorch (Expected Hypervolume Improvement for proper Pareto-optimal multi-objective BO)
- Monte Carlo robustness analysis: optimize for yield across process corners
- Write up as a 2-page abstract for ICCAD 2026 or DATE 2027

---

## Background: Why Bayesian Optimization?

For expensive black-box functions, BO is provably more sample-efficient than grid search, random search, or gradient-free methods like Nelder-Mead. The key insight is that a GP surrogate lets you reason about the *entire* parameter space after only a handful of evaluations — you know not just what you've measured, but also what you *expect* the unmeasured regions to look like and how confident you are.

The expected improvement acquisition function has a closed-form expression under a GP model, making it fast to evaluate (microseconds) compared to running a simulation (tens of seconds). This means you can afford to maximize EI exhaustively (20 restarts × 200 L-BFGS-B iterations) to find the globally best next point.

For analog IC sizing specifically, BO has been applied to OTA design, LNA optimization, and PLL loop filter sizing. This project is a clean, open-source, end-to-end demonstration on a free PDK with all simulation infrastructure included.

---

## Citation

If you use this code or methodology in your research:

```bibtex
@software{schmitt_bo_2026,
  author    = {Chennakeshava Dasa, Nithin},
  title     = {Bayesian Optimization for Analog IC Sizing: ML-Guided CMOS Schmitt Trigger on SKY130},
  year      = {2026},
  url       = {https://github.com/chennakeshavadasa/schmitt-bo},
  note      = {IIT Gandhinagar}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The SKY130 PDK is copyright Google LLC and SkyWater Technology, licensed under Apache 2.0.
