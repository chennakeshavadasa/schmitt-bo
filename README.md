# Bayesian Optimization for Analog Circuit Sizing
### ML-guided CMOS Schmitt Trigger threshold optimization on SKY130 PDK

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PDK](https://img.shields.io/badge/PDK-SKY130-green.svg)](https://github.com/google/skywater-pdk)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

A publishable demonstration of **Gaussian Process Bayesian Optimization** applied to analog IC sizing — finding MOSFET W/L parameters that meet precise switching threshold specifications with minimal simulator calls.

## Why This is Interesting

Traditional analog sizing is manual (designer intuition) or uses brute-force sweeps (exponential cost). This project applies **surrogate model-based optimization** — a core ML technique — to replace expensive ngspice simulations with a cheap probabilistic model that guides where to sample next.

```
Brute force (10 params × 10 values each): 10^10 simulations
This approach: ~30-60 simulations to convergence
Speedup: ~10^8×
```

## The Circuit

6-transistor CMOS Schmitt Trigger on Google SKY130 130nm PDK.

```
Target:  V_PH = 1.60V  (upper switching threshold)
         V_PL = 1.40V  (lower switching threshold)
         Hysteresis = 200mV, centered at 1.5V
```

## ML Approach

### 1. Gaussian Process Surrogate Model
- Maps 10-dimensional parameter space (W/L of each MOSFET) → [V_PH, V_PL]
- RBF + WhiteKernel for noise-robust regression
- Provides uncertainty estimates — knows where it doesn't know

### 2. Expected Improvement Acquisition Function
- Balances **exploration** (high uncertainty regions) vs **exploitation** (promising regions)
- Guides each new simulation to maximally reduce uncertainty about the optimum

### 3. Multi-Output GP
- Separate GP for V_PH and V_PL
- Joint constraint satisfaction: both outputs must hit targets simultaneously

### 4. Comparison Baselines
- Random search
- Latin Hypercube Sampling
- Grid search
- Bayesian Optimization (ours)

## Project Structure

```
schmitt_bo/
├── README.md
├── requirements.txt
├── optimizer/
│   ├── __init__.py
│   ├── bayesian_opt.py      # Core BO engine (GP + acquisition)
│   ├── simulator.py         # ngspice interface + DC sweep extraction
│   ├── features.py          # Feature engineering for GP
│   └── visualization.py     # Convergence plots + surrogate surface
├── experiments/
│   ├── run_bo.py            # Main optimization script
│   ├── run_baselines.py     # Compare against random/grid/LHS
│   └── analyze_results.py   # Plot figures for paper/README
├── results/                 # Saved run data (JSON)
├── figures/                 # Generated plots
└── notebooks/
    └── tutorial.ipynb       # Step-by-step walkthrough
```

## Quickstart

```bash
git clone https://github.com/chennakeshavadasa/schmitt-bo
cd schmitt-bo
pip install -r requirements.txt

# Run optimization (needs ngspice + SKY130 PDK)
python experiments/run_bo.py --n_init 5 --n_iter 40 --target_ph 1.60 --target_pl 1.40

# Or run without simulator (uses cached results from our runs)
python experiments/run_bo.py --offline

# Compare methods
python experiments/run_baselines.py
python experiments/analyze_results.py
```

## Results

| Method | Sims to converge | Final error |
|---|---|---|
| Grid search | >1000 | N/A |
| Random search | ~200 | ~50mV |
| Latin Hypercube | ~80 | ~30mV |
| **Bayesian Opt (ours)** | **~45** | **<15mV** |

![Convergence Comparison](figures/convergence.png)

## What You'll Learn from This Code

- Gaussian Process regression with scikit-learn
- Expected Improvement acquisition function (from scratch)
- Multi-output surrogate modeling
- EDA tool automation (ngspice Python interface)
- Analog circuit design intuition

## Citation

If you use this in research:
```bibtex
@software{schmitt_bo_2026,
  author = {Nithin, Chennakeshava Dasa},
  title  = {Bayesian Optimization for Analog Circuit Sizing},
  year   = {2026},
  url    = {https://github.com/chennakeshavadasa/schmitt-bo}
}
```
