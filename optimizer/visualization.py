"""
visualization.py
================
Publication-quality plots for the BO optimization run.

All figures saved to figures/ as both .png (200 dpi) and .pdf.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

COLORS = {
    "bo":     "#2196F3",
    "random": "#FF5722",
    "lhs":    "#4CAF50",
    "grid":   "#9C27B0",
    "target": "#F44336",
    "gp":     "#00BCD4",
    "best":   "#FFC107",
}

_RC = {
    "font.family":    "serif",
    "font.size":      9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def plot_convergence(bo_history, baselines=None,
                     target_ph=1.60, target_pl=1.38, tol=0.015):
    """
    Running-best error vs number of simulations for BO and baselines.

    Parameters
    ----------
    bo_history : dict with keys 'vph', 'vpl'
    baselines  : dict of {name: {'vph': [...], 'vpl': [...]}} or None
    """
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    def _plot(ax_ph, ax_pl, hist, label, color, ls="-", lw=2.0):
        n = range(1, len(hist["vph"]) + 1)
        best_ph = np.minimum.accumulate([abs(v - target_ph) * 1e3
                                          for v in hist["vph"]])
        best_pl = np.minimum.accumulate([abs(v - target_pl) * 1e3
                                          for v in hist["vpl"]])
        ax_ph.plot(n, best_ph, label=label, color=color, ls=ls, lw=lw)
        ax_pl.plot(n, best_pl, label=label, color=color, ls=ls, lw=lw)

    _plot(axes[0], axes[1], bo_history,
          "Bayesian Opt (ours)", COLORS["bo"], lw=2.5)

    if baselines:
        for name, hist in baselines.items():
            _plot(axes[0], axes[1], hist,
                  name.replace("_", " ").title(),
                  COLORS.get(name, "gray"), ls="--", lw=1.5)

    for ax, title in [(axes[0], "V_PH Error"), (axes[1], "V_PL Error")]:
        ax.axhline(tol * 1e3, color=COLORS["target"], ls=":",
                   lw=1.5, label=f"Tolerance (+-{tol*1e3:.0f} mV)")
        ax.axhspan(0, tol * 1e3, alpha=0.06, color=COLORS["target"])
        ax.set_xlabel("Number of Simulations")
        ax.set_ylabel("Best Error (mV)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(framealpha=0.9)

    plt.suptitle("BO vs Classical Search -- SKY130 Schmitt Trigger",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / "convergence.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "convergence.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {FIGS / 'convergence.png'}")


def plot_observation_history(Yph, Ypl, history,
                              N_INIT, target_ph=1.60,
                              target_pl_realistic=1.05,
                              target_pl_desired=1.40,
                              vpl_max=1.38, tol=0.015):
    """
    Three-panel plot: V_PH history, V_PL history, EI + xi trace.

    Parameters
    ----------
    Yph, Ypl  : lists of all observations (init + BO loop)
    history   : dict with 'vph', 'vpl', 'ei', 'xi' from the BO loop
    N_INIT    : number of LHS initialization points
    """
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    n_obs = len(Yph)
    iters = range(1, n_obs + 1)

    # Panel 1: V_PH
    axes[0].scatter(iters, Yph, c=COLORS["bo"], s=14, alpha=0.75, zorder=3)
    axes[0].axhline(target_ph, color=COLORS["target"], lw=1.5,
                    label=f"Target {target_ph} V")
    axes[0].axhspan(target_ph - tol, target_ph + tol,
                    alpha=0.12, color=COLORS["target"])
    best_ph_trace = [
        min(abs(v - target_ph) for v in Yph[: i + 1]) for i in range(n_obs)
    ]
    axes[0].plot(iters, [target_ph - e for e in best_ph_trace],
                 COLORS["bo"], lw=1.5, alpha=0.5, label="Running best")
    axes[0].set(ylabel="V_PH (V)", title="V_PH per Simulation")
    axes[0].legend(fontsize=7, loc="upper right")

    # Panel 2: V_PL
    axes[1].scatter(iters, Ypl, c="#E91E63", s=14, alpha=0.75, zorder=3)
    axes[1].axhline(target_pl_realistic, color=COLORS["target"], lw=1.5,
                    label=f"Realistic target {target_pl_realistic} V")
    axes[1].axhline(vpl_max, color="#FF9800", lw=1.2, ls="--",
                    label=f"Physics ceiling {vpl_max} V")
    axes[1].axhline(target_pl_desired, color="gray", lw=1.0, ls=":",
                    label=f"Desired {target_pl_desired} V (topology-limited)")
    axes[1].axhspan(target_pl_realistic - tol, target_pl_realistic + tol,
                    alpha=0.12, color=COLORS["target"])
    axes[1].set(ylabel="V_PL (V)", title="V_PL per Simulation")
    axes[1].legend(fontsize=7, loc="lower right")

    # Panel 3: EI + xi annealing
    if history.get("ei"):
        ei_vals = np.array([max(e, 1e-12) for e in history["ei"]])
        xi_vals = history.get("xi", [])
        bo_iters = range(N_INIT + 1, N_INIT + 1 + len(ei_vals))

        ax3 = axes[2]
        ax3.semilogy(bo_iters, ei_vals, COLORS["gp"], lw=1.5,
                     alpha=0.85, label="Acquisition value (log)")
        ax3.fill_between(bo_iters, 1e-12, ei_vals,
                         alpha=0.15, color=COLORS["gp"])

        if xi_vals:
            ax3b = ax3.twinx()
            ax3b.plot(bo_iters, xi_vals, COLORS["random"],
                      lw=1.2, ls="--", alpha=0.8, label="xi (annealing)")
            ax3b.set_ylabel("xi", color=COLORS["random"], fontsize=8)
            ax3b.tick_params(axis="y", labelcolor=COLORS["random"])
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=7, loc="upper right")

        ax3.set(ylabel="EI", title="Acquisition Function & Exploration Annealing")

    axes[2].set_xlabel("Simulation Number")
    plt.suptitle("Bayesian Optimization: Full Run History",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / "observation_history.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "observation_history.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {FIGS / 'observation_history.png'}")


def plot_surrogate_surface(surrogate, x_best, param_keys, param_bounds,
                            target_ph=1.60, target_pl=1.38, tol=0.015,
                            n_grid=40):
    """
    2D slice of the learned GP surrogate: V_PH and V_PL over L4 x W4.
    Other dimensions are fixed at x_best.
    """
    plt.rcParams.update(_RC)

    iL4 = param_keys.index("L4")
    iW4 = param_keys.index("W4")
    loL4, hiL4 = param_bounds["L4"]
    loW4, hiW4 = param_bounds["W4"]

    l4_range = np.linspace(loL4, hiL4, n_grid)
    w4_range = np.linspace(loW4, hiW4, n_grid)
    G_L4, G_W4 = np.meshgrid(l4_range, w4_range)

    X_grid = np.tile(x_best, (n_grid * n_grid, 1))
    X_grid[:, iL4] = (G_L4.ravel() - loL4) / (hiL4 - loL4)
    X_grid[:, iW4] = (G_W4.ravel() - loW4) / (hiW4 - loW4)

    mu_ph, _, mu_pl, _ = surrogate.predict(X_grid)
    MU_PH = mu_ph.reshape(n_grid, n_grid)
    MU_PL = mu_pl.reshape(n_grid, n_grid)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, MU, title, target, cmap in [
        (axes[0], MU_PH, "GP Surrogate: V_PH (L4 x W4)", target_ph, "RdYlGn"),
        (axes[1], MU_PL, "GP Surrogate: V_PL (L4 x W4)", target_pl, "Blues"),
    ]:
        cf = ax.contourf(G_L4, G_W4, MU, levels=25, cmap=cmap)
        cs = ax.contour(
            G_L4, G_W4, MU,
            levels=[target - tol, target, target + tol],
            colors=["orange", "red", "orange"],
            linewidths=[1.2, 2.0, 1.2],
        )
        ax.clabel(cs, fmt="%.2f V", fontsize=7)
        plt.colorbar(cf, ax=ax, label="Predicted Voltage (V)")
        ax.set(
            xlabel="L4 -- PMOS Length (um)",
            ylabel="W4 -- PMOS Width (um)",
            title=title,
        )

    plt.suptitle("Learned GP Surrogate: red contour = target +/- tolerance",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / "surrogate_surface.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "surrogate_surface.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {FIGS / 'surrogate_surface.png'}")
