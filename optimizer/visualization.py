"""
visualization.py
================
Publication-quality plots for the BO optimization run.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path

FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

COLORS = {'bo':'#2196F3','random':'#FF5722','lhs':'#4CAF50','grid':'#9C27B0'}
IEEE_RC = {
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3,
}

def plot_convergence(bo_history: dict, baselines: dict = None,
                     target_ph=1.60, target_pl=1.40, tol=0.015):
    """
    Convergence plot: error vs number of simulations.
    Shows BO vs baselines.
    """
    plt.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    def plot_method(ax_ph, ax_pl, history, label, color, ls='-'):
        n_sims = range(1, len(history['vph'])+1)
        # Running best error
        best_ph = np.minimum.accumulate([abs(v-target_ph) for v in history['vph']])
        best_pl = np.minimum.accumulate([abs(v-target_pl) for v in history['vpl']])
        ax_ph.plot(n_sims, best_ph*1e3, label=label, color=color, ls=ls, lw=1.5)
        ax_pl.plot(n_sims, best_pl*1e3, label=label, color=color, ls=ls, lw=1.5)

    plot_method(axes[0], axes[1], bo_history, 'Bayesian Opt (ours)', COLORS['bo'])
    if baselines:
        for name, hist in baselines.items():
            plot_method(axes[0], axes[1], hist,
                        name.replace('_',' ').title(),
                        COLORS.get(name,'gray'), '--')

    for ax, title, target in [(axes[0],'V_PH Error',target_ph),
                               (axes[1],'V_PL Error',target_pl)]:
        ax.axhline(tol*1e3, color='red', ls=':', lw=1, label=f'Tolerance (±{tol*1e3:.0f}mV)')
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Best Error (mV)')
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(FIGS/'convergence.pdf', bbox_inches='tight')
    plt.savefig(FIGS/'convergence.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {FIGS/'convergence.png'}")


def plot_surrogate_surface(optimizer, param1='L4', param2='W4',
                            target_ph=1.60, target_pl=1.40):
    """
    2D slice of the learned GP surrogate model.
    Shows what the ML model has learned about the circuit.
    """
    plt.rcParams.update(IEEE_RC)
    result = optimizer.surrogate_predict_grid(param1, param2, n_grid=40)
    if result is None: return

    G1, G2, MU_PH, MU_PL = result

    from .simulator import PARAM_BOUNDS
    lo1,hi1 = PARAM_BOUNDS[param1]; lo2,hi2 = PARAM_BOUNDS[param2]
    X1 = lo1 + G1*(hi1-lo1); X2 = lo2 + G2*(hi2-lo2)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    for ax, data, title, target in [
            (axes[0], MU_PH, f'GP Surrogate: V_PH ({param1} vs {param2})', target_ph),
            (axes[1], MU_PL, f'GP Surrogate: V_PL ({param1} vs {param2})', target_pl)]:

        cf = ax.contourf(X1, X2, data, levels=20, cmap='RdYlGn')
        cs = ax.contour(X1, X2, data, levels=[target-0.015, target, target+0.015],
                        colors=['orange','red','orange'], linewidths=[1,2,1])
        ax.clabel(cs, fmt='%.2fV', fontsize=7)
        plt.colorbar(cf, ax=ax, label='Predicted V (V)')

        # Scatter observed points
        X_obs = np.array(optimizer.state.X)
        from .simulator import PARAM_KEYS
        i1,i2 = PARAM_KEYS.index(param1), PARAM_KEYS.index(param2)
        lo1b,hi1b = PARAM_BOUNDS[param1]; lo2b,hi2b = PARAM_BOUNDS[param2]
        ax.scatter(lo1b+X_obs[:,i1]*(hi1b-lo1b),
                   lo2b+X_obs[:,i2]*(hi2b-lo2b),
                   c='white', s=20, zorder=5, edgecolors='black', lw=0.5,
                   label='Observations')
        ax.set_xlabel(f'{param1} (µm)'); ax.set_ylabel(f'{param2} (µm)')
        ax.set_title(title); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGS/'surrogate_surface.pdf', bbox_inches='tight')
    plt.savefig(FIGS/'surrogate_surface.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {FIGS/'surrogate_surface.png'}")


def plot_observation_history(state, target_ph=1.60, target_pl=1.40, tol=0.015):
    """Plot all V_PH and V_PL observations vs iteration."""
    plt.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    n = range(1, len(state.Y_ph)+1)
    for ax, Y, target, label, color in [
            (axes[0], state.Y_ph, target_ph, 'V_PH', COLORS['bo']),
            (axes[1], state.Y_pl, target_pl, 'V_PL', '#E91E63')]:
        ax.scatter(n, Y, c=color, s=15, alpha=0.7, zorder=3)
        ax.axhline(target, color='green', lw=1.5, label=f'Target {target:.2f}V')
        ax.axhspan(target-tol, target+tol, alpha=0.15, color='green', label=f'±{tol*1e3:.0f}mV')
        ax.set_ylabel(f'{label} (V)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Simulation #')
    fig.suptitle('BO Observation History', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGS/'observation_history.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {FIGS/'observation_history.png'}")
