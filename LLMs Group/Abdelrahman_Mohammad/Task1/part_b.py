"""
Part B - Continuous chaotic systems: Lorenz, Rossler, Chen, Hyperchaotic Rossler.
Generates 3-D phase-space plots and time-series plots for each.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
})

FIGDIR = "figs"


def phase_and_timeseries(t, sol, name, labels=("x", "y", "z"), proj_pairs=None,
                          color="#1f77b4"):
    """sol: array shape (n_states, n_t). Saves a 3D phase plot (if 3 states)
    or 2D projections (if >3 states), plus a time-series figure."""
    n_states = sol.shape[0]

    if n_states == 3:
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(sol[0], sol[1], sol[2], lw=0.4, color=color)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title(f"{name}: phase-space trajectory")
        fig.tight_layout()
        fig.savefig(f"{FIGDIR}/{name.lower()}_phase3d.png")
        plt.close(fig)
    else:
        # >3 states: plot requested 2D projections
        n_proj = len(proj_pairs)
        fig, axes = plt.subplots(1, n_proj, figsize=(4.2 * n_proj, 4))
        if n_proj == 1:
            axes = [axes]
        for ax, (i, j) in zip(axes, proj_pairs):
            ax.plot(sol[i], sol[j], lw=0.3, color=color)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_title(f"{labels[i]}-{labels[j]} projection")
        fig.suptitle(f"{name}: phase-space projections")
        fig.tight_layout()
        fig.savefig(f"{FIGDIR}/{name.lower()}_phase_proj.png")
        plt.close(fig)

    # time series
    fig, axes = plt.subplots(n_states, 1, figsize=(7, 1.6 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    for k in range(n_states):
        axes[k].plot(t, sol[k], lw=0.6, color=color)
        axes[k].set_ylabel(labels[k])
        axes[k].grid(alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"{name}: time series")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{name.lower()}_timeseries.png")
    plt.close(fig)


# ---------------------------------------------------------------
# B.2 Lorenz system
# ---------------------------------------------------------------
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


t_span = (0, 50)
t_eval = np.arange(0, 50, 0.01)
sol = solve_ivp(lorenz, t_span, [1, 1, 1], method="RK45", t_eval=t_eval,
                 max_step=0.01, rtol=1e-9, atol=1e-9)
phase_and_timeseries(sol.t, sol.y, "Lorenz", color="#1f77b4")
print("Lorenz done.")

# ---------------------------------------------------------------
# B.3 Rossler system
# ---------------------------------------------------------------
def rossler(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


t_span = (0, 250)
t_eval = np.arange(0, 250, 0.05)
sol = solve_ivp(rossler, t_span, [1, 1, 1], method="RK45", t_eval=t_eval,
                 max_step=0.01, rtol=1e-9, atol=1e-9)
phase_and_timeseries(sol.t, sol.y, "Rossler", color="#2ca02c")
print("Rossler done.")

# ---------------------------------------------------------------
# B.4 Chen system
# ---------------------------------------------------------------
def chen(t, state, a=35.0, b=3.0, c=28.0):
    x, y, z = state
    return [a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z]


t_span = (0, 50)
t_eval = np.arange(0, 50, 0.002)
sol = solve_ivp(chen, t_span, [-0.1, 0.5, -0.6], method="RK45", t_eval=t_eval,
                 max_step=0.002, rtol=1e-10, atol=1e-10)
phase_and_timeseries(sol.t, sol.y, "Chen", color="#d62728")
print("Chen done.")

# ---------------------------------------------------------------
# B.5 Hyperchaotic Rossler system
# ---------------------------------------------------------------
def hyper_rossler(t, state, a=0.25, b=3.0, c=0.5, d=0.05):
    x, y, z, w = state
    return [-y - z, x + a * y + w, b + x * z, -c * z + d * w]


t_span = (0, 250)
t_eval = np.arange(0, 250, 0.02)
sol = solve_ivp(hyper_rossler, t_span, [-10, -6, 0, 10], method="RK45",
                 t_eval=t_eval, max_step=0.01, rtol=1e-9, atol=1e-9)
phase_and_timeseries(sol.t, sol.y, "HyperRossler", labels=("x", "y", "z", "w"),
                      proj_pairs=[(0, 1), (0, 2), (1, 3)], color="#9467bd")
print("Hyperchaotic Rossler done.")

print("Part B complete.")
