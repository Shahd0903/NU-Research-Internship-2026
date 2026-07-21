"""
Part E - Fractal (box-counting) dimension of the Lorenz attractor, computed
directly on the original (standardized) 3-D trajectory points.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
FIGDIR = "figs"

from generate_series import lorenz_ode, rk4_integrate, SIGMA, RHO, BETA, DT

# Box counting needs far more points than the 24.5k used elsewhere to get a
# usable dynamic range in epsilon (3-D box counting is much more data-hungry
# than the pairwise-distance-based correlation sum in Part D), so a longer
# trajectory is generated here specifically for Part E.
_t_long, states_long = rk4_integrate(lorenz_ode, np.array([1.0, 1.0, 1.0]), DT, 2000.0,
                                      sigma=SIGMA, rho=RHO, beta=BETA)
n_transient_long = int(round(5.0 / DT))
states = states_long[n_transient_long:]
print(f"Box-counting sample size: {len(states)} points")

# standardize each coordinate so the box grid is meaningful across axes
states_std = (states - states.mean(axis=0)) / states.std(axis=0)
N = len(states_std)
diam = states_std.max(axis=0) - states_std.min(axis=0)
diam = np.linalg.norm(diam)

eps_max = 0.5 * diam
eps_min = diam / (N ** (1 / 3))          # resolution limit set by point count
eps_values = np.logspace(np.log10(eps_min), np.log10(eps_max), 25)


def box_count(points, eps):
    idx = np.floor(points / eps).astype(np.int64)
    unique_boxes = np.unique(idx, axis=0)
    return len(unique_boxes)


N_eps = np.array([box_count(states_std, eps) for eps in eps_values])

logN = np.log(N_eps)
log_inv_eps = np.log(1.0 / eps_values)


def best_linear_fit(xv, yv, min_window=8):
    n = len(xv)
    best_r2, best_slope, best_range = -np.inf, np.nan, (0, n)
    for start in range(0, n - min_window):
        for end in range(start + min_window, n + 1):
            xr, yr = xv[start:end], yv[start:end]
            slope, intercept = np.polyfit(xr, yr, 1)
            pred = slope * xr + intercept
            ss_res = np.sum((yr - pred) ** 2)
            ss_tot = np.sum((yr - yr.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            if r2 > best_r2:
                best_r2, best_slope, best_range = r2, slope, (start, end)
    return best_slope, best_range, best_r2


D0, (s, e), r2 = best_linear_fit(log_inv_eps, logN)
print(f"Box-counting dimension D0 = {D0:.3f} (R^2={r2:.4f}, "
      f"scaling region: {e-s} of {len(eps_values)} points)")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(log_inv_eps, logN, "o-", ms=4, color="#1f77b4", label="data")
ax.plot(log_inv_eps[s:e], logN[s:e], lw=3, color="black", alpha=0.4, label="scaling region")
ax.set_xlabel(r"$\ln(1/\varepsilon)$")
ax.set_ylabel(r"$\ln N(\varepsilon)$")
ax.set_title(rf"Box-counting dimension: $D_0 \approx {D0:.3f}$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/box_counting.png")
plt.close(fig)

np.savez("part_e_results.npz", D0=D0, eps_values=eps_values, N_eps=N_eps)
print("Part E complete.")
