# ============================================================
#  PART E — Fractal (Box-Counting) Dimension
#  Paste this entire cell into Google Colab and run.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ────────────────────────────────────────────────
#  Regenerate the same scalar x(t) series used in Parts A-D
# ────────────────────────────────────────────────
def lorenz_flat(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

dt = 0.01
T_total = 170.0
t_eval = np.arange(0, T_total, dt)
state0 = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz_flat, (0, T_total), state0, t_eval=t_eval,
                 method="RK45", rtol=1e-9, atol=1e-9)

transient_cutoff = 20.0
mask = sol.t >= transient_cutoff
x_series = sol.y[0][mask]
print(f"Series length after transient discard: {len(x_series)}")

tau_opt, m_opt = 16, 3   # from Part A


def reconstruct_matrix(x, tau, d):
    N = len(x)
    M = N - (d - 1) * tau
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


X = reconstruct_matrix(x_series, tau_opt, m_opt)
M = len(X)
print(f"Reconstructed attractor: {M} points in {m_opt}-D phase space (tau={tau_opt})")

# ────────────────────────────────────────────────
#  Box-counting: for a given epsilon, count occupied grid cells
# ────────────────────────────────────────────────
def box_count(X, eps):
    """Count occupied boxes of size eps covering the point cloud X."""
    box_idx = np.floor(X / eps).astype(np.int64)
    # unique rows = unique occupied boxes
    view = np.ascontiguousarray(box_idx).view(
        np.dtype((np.void, box_idx.dtype.itemsize * box_idx.shape[1])))
    n_unique = len(np.unique(view))
    return n_unique


# ────────────────────────────────────────────────
#  Box-size (epsilon) range: from attractor diameter down to a
#  resolution limit set by the number of data points (below this,
#  boxes are mostly empty/singleton, which artificially saturates
#  N(eps) toward M and biases the slope upward)
# ────────────────────────────────────────────────
mins = X.min(axis=0)
maxs = X.max(axis=0)
diameter = np.linalg.norm(maxs - mins)
print(f"Attractor bounding-box diagonal (diameter proxy): {diameter:.3f}")

eps_min_resolution = diameter / (M ** (1.0 / m_opt))
print(f"Resolution-limit epsilon (diameter / M^(1/m)): {eps_min_resolution:.4f}")

n_eps = 30
eps_vals = np.geomspace(diameter * 0.5, eps_min_resolution * 0.3, n_eps)

N_eps = np.array([box_count(X, eps) for eps in eps_vals])
print("\nBox counts:")
for e, n in zip(eps_vals, N_eps):
    print(f"  eps={e:8.4f}   N(eps)={n:6d}")

# ────────────────────────────────────────────────
#  ln N(eps) vs ln(1/eps), and the local slope (= local D0 estimate)
# ────────────────────────────────────────────────
log_inv_eps = np.log(1.0 / eps_vals)
log_N = np.log(N_eps)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(log_inv_eps, log_N, "o-", color="#2a78d6", ms=4, lw=1.2)
ax.set_xlabel("ln(1/eps)")
ax.set_ylabel("ln N(eps)")
ax.set_title("Box-Counting: ln N(eps) vs. ln(1/eps)")
plt.tight_layout()
plt.savefig("E_logN_vs_loginveps.png", bbox_inches="tight")
plt.show()
print("\nSaved: E_logN_vs_loginveps.png")


def local_slope(x, y):
    slope = np.full(len(x), np.nan)
    for i in range(1, len(x) - 1):
        slope[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return slope

D0_r = local_slope(log_inv_eps, log_N)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(log_inv_eps, D0_r, "o-", color="#e34948", ms=4, lw=1.2)
ax.set_xlabel("ln(1/eps)")
ax.set_ylabel("Local slope D0(eps)")
ax.set_title("Box-Counting Dimension D0(eps) vs. ln(1/eps)")
plt.tight_layout()
plt.savefig("E_D0_vs_eps.png", bbox_inches="tight")
plt.show()
print("Saved: E_D0_vs_eps.png")

# ────────────────────────────────────────────────
#  Identify the scaling region and average -> D0 estimate
#  (exclude large-eps end: too few boxes to show structure;
#   exclude small-eps end: resolution-limited, biased toward M)
# ────────────────────────────────────────────────
scaling_region = slice(16, 28)   # stable middle-to-fine portion; excludes the sparse, noisy
                                 # large-eps end (N<~100, only a handful of boxes) where the
                                 # slope is still climbing/noisy rather than settled

vals = D0_r[scaling_region]
vals = vals[~np.isnan(vals)]
D0_estimate = vals.mean()
D0_std = vals.std()

print(f"\n===== SUMMARY =====")
print(f"D0 estimate (scaling region mean): {D0_estimate:.3f}  (std: {D0_std:.3f})")
print(f"Compare with D2 (Part D plateau, ~2.02): D0 should be >= D2")