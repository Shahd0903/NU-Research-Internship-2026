# ============================================================
#  PART D — Correlation Dimension (D2)
#  Paste this entire cell into Google Colab and run.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ────────────────────────────────────────────────
#  Regenerate the same scalar x(t) series used in Parts A-C
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

tau_opt = 16   # from Part A


def reconstruct_matrix(x, tau, d):
    N = len(x)
    M = N - (d - 1) * tau
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def correlation_sum(X, r_vals, theiler, n_ref=1500, seed=0):
    rng = np.random.default_rng(seed)
    M = len(X)
    n_ref = min(n_ref, M)
    ref_idx = rng.choice(M, size=n_ref, replace=False)

    tree = cKDTree(X)
    counts = np.zeros(len(r_vals))
    max_r = r_vals[-1]

    for i in ref_idx:
        idxs = tree.query_ball_point(X[i], max_r)
        idxs = np.asarray(idxs)
        idxs = idxs[np.abs(idxs - i) > theiler]
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(X[idxs] - X[i], axis=1)
        for k, r in enumerate(r_vals):
            counts[k] += np.sum(d < r)

    denom = n_ref * (M - 1)
    C = counts / denom
    C = np.clip(C, 1e-12, None)
    return C


# ────────────────────────────────────────────────
#  Radius grid: log-spaced, spanning ~0.1%-50% of attractor diameter
# ────────────────────────────────────────────────
X3 = reconstruct_matrix(x_series, tau_opt, 3)
rng = np.random.default_rng(1)
sample = X3[rng.choice(len(X3), size=1500, replace=False)]
pdist = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
diameter = pdist.max()
print(f"Estimated attractor diameter (m=3 embedding): {diameter:.3f}")

n_r = 30
r_vals = np.geomspace(0.001 * diameter, 0.5 * diameter, n_r)

# ────────────────────────────────────────────────
#  Correlation sums for m = 1..8
# ────────────────────────────────────────────────
dims = np.arange(1, 9)
C_all = {}
for m in dims:
    Xm = reconstruct_matrix(x_series, tau_opt, m)
    theiler = tau_opt * m
    C_all[m] = correlation_sum(Xm, r_vals, theiler=theiler, n_ref=1500)
    print(f"m={m}: C(r) computed, range [{C_all[m].min():.2e}, {C_all[m].max():.2e}]")

log_r = np.log(r_vals)

# ────────────────────────────────────────────────
#  D2(r): local slope of ln C(r) vs ln r, via central differences
# ────────────────────────────────────────────────
def local_slope(log_r, log_C):
    slope = np.full(len(log_r), np.nan)
    for i in range(1, len(log_r) - 1):
        slope[i] = (log_C[i+1] - log_C[i-1]) / (log_r[i+1] - log_r[i-1])
    return slope

D2_curves = {}
for m in dims:
    log_C = np.log(C_all[m])
    D2_curves[m] = local_slope(log_r, log_C)

# ────────────────────────────────────────────────
#  Plot 1: D2(r) vs log(r) for a few dimensions -> identify the plateau
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in [1, 2, 3, 5, 8]:
    ax.plot(log_r, D2_curves[m], "o-", ms=3, lw=1, label=f"m={m}")
ax.set_xlabel("ln(r)")
ax.set_ylabel("Local slope D2(r)")
ax.set_title("Correlation Dimension D2(r) vs. Radius")
ax.legend()
plt.tight_layout()
plt.savefig("D_D2_vs_r.png", bbox_inches="tight")
plt.show()
print("Saved: D_D2_vs_r.png")

# ────────────────────────────────────────────────
#  Plateau / scaling region: middle portion of the r-grid
# ────────────────────────────────────────────────
scaling_region = slice(10, 21)   # middle ~11 of the 30 r points

D2_vs_m = []
D2_std_vs_m = []
for m in dims:
    vals = D2_curves[m][scaling_region]
    vals = vals[~np.isnan(vals)]
    D2_vs_m.append(vals.mean())
    D2_std_vs_m.append(vals.std())
    print(f"m={m}: D2 = {vals.mean():.3f}  (std over scaling region: {vals.std():.3f})")

D2_vs_m = np.array(D2_vs_m)
D2_std_vs_m = np.array(D2_std_vs_m)

# ────────────────────────────────────────────────
#  Plot 2: D2 vs. embedding dimension m -> does it saturate?
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(dims, D2_vs_m, yerr=D2_std_vs_m, fmt="o-", color="#2a78d6",
            lw=1.5, ms=6, capsize=3)
ax.axhline(2.05, color="#e34948", ls="--", lw=1.2,
           label="literature reference (D2 ~ 2.05 for Lorenz)")
ax.set_xlabel("Embedding dimension m")
ax.set_ylabel("Correlation dimension D2")
ax.set_title("Correlation Dimension D2 vs. Embedding Dimension")
ax.legend()
plt.tight_layout()
plt.savefig("D_D2_vs_m.png", bbox_inches="tight")
plt.show()
print("Saved: D_D2_vs_m.png")

print("\n===== SUMMARY =====")
print(f"D2 estimates by m: {np.round(D2_vs_m, 3)}")
print(f"D2 at largest m tested (m=8): {D2_vs_m[-1]:.3f}")
print("Literature reference for Lorenz (sigma=10, rho=28, beta=8/3): D2 ~ 2.05")