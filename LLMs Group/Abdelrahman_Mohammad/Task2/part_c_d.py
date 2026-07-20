"""
Part C - Kolmogorov (K2) entropy
Part D - Correlation dimension (Grassberger-Procaccia)

Both are derived from the same correlation sums C_m(r), computed here for a
range of embedding dimensions m (using the same tau found in Part A) and a
log-spaced set of radii r.
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psr import reconstruct_matrix

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
FIGDIR = "figs"
RNG = np.random.default_rng(0)

data = np.load("lorenz_series.npz")
x = data["x"]
dt = float(data["dt"])
a = np.load("part_a_results.npz")
tau_opt = int(a["tau_opt"])
theiler = int(a["orbital_period_samples"])

x_std = (x - x.mean()) / x.std()

M_DIMS = [3, 4, 5, 6, 7]
N_R = 30
N_REF = 1200


def correlation_sum(Y, theiler, r_values, n_ref, rng):
    """Estimate C(r) for each r in r_values via a subsample of reference points."""
    M = len(Y)
    tree = cKDTree(Y)
    r_max = r_values[-1]
    ref_idx = rng.choice(M, size=min(n_ref, M), replace=False)

    counts = np.zeros(len(r_values))
    n_pairs_total = 0.0

    for i in ref_idx:
        idxs = tree.query_ball_point(Y[i], r_max)
        idxs = np.asarray(idxs)
        valid = np.abs(idxs - i) > theiler
        idxs = idxs[valid]
        n_excluded = min(i + theiler, M - 1) - max(i - theiler, 0) + 1
        n_valid_possible = M - n_excluded
        n_pairs_total += n_valid_possible
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(Y[idxs] - Y[i], axis=1)
        d.sort()
        counts += np.searchsorted(d, r_values, side="right")

    Cr = counts / n_pairs_total
    return Cr


def best_linear_fit(logr, logC, min_window=8):
    """Slide a window over the (logr, logC) curve and return the slope of the
    window with the best linear fit (highest R^2), among windows of at least
    min_window points, restricted to points where C>0."""
    valid = np.isfinite(logC)
    logr, logC = logr[valid], logC[valid]
    n = len(logr)
    best_r2, best_slope, best_range = -np.inf, np.nan, (0, n)
    for start in range(0, n - min_window):
        for end in range(start + min_window, n + 1):
            xr, yr = logr[start:end], logC[start:end]
            if len(xr) < min_window:
                continue
            slope, intercept = np.polyfit(xr, yr, 1)
            pred = slope * xr + intercept
            ss_res = np.sum((yr - pred) ** 2)
            ss_tot = np.sum((yr - yr.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            if r2 > best_r2 and (end - start) >= min_window:
                best_r2, best_slope, best_range = r2, slope, (start, end)
    return best_slope, best_range, best_r2


# ---------------------------------------------------------------
# Compute correlation sums for each embedding dimension
# ---------------------------------------------------------------
results = {}
diam = None
for m in M_DIMS:
    Y = reconstruct_matrix(x_std, tau_opt, m)
    if diam is None:
        sample = Y[RNG.choice(len(Y), size=min(400, len(Y)), replace=False)]
        pdist = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
        diam = pdist.max()
        r_values = np.logspace(np.log10(0.001 * diam), np.log10(0.5 * diam), N_R)
    Cr = correlation_sum(Y, theiler, r_values, N_REF, RNG)
    results[m] = Cr
    print(f"m={m}: C(r) computed, nonzero at {np.sum(Cr>0)}/{N_R} radii")

np.savez("part_cd_correlation_sums.npz", r_values=r_values, diam=diam,
         **{f"Cr_m{m}": results[m] for m in M_DIMS})

# ---------------------------------------------------------------
# Part D: correlation dimension D2 vs r (log-log plot) and vs m
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
D2_values = {}
scaling_ranges = {}
for m in M_DIMS:
    Cr = results[m]
    mask = Cr > 0
    logr = np.log(r_values[mask])
    logC = np.log(Cr[mask])
    ax.plot(logr, logC, "o-", ms=3, label=f"m={m}")
    slope, (s, e), r2 = best_linear_fit(logr, logC)
    D2_values[m] = slope
    scaling_ranges[m] = (logr[s], logr[e - 1])
    ax.plot(logr[s:e], logC[s:e], lw=3, color="black", alpha=0.4)

ax.set_xlabel(r"$\ln r$")
ax.set_ylabel(r"$\ln C(r)$")
ax.set_title("Correlation sum vs. radius (scaling region shaded)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/correlation_sum_loglog.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ms = list(D2_values.keys())
d2s = [D2_values[m] for m in ms]
ax.plot(ms, d2s, "o-", color="#d62728")
ax.set_xlabel("Embedding dimension $m$")
ax.set_ylabel("$D_2$ estimate")
ax.set_title("Correlation dimension vs. embedding dimension")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/D2_vs_m.png")
plt.close(fig)

D2_saturated = d2s[-1]
print("D2 estimates per m:", D2_values)
print(f"Saturated D2 (largest m={ms[-1]}): {D2_saturated:.3f}")

# ---------------------------------------------------------------
# Part C: K2 entropy from consecutive-m correlation-sum ratios
# ---------------------------------------------------------------
delta_t = tau_opt * dt  # time increment per extra embedding coordinate
K2_values = {}
for m in M_DIMS[:-1]:
    Cr_m, Cr_m1 = results[m], results[m + 1]
    s, e = scaling_ranges[m]  # use scaling region (in log r) identified for D2 at m
    mask = (Cr_m > 0) & (Cr_m1 > 0) & (np.log(r_values) >= s) & (np.log(r_values) <= e)
    if np.sum(mask) < 3:
        K2_values[m] = np.nan
        continue
    # NOTE: since C_m(r) ~ A r^D2 exp(-m*dt*K2), C_m(r) is *larger* than
    # C_{m+1}(r) for K2>0 (a stricter, higher-dimensional condition admits
    # fewer pairs), so ln[C_m/C_{m+1}] > 0 and K2 = +(1/dt) ln[C_m/C_{m+1}].
    # (The task sheet's formula has a leading minus sign relative to this
    # standard Grassberger-Procaccia derivation; the positive-sign form is
    # used here so that K2 comes out positive, as physically required.)
    ratio = Cr_m[mask] / Cr_m1[mask]
    K2_local = (1.0 / delta_t) * np.log(ratio)
    K2_values[m] = np.mean(K2_local)

print("K2 estimates per (m, m+1) pair:", K2_values)

fig, ax = plt.subplots(figsize=(6, 4))
pairs = list(K2_values.keys())
K2_vals_list = [K2_values[m] for m in pairs]
ax.plot(pairs, K2_vals_list, "o-", color="#9467bd")
ax.set_xlabel("$m$ (pair $m,m{+}1$)")
ax.set_ylabel("$K_2$ estimate (nats/time)")
ax.set_title("K2 entropy estimate vs. embedding dimension")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/K2_vs_m.png")
plt.close(fig)

np.savez("part_c_results.npz", K2_values_keys=np.array(pairs),
         K2_values=np.array(K2_vals_list), delta_t=delta_t)
np.savez("part_d_results.npz", D2_values_keys=np.array(ms), D2_values=np.array(d2s),
         D2_saturated=D2_saturated)
print("Parts C and D complete.")
