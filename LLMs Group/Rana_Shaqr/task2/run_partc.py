# ============================================================
#  PART C — Kolmogorov (K2) Entropy
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
#  Regenerate the same scalar x(t) series used in Parts A & B
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

tau_opt = 16          # from Part A
dt_embed = tau_opt * dt   # physical time spanned by one embedding lag step
print(f"Delta_t used in K2 formula (tau*dt): {dt_embed:.3f} time units")


def reconstruct_matrix(x, tau, d):
    N = len(x)
    M = N - (d - 1) * tau
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def correlation_sum(X, r_vals, theiler, n_ref=1500, seed=0):
    """
    Grassberger-Procaccia correlation sum C(r) for a range of radii,
    excluding pairs within a Theiler window of each other in time.
    Uses a random subsample of reference points against the full
    dataset (via a KD-tree) to keep this tractable on ~15,000 points.
    """
    rng = np.random.default_rng(seed)
    M = len(X)
    n_ref = min(n_ref, M)
    ref_idx = rng.choice(M, size=n_ref, replace=False)

    tree = cKDTree(X)
    counts = np.zeros(len(r_vals))
    max_r = r_vals[-1]

    for i in ref_idx:
        # all neighbours within the largest radius, once
        idxs = tree.query_ball_point(X[i], max_r)
        idxs = np.asarray(idxs)
        idxs = idxs[np.abs(idxs - i) > theiler]      # Theiler window exclusion
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(X[idxs] - X[i], axis=1)
        for k, r in enumerate(r_vals):
            counts[k] += np.sum(d < r)

    denom = n_ref * (M - 1)   # approx. normalisation (pairs considered per reference point)
    C = counts / denom
    C = np.clip(C, 1e-12, None)   # avoid log(0) downstream
    return C


# ────────────────────────────────────────────────
#  Correlation sums C_m(r) for a range of embedding dimensions
# ────────────────────────────────────────────────
dims = np.arange(1, 9)          # m = 1..8
n_r = 15

# Build a radius grid from the pairwise-distance scale of the m=3 embedding
X3 = reconstruct_matrix(x_series, tau_opt, 3)
sample = X3[np.random.default_rng(1).choice(len(X3), size=1500, replace=False)]
pdist = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
pdist = pdist[np.triu_indices_from(pdist, k=1)]
scale = np.median(pdist)
r_vals = np.geomspace(0.02 * scale, 0.8 * scale, n_r)

C_all = {}
for m in dims:
    Xm = reconstruct_matrix(x_series, tau_opt, m)
    theiler = tau_opt * m
    C_all[m] = correlation_sum(Xm, r_vals, theiler=theiler, n_ref=1500)
    print(f"m={m}: C(r) range = [{C_all[m].min():.2e}, {C_all[m].max():.2e}]")

# ────────────────────────────────────────────────
#  Plot log C_m(r) vs log r for a few dimensions
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for m in [1, 2, 3, 5, 8]:
    ax.plot(np.log(r_vals), np.log(C_all[m]), "o-", ms=3, lw=1, label=f"m={m}")
ax.set_xlabel("ln(r)")
ax.set_ylabel("ln C_m(r)")
ax.set_title("Correlation Sum vs. Radius (Lorenz x(t) reconstruction)")
ax.legend()
plt.tight_layout()
plt.savefig("C_logC_vs_logr.png", bbox_inches="tight")
plt.show()
print("Saved: C_logC_vs_logr.png")

# ────────────────────────────────────────────────
#  K2 estimate: -(1/Delta_t) * ln[C_m(r)/C_{m+1}(r)],
#  averaged over a "scaling region" (middle portion of the r-grid)
# ────────────────────────────────────────────────
scaling_region = slice(4, 11)   # middle ~7 of the 15 r points

K2_vs_m = []
for m in dims[:-1]:
    ratio = C_all[m][scaling_region] / C_all[m + 1][scaling_region]
    K2_r = (1.0 / dt_embed) * np.log(ratio)
    K2_vs_m.append(K2_r.mean())
    print(f"K2 estimate from m={m}->m+1={m+1}: {K2_r.mean():.4f} nats/time "
          f"(std over scaling region: {K2_r.std():.4f})")

K2_vs_m = np.array(K2_vs_m)
m_pairs = dims[:-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(m_pairs, K2_vs_m, "o-", color="#2a78d6", lw=1.5, ms=6)
ax.axhline(0.9003, color="#e34948", ls="--", lw=1.2,
           label="lambda_1 from Part B (0.9003 nats/time)")
ax.set_xlabel("Embedding dimension m (K2 computed from m -> m+1)")
ax.set_ylabel("K2 estimate (nats/time)")
ax.set_title("K2 Entropy Estimate vs. Embedding Dimension")
ax.legend()
plt.tight_layout()
plt.savefig("C_K2_vs_m.png", bbox_inches="tight")
plt.show()
print("Saved: C_K2_vs_m.png")

print("\n===== SUMMARY =====")
print(f"K2 estimates by m: {np.round(K2_vs_m, 4)}")
print(f"K2 at largest m tested: {K2_vs_m[-1]:.4f} nats/time")
print(f"Pesin identity reference (lambda_1 from Part B): 0.9003 nats/time")