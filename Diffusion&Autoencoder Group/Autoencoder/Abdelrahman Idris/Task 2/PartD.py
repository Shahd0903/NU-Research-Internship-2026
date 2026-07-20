import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# --- 1. Helper: time-delay reconstruction ---

def reconstruct_matrix(x, tau, d):
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError("Time series too short for the requested dimension and delay.")
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

# --- 2. Correlation sum via Grassberger-Procaccia ---

def correlation_sum(Y, radii, max_pairs=3000):
    M = len(Y)
    if M > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(M, size=max_pairs, replace=False)
        Y = Y[idx]
        M = max_pairs

    diffs = Y[:, None, :] - Y[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(M, k=1)
    pair_dists = dists[iu]
    n_pairs = len(pair_dists)

    C_r = np.array([np.sum(pair_dists < r) / n_pairs for r in radii])
    return C_r

def estimate_correlation_dimension(logr, logC, valid_mask=None):
    finite = np.isfinite(logC)
    logr_f, logC_f = logr[finite], logC[finite]

    if valid_mask is None:
        n = len(logr_f)
        lo, hi = n // 4, 3 * n // 4
        logr_fit, logC_fit = logr_f[lo:hi], logC_f[lo:hi]
    else:
        logr_fit, logC_fit = logr_f[valid_mask], logC_f[valid_mask]

    slope, intercept = np.polyfit(logr_fit, logC_fit, 1)
    return slope, intercept, (logr_fit, logC_fit)

# --- 3. Load data and run across embedding dimensions ---

data = np.load('lorenz_data.npz')
x_series = data['x_series']
tau = int(data['tau'])

dims_to_test = [3, 4, 5, 6, 7, 8]
n_radii = 40

D2_values = []
plt.figure(figsize=(8, 6))

for d in dims_to_test:
    Y = reconstruct_matrix(x_series, tau, d)

    sample = Y[:min(2000, len(Y))]
    nbrs = NearestNeighbors(n_neighbors=2).fit(sample)
    dists, _ = nbrs.kneighbors(sample)
    r_min = np.percentile(dists[:, 1], 5)
    r_max = np.std(Y) * 2

    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    C_r = correlation_sum(Y, radii, max_pairs=2000)

    logr, logC = np.log(radii), np.log(C_r)
    D2, intercept, (logr_fit, logC_fit) = estimate_correlation_dimension(logr, logC)
    D2_values.append(D2)

    print(f"m={d}:  D2 = {D2:.4f}")

    plt.plot(logr, logC, marker='o', markersize=3, linewidth=1, label=f'm={d} (D2={D2:.2f})')
    plt.plot(logr_fit, np.polyval([D2, intercept], logr_fit), '--', color='black', linewidth=1.5)

plt.xlabel('log r')
plt.ylabel('log C(r)')
plt.title('Correlation Sum: log C(r) vs. log r (Lorenz, several embedding dims)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('partD_logC_vs_logr.png', dpi=150)
plt.show()

plt.figure(figsize=(7, 4.5))
plt.plot(dims_to_test, D2_values, marker='o', linewidth=1.5)
plt.xlabel('Embedding dimension m')
plt.ylabel('Correlation dimension D2')
plt.title('D2 vs. Embedding Dimension (saturation check)')
plt.tight_layout()
plt.savefig('partD_D2_vs_m.png', dpi=150)
plt.show()

D2_saturated = D2_values[-1]
print(f"\n--- Part D Summary ---")
print(f"D2 estimates across m: {dict(zip(dims_to_test, np.round(D2_values, 3)))}")
print(f"Saturated D2 (at largest m tested) = {D2_saturated:.4f}")
print("Known reference value for the Lorenz attractor: D2 ~ 2.05 (literature)")

np.savez('correlation_dim_results.npz', dims=dims_to_test, D2_values=D2_values, D2_saturated=D2_saturated)