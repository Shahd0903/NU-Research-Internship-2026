import os
import numpy as np
import matplotlib.pyplot as plt
from psr import reconstruct_matrix

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

params = np.load(os.path.join(os.path.dirname(__file__), 'params.npz'))
dt = float(params['dt'])
tau = int(params['tau'])
x_series = np.load(os.path.join(os.path.dirname(__file__), 'lorenz_x.npy'))


def correlation_sum(X, r_values, theiler=50):
    N = len(X)
    max_pts = min(3000, N)
    idx = np.random.choice(N, max_pts, replace=False)
    X_sub = X[idx]
    idx_sorted = np.sort(idx)

    C = np.zeros(len(r_values))
    count = 0

    for i in range(max_pts):
        for j in range(i + 1, max_pts):
            if abs(idx[i] - idx[j]) <= theiler:
                continue
            dist = np.linalg.norm(X_sub[i] - X_sub[j])
            C += (dist < r_values).astype(float)
            count += 1

    if count > 0:
        C /= count
    return C


# Test multiple embedding dimensions
dims_to_test = [2, 3, 4, 5, 6, 7, 8]
r_values = np.logspace(-1, 1.5, 30) 
theiler = 50

print("Computing correlation sums (this takes a minute)...")

# Store results for Part C
all_log_C = {}
all_log_r = np.log10(r_values)

plt.figure(figsize=(8, 5))

for m in dims_to_test:
    print(f"  m = {m}...")
    X = reconstruct_matrix(x_series, tau, m)
    C = correlation_sum(X, r_values, theiler=theiler)

    # Avoid log(0)
    valid = C > 0
    log_r = np.log10(r_values[valid])
    log_C = np.log10(C[valid])

    all_log_C[m] = (log_r, log_C)
    plt.plot(log_r, log_C, 'o-', markersize=3, label=f'm={m}')

plt.xlabel('log₁₀(r)')
plt.ylabel('log₁₀(C(r))')
plt.title('Correlation Sum — Lorenz')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'D1_correlation_sum.png'), dpi=150)
plt.show()


print("\nCorrelation dimension D₂ vs embedding dimension:")
D2_values = []

for m in dims_to_test:
    log_r, log_C = all_log_C[m]
    n = len(log_r)
    i_start = n // 4
    i_end = 3 * n // 4
    if i_end - i_start < 3:
        D2_values.append(np.nan)
        continue
    slope, _ = np.polyfit(log_r[i_start:i_end], log_C[i_start:i_end], 1)
    D2_values.append(slope)
    print(f"  m={m}: D₂ = {slope:.3f}")

# D2 saturation plot
plt.figure(figsize=(8, 4))
plt.plot(dims_to_test, D2_values, 'ro-', markersize=8, lw=2)
plt.xlabel('Embedding Dimension m')
plt.ylabel('Correlation Dimension D₂')
plt.title('D₂ Saturation — Lorenz')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'D2_saturation.png'), dpi=150)
plt.show()

D2_saturated = D2_values[-1]
print(f"\nSaturated D₂ ≈ {D2_saturated:.3f}")
print(f"Expected: ~2.05 for Lorenz")

# Save for Part C
np.savez(os.path.join(os.path.dirname(__file__), 'correlation_data.npz'),
         dims=dims_to_test, D2_values=D2_values,
         r_values=r_values, allow_pickle=True)
# Save all_log_C dict separately
np.save(os.path.join(os.path.dirname(__file__), 'all_log_C.npy'), all_log_C, allow_pickle=True)