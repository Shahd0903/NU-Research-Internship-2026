
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_matrix(x, tau, d):
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError("Time series too short for the requested dimension and delay.")
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def box_count_averaged(Y, eps, n_offsets=8, seed=0):
    rng = np.random.default_rng(seed)
    mins = Y.min(axis=0)
    d = Y.shape[1]
    counts = []
    for _ in range(n_offsets):
        offset = rng.uniform(0, eps, size=d)
        box_idx = np.floor((Y - mins + offset) / eps).astype(np.int64)
        counts.append(len(np.unique(box_idx, axis=0)))
    return np.mean(counts)

data = np.load('lorenz_data.npz')
x_series = data['x_series']
tau = int(data['tau'])
m = int(data['m'])

Y = reconstruct_matrix(x_series, tau, m)

max_points = 40000
if len(Y) > max_points:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Y), size=max_points, replace=False)
    Y = Y[idx]

print(f"Using embedding dimension m={m}, {len(Y)} points for box-counting")

attractor_span = Y.max(axis=0) - Y.min(axis=0)
eps_max = np.max(attractor_span) / 4
eps_min = np.max(attractor_span) / 100
n_eps = 20

eps_values = np.logspace(np.log10(eps_max), np.log10(eps_min), n_eps)

N_eps = []
for eps in eps_values:
    count = box_count_averaged(Y, eps, n_offsets=8, seed=0)
    N_eps.append(count)
    print(f"eps = {eps:.5f}  ->  N(eps) = {count:.1f}")

N_eps = np.array(N_eps)

log_inv_eps = np.log(1.0 / eps_values)
log_N = np.log(N_eps)

n = len(log_N)
lo, hi = n // 4, 3 * n // 4
D0, intercept = np.polyfit(log_inv_eps[lo:hi], log_N[lo:hi], 1)

print(f"\nBox-counting (fractal) dimension D0 = {D0:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(log_inv_eps, log_N, marker='o', markersize=4, linewidth=1, label='ln N(eps) vs ln(1/eps)')
plt.plot(log_inv_eps[lo:hi], np.polyval([D0, intercept], log_inv_eps[lo:hi]),
         '--', color='red', linewidth=2, label=f'fit slope (D0) = {D0:.3f}')
plt.xlabel('ln(1/eps)')
plt.ylabel('ln N(eps)')
plt.title('Box-Counting Dimension (Lorenz attractor)')
plt.legend()
plt.tight_layout()
plt.savefig('partE_box_counting.png', dpi=150)
plt.show()

try:
    d_data = np.load('correlation_dim_results.npz')
    D2_saturated = float(d_data['D2_saturated'])
    print(f"\n--- Part E Summary ---")
    print(f"D0 (box-counting) = {D0:.4f}")
    print(f"D2 (correlation, from Part D) = {D2_saturated:.4f}")
    print("Literature value for Lorenz: D0 ~ 2.06, D2 ~ 2.05.")
except FileNotFoundError:
    print(f"D0 (box-counting) = {D0:.4f}")

np.savez('box_counting_results.npz', eps_values=eps_values, N_eps=N_eps, D0=D0)