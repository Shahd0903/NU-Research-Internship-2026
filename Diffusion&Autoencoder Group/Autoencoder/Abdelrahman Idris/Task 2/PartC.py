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

def correlation_sum_at_r(Y, r, max_pairs=2000, seed=42):
    M = len(Y)
    if M > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=max_pairs, replace=False)
        Y = Y[idx]
        M = max_pairs

    diffs = Y[:, None, :] - Y[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(M, k=1)
    pair_dists = dists[iu]
    n_pairs = len(pair_dists)

    return np.sum(pair_dists < r) / n_pairs

data = np.load('lorenz_data.npz')
x_series = data['x_series']
tau = int(data['tau'])
dt = float(data['dt'])

dims_to_test = list(range(3, 16))

Y_ref = reconstruct_matrix(x_series, tau, dims_to_test[0])
r_fixed = 0.3 * np.std(Y_ref)
Delta_t = tau * dt

print(f"Using fixed radius r = {r_fixed:.4f}")
print(f"Using Delta_t = tau*dt = {Delta_t:.4f}")

C_m_values = []
for d in dims_to_test:
    Y = reconstruct_matrix(x_series, tau, d)
    C_m = correlation_sum_at_r(Y, r_fixed, max_pairs=2000, seed=42)
    C_m = max(C_m, 1e-12)
    C_m_values.append(C_m)
    print(f"m={d}:  C_m(r) = {C_m:.5f}")

C_m_values = np.array(C_m_values)

K2_per_m = []
m_pairs = []
for i in range(len(dims_to_test) - 1):
    ratio = C_m_values[i] / C_m_values[i + 1]
    K2_est = np.log(ratio) / Delta_t
    K2_per_m.append(K2_est)
    m_pairs.append(dims_to_test[i])
    print(f"K2 estimate using (m={dims_to_test[i]}, m+1={dims_to_test[i+1]}): {K2_est:.4f}")

K2_per_m = np.array(K2_per_m)
m_pairs = np.array(m_pairs)
K2_saturated = np.mean(K2_per_m[-3:])

plt.figure(figsize=(7, 4.5))
plt.plot(m_pairs, K2_per_m, marker='o', linewidth=1.5)
plt.axhline(K2_saturated, color='red', linestyle='--', label=f'saturated K2 = {K2_saturated:.4f}')
plt.xlabel('Embedding dimension m (paired with m+1)')
plt.ylabel('K2 estimate (per time unit)')
plt.title('K2 Entropy Estimate vs. m (saturation check)')
plt.legend()
plt.tight_layout()
plt.savefig('partC_K2_saturation.png', dpi=150)
plt.show()

print(f"\n--- Part C Summary ---")
print(f"Saturated K2 (avg of last 3 pairs) = {K2_saturated:.4f} per time unit")
print("Compare against lambda_1 (Part B). By Pesin's identity K2 <= sum of")
print("positive Lyapunov exponents; for Lorenz there's only one positive")
print("exponent, so K2 is theoretically bounded above by lambda_1.")

np.savez('K2_results.npz', K2_per_m=K2_per_m, m_pairs=m_pairs, K2_saturated=K2_saturated)