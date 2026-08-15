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


all_log_C = np.load(os.path.join(os.path.dirname(__file__), 'all_log_C.npy'),
                     allow_pickle=True).item()
corr_data = np.load(os.path.join(os.path.dirname(__file__), 'correlation_data.npz'))
r_values = corr_data['r_values']
dims = list(corr_data['dims'])


delta_t = tau * dt

print("Computing K₂ entropy estimates...")
print(f"Δt = τ × dt = {tau} × {dt} = {delta_t}")

K2_per_m = []
m_pairs = []

for i in range(len(dims) - 1):
    m = dims[i]
    m1 = dims[i + 1]

    log_r_m, log_C_m = all_log_C[m]
    log_r_m1, log_C_m1 = all_log_C[m1]

    r_min = max(log_r_m.min(), log_r_m1.min())
    r_max = min(log_r_m.max(), log_r_m1.max())

    common_r = np.linspace(r_min, r_max, 20)
    C_m_interp = np.interp(common_r, log_r_m, log_C_m)
    C_m1_interp = np.interp(common_r, log_r_m1, log_C_m1)

    diff = (C_m_interp - C_m1_interp) * np.log(10)
    K2 = diff / delta_t

    n = len(K2)
    K2_mid = K2[n // 4: 3 * n // 4]
    K2_avg = np.mean(K2_mid[np.isfinite(K2_mid)])

    K2_per_m.append(K2_avg)
    m_pairs.append(f"{m}→{m1}")
    print(f"  m={m}→{m1}: K₂ ≈ {K2_avg:.4f} nats/time")

# Plot K2 vs embedding dimension
plt.figure(figsize=(8, 4))
plt.plot(dims[:-1], K2_per_m, 'rs-', markersize=8, lw=2)
plt.xlabel('Embedding Dimension m')
plt.ylabel('K₂ (nats/time)')
plt.title('Kolmogorov Entropy K₂ vs. Embedding Dimension — Lorenz')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'C_K2_entropy.png'), dpi=150)
plt.show()

K2_final = K2_per_m[-1]
lambda1 = 0.8988  # from Part B ODE

print(f"\n{'='*55}")
print(f"PART C — RESULTS")
print(f"{'='*55}")
print(f"  K₂ (last pair): {K2_final:.4f} nats/time")
print(f"  Sum of positive λ (Pesin): {lambda1:.4f} nats/time")
print(f"  K₂ should be ≤ Σλ+ (Pesin's identity)")
print(f"  Saturates? {'Yes — finite chaos' if K2_final > 0 else 'Check values'}")
print(f"{'='*55}")