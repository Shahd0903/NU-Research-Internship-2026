import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from psr import estimate_delay_ami, estimate_dimension_fnn, AMIConfig, FNNConfig

# --- 1. Lorenz system definition ---

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

def lorenz(t, state, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# --- 2. Generate a long scalar time series x(t) ---

dt = 0.01
t_transient = 50.0
t_total = 500.0
n_transient = int(t_transient / dt)

x0 = [1.0, 1.0, 1.0]

t_eval_full = np.arange(0, t_transient + t_total, dt)
sol = solve_ivp(lorenz, [0, t_transient + t_total], x0,
                 t_eval=t_eval_full, method='RK45', rtol=1e-9, atol=1e-9)

x_series = sol.y[0, n_transient:]
print(f"Time series length after discarding transient: {len(x_series)} samples")

# --- 3. Part A.2 - Time delay via AMI ---

ami_cfg = AMIConfig(max_lag=100, n_bins=32, criterion="first_local_min")
tau_opt, lags, ami_vals = estimate_delay_ami(x_series, cfg=ami_cfg, standardize=True)
print(f"Optimal time delay (tau): {tau_opt} samples  ({tau_opt * dt:.3f} time units)")

plt.figure(figsize=(7, 4.5))
plt.plot(lags, ami_vals, marker='o', markersize=3, linewidth=1)
plt.axvline(tau_opt, color='red', linestyle='--', label=f'tau_opt = {tau_opt}')
plt.xlabel('Lag (samples)')
plt.ylabel('Average Mutual Information')
plt.title('AMI vs. Lag (Lorenz, x-coordinate)')
plt.legend()
plt.tight_layout()
plt.savefig('partA_ami.png', dpi=150)
plt.show()

# --- 4. Part A.3 - Embedding dimension via FNN ---

fnn_cfg = FNNConfig(max_dim=15, R_tol=10.0, A_tol=2.0, threshold_percent=1.0, theiler=int(tau_opt * 3))
m_opt, dims, fnn_pct = estimate_dimension_fnn(x_series, tau_opt, cfg=fnn_cfg, standardize=True)
print(f"Optimal embedding dimension (m): {m_opt}")

plt.figure(figsize=(7, 4.5))
plt.plot(dims, fnn_pct, marker='o', markersize=4, linewidth=1)
plt.axvline(m_opt, color='red', linestyle='--', label=f'm_opt = {m_opt}')
plt.axhline(fnn_cfg.threshold_percent, color='gray', linestyle=':', label=f'threshold = {fnn_cfg.threshold_percent}%')
plt.xlabel('Embedding dimension')
plt.ylabel('False Nearest Neighbours (%)')
plt.title('FNN % vs. Embedding Dimension (Lorenz, x-coordinate)')
plt.legend()
plt.tight_layout()
plt.savefig('partA_fnn.png', dpi=150)
plt.show()

# --- 5. Summary ---
print("\n--- Part A Summary ---")
print(f"tau = {tau_opt} samples")
print(f"m   = {m_opt}")
print("Known Lorenz state dimension = 3 (compare against m above)")

# --- 6. Bonus: side-by-side 3D comparison of true vs. reconstructed attractor ---

from psr import reconstruct_matrix

x_true, y_true, z_true = sol.y[0, n_transient:], sol.y[1, n_transient:], sol.y[2, n_transient:]
Y_recon = reconstruct_matrix(x_series, tau=tau_opt, d=m_opt)

print(f"\nReconstructed matrix shape: {Y_recon.shape}  ({m_opt} delay coords)")

fig = plt.figure(figsize=(12, 5.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(x_true, y_true, z_true, lw=0.4, color='steelblue')
ax1.set_title('Original Attractor\n(true state space)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
if m_opt >= 3:
    ax2.plot(Y_recon[:, 0], Y_recon[:, 1], Y_recon[:, 2], lw=0.4, color='steelblue')
    ax2.set_zlabel(f'x(t+{2*tau_opt})')
else:
    ax2.plot(Y_recon[:, 0], Y_recon[:, 1], np.zeros(len(Y_recon)), lw=0.4, color='steelblue')
ax2.set_title(f'Reconstructed Attractor\n(delay embedding of x(t), tau={tau_opt}, m={m_opt})')
ax2.set_xlabel('x(t)')
ax2.set_ylabel(f'x(t+{tau_opt})')

fig.suptitle('Lorenz System: Original vs. Delay-Reconstructed Attractor', fontsize=13)
fig.tight_layout()
fig.savefig('partA_original_vs_reconstructed.png', dpi=150)
plt.show()
print("Saved partA_original_vs_reconstructed.png")

# Save the series + results for use in later parts (B, D, E)
np.savez('lorenz_data.npz', x_series=x_series, dt=dt, tau=tau_opt, m=m_opt,
         full_state=sol.y[:, n_transient:])