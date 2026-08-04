import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from psr import estimate_delay_ami, estimate_dimension_fnn, AMIConfig, FNNConfig

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ────────────────────────────────────────────────
#  Generate a long Lorenz x(t) scalar series
#  Same parameters/IC/solver/dt as Assignment 1, B.2
# ────────────────────────────────────────────────
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

dt = 0.01
T_total = 170.0                 # long enough to give >10,000 post-transient samples
t_eval = np.arange(0, T_total, dt)
state0 = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz, (0, T_total), state0, t_eval=t_eval,
                 method="RK45", rtol=1e-9, atol=1e-9)

transient_cutoff = 20.0
mask = sol.t >= transient_cutoff
x_series = sol.y[0][mask]        # scalar series: x(t) only
t_series = sol.t[mask]

print(f"Total samples after discarding transient: {len(x_series)}")

# ────────────────────────────────────────────────
#  Step 1 — Time delay via AMI
# ────────────────────────────────────────────────
ami_cfg = AMIConfig(max_lag=200, n_bins=32, criterion="first_local_min")
tau_opt, lags, ami_vals = estimate_delay_ami(x_series, cfg=ami_cfg, standardize=True)

print(f"Optimal time delay (tau): {tau_opt} samples  ->  {tau_opt*dt:.2f} time units")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(lags, ami_vals, lw=1.0, color="#2a78d6")
ax.axvline(tau_opt, color="#e34948", ls="--", lw=1.2,
           label=f"τ_opt = {tau_opt}")
ax.set_xlabel("Lag τ (samples)")
ax.set_ylabel("Average Mutual Information")
ax.set_title("Lorenz x(t) — AMI vs. Lag")
ax.legend()
plt.tight_layout()
plt.savefig("A_ami_plot.png", bbox_inches="tight")
plt.show()
print("Saved: A_ami_plot.png")

# ────────────────────────────────────────────────
#  Step 2 — Embedding dimension via FNN
# ────────────────────────────────────────────────
fnn_cfg = FNNConfig(max_dim=15, R_tol=10.0, A_tol=2.0,
                     threshold_percent=1.0, theiler=50)
m_opt, dims, fnn_pct = estimate_dimension_fnn(x_series, tau=tau_opt, cfg=fnn_cfg, standardize=True)

print(f"Optimal embedding dimension (m): {m_opt}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(dims, fnn_pct, "o-", lw=1.2, color="#2ca02c", ms=4)
ax.axhline(fnn_cfg.threshold_percent, color="#e34948", ls="--", lw=1.0,
           label=f"threshold = {fnn_cfg.threshold_percent}%")
ax.axvline(m_opt, color="#7b3fa0", ls=":", lw=1.2,
           label=f"m_opt = {m_opt}")
ax.set_xlabel("Embedding dimension d")
ax.set_ylabel("False Nearest Neighbours (%)")
ax.set_title("Lorenz x(t) — FNN % vs. Embedding Dimension")
ax.legend()
plt.tight_layout()
plt.savefig("A_fnn_plot.png", bbox_inches="tight")
plt.show()
print("Saved: A_fnn_plot.png")

# ────────────────────────────────────────────────
#  Summary
# ────────────────────────────────────────────────
print("\n===== SUMMARY =====")
print(f"tau_opt = {tau_opt} samples ({tau_opt*dt:.2f} time units)")
print(f"m_opt   = {m_opt}")
print("Theoretical expected range for Lorenz: tau ~ 10-17, m ~ 3-7")