import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from psr import estimate_delay_ami, AMIConfig, estimate_dimension_fnn, FNNConfig

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sigma, rho, beta = 10.0, 28.0, 8/3
dt = 0.01
t_span = (0, 150) 
t_eval = np.arange(t_span[0], t_span[1], dt)
state0 = [1.0, 1.0, 1.0]

print("Generating Lorenz system data...")
sol = solve_ivp(lorenz, t_span, state0, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')

x_series = sol.y[0][5000:] 

print("Calculating optimal time delay (Tau)...")
cfg_ami = AMIConfig(max_lag=150, n_bins=32, criterion='first_local_min')
tau_opt, lags, ami_vals = estimate_delay_ami(x_series, cfg=cfg_ami, standardize=True)
print(f"Optimal time delay (Tau): {tau_opt}")

plt.figure(figsize=(8, 4))
plt.plot(lags, ami_vals)
plt.axvline(tau_opt, color='r', linestyle='--', label=f'Tau = {tau_opt}')
plt.title("Average Mutual Information (AMI)")
plt.xlabel('Lag')
plt.ylabel('AMI')
plt.legend()
plt.grid()
plt.savefig('AMI_plot.png', bbox_inches='tight')
plt.show()

print("Calculating optimal embedding dimension (m)...")
cfg_fnn = FNNConfig(max_dim=10, R_tol=10.0, A_tol=2.0, threshold_percent=1.0, theiler=75)
m_opt, dims, fnn_pct = estimate_dimension_fnn(x_series, tau=tau_opt, cfg=cfg_fnn)
print(f"Optimal embedding dimension (m): {m_opt}")

plt.figure(figsize=(8, 4))
plt.plot(dims, fnn_pct, marker='o')
plt.axhline(1.0, color='r', linestyle='--', label='1% Threshold')
plt.axvline(m_opt, color='g', linestyle='--', label=f'm = {m_opt}')
plt.title("False Nearest Neighbours (FNN)")
plt.xlabel('Embedding Dimension (m)')
plt.ylabel('FNN %')
plt.legend()
plt.grid()
plt.savefig('FNN_plot.png', bbox_inches='tight')
plt.show()