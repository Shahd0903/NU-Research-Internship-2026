import os
import psr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')


def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

dt = 0.01
t_span =[0,510]
t_eval =np.arange(0, 510, 0.01)
y0 = [1.0, 1.0, 1.0]


sol = solve_ivp(lorenz, t_span, y0, method='RK45',t_eval=t_eval, rtol=1e-10, atol=1e-12)

n_transient=int(10/dt)
x_series=sol.y[0, n_transient:]


ami_config= psr.AMIConfig(max_lag=200, n_bins=32, criterion="first_local_min")

tau_opt, lags, ami_vals = psr.estimate_delay_ami(x_series, cfg=ami_config, standardize=True)

fnn_cfg = psr.FNNConfig(max_dim=10, R_tol=10.0, A_tol=2.0,
                    threshold_percent=1.0, theiler=50)

m_opt, dims, fnn_pct = psr.estimate_dimension_fnn(x_series, tau=tau_opt, cfg=fnn_cfg, standardize=True)




print(f"Optimal embedding dimension: m = {m_opt}")
print(f"Optimal time delay: tau = {tau_opt}")
print(f"Total samples: {len(t_eval)}")
print(f"After transient: {len(x_series)}")

plt.figure(figsize=(8, 4))
plt.plot(lags, ami_vals, 'b-', lw=1.2, label='AMI')
plt.axvline(tau_opt, color='r', ls='--', lw=1.5, label=f'τ = {tau_opt}')
plt.xlabel('Lag τ (samples)')
plt.ylabel('Average Mutual Information')
plt.title('AMI vs. Lag — Lorenz x(t)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'A1_ami_curve.png'), dpi=150)
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(dims, fnn_pct, 'bo-', lw=1.5, markersize=7, label='FNN %')
plt.axhline(1.0, color='r', ls='--', lw=1.5, label='Threshold = 1%')
plt.axvline(m_opt, color='g', ls=':', lw=1.5, label=f'm = {m_opt}')
plt.xlabel('Embedding Dimension d')
plt.ylabel('False Nearest Neighbours (%)')
plt.title('FNN% vs. Embedding Dimension — Lorenz x(t)')
plt.xticks(dims)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'A2_fnn_curve.png'), dpi=150)
plt.show()



np.save(os.path.join(os.path.dirname(__file__), 'lorenz_x.npy'), x_series)
np.savez(os.path.join(os.path.dirname(__file__), 'params.npz'),
         dt=dt, tau=tau_opt, m=m_opt)
print("Saved time series and parameters for Parts B-E.")

np.save(os.path.join(os.path.dirname(__file__), 'lorenz_full_state.npy'),
        sol.y[:, n_transient:].T)