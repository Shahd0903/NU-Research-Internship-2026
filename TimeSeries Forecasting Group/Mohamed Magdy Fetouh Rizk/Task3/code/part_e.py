import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

def reconstruct(x, tau, m):
    N = len(x)
    M = N - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(M)])

sigma, rho, beta = 10.0, 28.0, 8/3
dt = 0.01
t_span = (0, 150)
t_eval = np.arange(t_span[0], t_span[1], dt)
state0 = np.array([1.0, 1.0, 1.0])

sol = solve_ivp(lambda t, state: lorenz(state, t, sigma, rho, beta), t_span, state0, t_eval=t_eval, method='RK45')
x_series = sol.y[0][5000:8000] 

tau = 17
m_opt = 3

Y_opt = reconstruct(x_series, tau, m_opt)
epsilons = np.logspace(-0.5, 1.5, 20)
N_eps = []

for eps in epsilons:
    mins = np.min(Y_opt, axis=0)
    maxs = np.max(Y_opt, axis=0)
    bins = [np.arange(mins[i], maxs[i] + eps, eps) for i in range(m_opt)]
    H, _ = np.histogramdd(Y_opt, bins=bins)
    N_eps.append(np.sum(H > 0))

N_eps = np.array(N_eps)
log_eps = np.log(1.0 / epsilons)
log_N = np.log(N_eps)

valid_eps = N_eps > 0
slope_D0, _ = np.polyfit(log_eps[valid_eps], log_N[valid_eps], 1)
print(f"Box-Counting Dimension (D0): {slope_D0:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(log_eps[valid_eps], log_N[valid_eps], marker='^', color='m')
plt.title("Box-Counting Dimension: ln N(eps) vs ln(1/eps)")
plt.xlabel("ln(1/eps)")
plt.ylabel("ln N(eps)")
plt.grid()
plt.savefig('Box_Counting_D0.png', bbox_inches='tight')
plt.show()