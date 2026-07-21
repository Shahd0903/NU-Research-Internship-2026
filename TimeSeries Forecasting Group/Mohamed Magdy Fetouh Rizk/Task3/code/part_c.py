import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

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
max_m = 6
rs = np.logspace(-1, 1.5, 20)
theiler = 75
C_r_m = {}

for m in range(2, max_m + 1):
    Y = reconstruct(x_series, tau, m)
    tree = cKDTree(Y)
    C_r = np.zeros_like(rs)
    for i, r in enumerate(rs):
        pairs = tree.query_pairs(r)
        pairs = [p for p in pairs if abs(p[0]-p[1]) > theiler]
        C_r[i] = 2.0 * len(pairs) / (len(Y) * (len(Y) - 1)) if len(pairs) > 0 else 0
    C_r_m[m] = C_r

print("K2 Entropy Estimates:")
for m in range(2, max_m):
    ratio = C_r_m[m] / (C_r_m[m+1] + 1e-15)
    valid = (C_r_m[m+1] > 0) & (C_r_m[m] > 0)
    if np.any(valid):
        k2 = - (1 / dt) * np.mean(np.log(ratio[valid]))
        print(f"m = {m} to {m+1}: {k2:.4f}")