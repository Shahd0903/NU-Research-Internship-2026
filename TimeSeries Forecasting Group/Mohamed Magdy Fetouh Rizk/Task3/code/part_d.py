import numpy as np
import matplotlib.pyplot as plt
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

D2_m = []
plt.figure(figsize=(8, 4))

for m in range(2, max_m + 1):
    Y = reconstruct(x_series, tau, m)
    tree = cKDTree(Y)
    C_r = np.zeros_like(rs)
    for i, r in enumerate(rs):
        pairs = tree.query_pairs(r)
        pairs = [p for p in pairs if abs(p[0]-p[1]) > theiler]
        C_r[i] = 2.0 * len(pairs) / (len(Y) * (len(Y) - 1)) if len(pairs) > 0 else 0
    
    valid = C_r > 0
    log_r = np.log(rs[valid])
    log_C = np.log(C_r[valid])
    
    if len(log_r) > 5:
        slope, _ = np.polyfit(log_r[2:-2], log_C[2:-2], 1)
        D2_m.append(slope)
    else:
        D2_m.append(np.nan)
    
    plt.plot(log_r, log_C, marker='o', label=f'm={m}')

plt.title("Correlation Sum: log C(r) vs log r")
plt.xlabel("log r")
plt.ylabel("log C(r)")
plt.legend()
plt.grid()
plt.savefig('Correlation_Sum.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(2, max_m + 1), D2_m, marker='s', color='g')
plt.title("Correlation Dimension (D2) vs Embedding Dimension (m)")
plt.xlabel("Embedding Dimension (m)")
plt.ylabel("D2")
plt.grid()
plt.savefig('D2_vs_m.png', bbox_inches='tight')
plt.show()

print("D2 values for m=2 to 6:", D2_m)