

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from psr import reconstruct_matrix

# --- 1. Regenerate the same Lorenz series as part_a.py (identical settings) ---

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

def lorenz(t, state, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

DT = 0.01
T_TRANSIENT = 50.0
N_SAMPLES = 20000
T_KEEP = N_SAMPLES * DT
x0 = np.array([1.0, 1.0, 1.0])

sol_transient = solve_ivp(lorenz, [0, T_TRANSIENT], x0, method='RK45',
                           t_eval=[T_TRANSIENT], rtol=1e-9, atol=1e-9)
x0_settled = sol_transient.y[:, -1]

t_eval = np.arange(0, T_KEEP, DT)
sol = solve_ivp(lorenz, [0, T_KEEP], x0_settled, method='RK45',
                 t_eval=t_eval, rtol=1e-9, atol=1e-9)

x_true, y_true, z_true = sol.y[0], sol.y[1], sol.y[2]

# --- 2. Reconstruct the attractor from x(t) alone, using tau, m from Part A ---

TAU_OPT = 16   # from estimate_delay_ami in part_a.py
M_OPT = 3      # from estimate_dimension_fnn in part_a.py

Y = reconstruct_matrix(x_true, tau=TAU_OPT, d=M_OPT)
print(f"Original series length: {len(x_true)}")
print(f"Reconstructed matrix shape: {Y.shape}  (M points, {M_OPT} delay coords)")

# --- 3. Side-by-side 3D comparison ---

fig = plt.figure(figsize=(12, 5.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(x_true, y_true, z_true, lw=0.4, color='steelblue')
ax1.set_title('Original Attractor\n(true state space)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], lw=0.4, color='steelblue')
ax2.set_title(f'Reconstructed Attractor\n(delay embedding of x(t), τ={TAU_OPT}, m={M_OPT})')
ax2.set_xlabel('x(t)')
ax2.set_ylabel(f'x(t+{TAU_OPT})')
ax2.set_zlabel(f'x(t+{2*TAU_OPT})')

fig.suptitle('Lorenz System: Original vs. Delay-Reconstructed Attractor', fontsize=13)
fig.tight_layout()
fig.savefig('part_a_original_vs_reconstructed.png', dpi=150)
print("Saved part_a_original_vs_reconstructed.png")
