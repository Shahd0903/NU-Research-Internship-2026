import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def hyperchaotic_rossler(t, state, a, b, c, d):
    x, y, z, w = state
    dx_dt = -y - z
    dy_dt = x + a * y + w
    dz_dt = b + x * z
    dw_dt = -c * z + d * w
    return [dx_dt, dy_dt, dz_dt, dw_dt]

a = 0.25
b = 3.0
c = 0.5
d = 0.05

initial_condition = [-10.0, -6.0, 0.0, 10.0]
t_span = (0, 250)
t_eval = np.arange(t_span[0], t_span[1], 0.01)

solution = solve_ivp(
    hyperchaotic_rossler, 
    t_span, 
    initial_condition, 
    args=(a, b, c, d), 
    t_eval=t_eval, 
    method='RK45'
)

t = solution.t
x, y, z, w = solution.y

plt.figure(figsize=(12, 6))
plt.plot(t, x, label='x(t)', linewidth=1.0)
plt.plot(t, y, label='y(t)', linewidth=1.0, alpha=0.8)
plt.plot(t, z, label='z(t)', linewidth=1.0, alpha=0.8)
plt.plot(t, w, label='w(t)', linewidth=1.0, alpha=0.8)

plt.title('Hyperchaotic Rössler Time-Series', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State Variables', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('hyperchaotic_rossler_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(x, y, lw=0.5, color='purple')
axes[0].set_title('x-y Projection', fontsize=12)
axes[0].set_xlabel('x', fontsize=10)
axes[0].set_ylabel('y', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(x, z, lw=0.5, color='teal')
axes[1].set_title('x-z Projection', fontsize=12)
axes[1].set_xlabel('x', fontsize=10)
axes[1].set_ylabel('z', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].plot(y, w, lw=0.5, color='crimson')
axes[2].set_title('y-w Projection', fontsize=12)
axes[2].set_xlabel('y', fontsize=10)
axes[2].set_ylabel('w', fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Hyperchaotic Rössler Phase-Space Projections', fontsize=14)
plt.tight_layout()

plt.savefig('hyperchaotic_rossler_phase_space.png', dpi=300, bbox_inches='tight')
plt.close()