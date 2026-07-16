import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rossler(t, state, a, b, c):
    x, y, z = state
    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]

a = 0.2
b = 0.2
c = 5.7

initial_condition = [1.0, 1.0, 1.0]
t_span = (0, 250)  # T = 200-300 time units
t_eval = np.arange(t_span[0], t_span[1], 0.01)

solution = solve_ivp(
    rossler, 
    t_span, 
    initial_condition, 
    args=(a, b, c), 
    t_eval=t_eval, 
    method='RK45'
)

t = solution.t
x, y, z = solution.y

plt.figure(figsize=(10, 6))
plt.plot(t, x, label='x(t)', linewidth=1.2)
plt.plot(t, y, label='y(t)', linewidth=1.2, alpha=0.8)
plt.plot(t, z, label='z(t)', linewidth=1.2, alpha=0.8)

plt.title('Rössler System Time-Series', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State Variables', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('rossler_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, lw=0.6, color='g')

ax.set_title('Rössler System Phase-Space Trajectory', fontsize=14, pad=20)
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Z Axis', fontsize=12)

ax.view_init(elev=30, azim=-45)
plt.tight_layout()

plt.savefig('rossler_phase_space.png', dpi=300, bbox_inches='tight')
plt.close()