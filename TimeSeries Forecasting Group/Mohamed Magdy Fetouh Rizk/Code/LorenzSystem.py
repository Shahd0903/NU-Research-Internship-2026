import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

initial_condition = [1.0, 1.0, 1.0]
t_span = (0, 50)
t_eval = np.arange(t_span[0], t_span[1], 0.01)

solution = solve_ivp(
    lorenz, 
    t_span, 
    initial_condition, 
    args=(sigma, rho, beta), 
    t_eval=t_eval, 
    method='RK45'
)

t = solution.t
x, y, z = solution.y

plt.figure(figsize=(10, 6))
plt.plot(t, x, label='x(t)', linewidth=1.2)
plt.plot(t, y, label='y(t)', linewidth=1.2, alpha=0.8)
plt.plot(t, z, label='z(t)', linewidth=1.2, alpha=0.8)

plt.title('Lorenz System Time-Series', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State Variables', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('lorenz_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, lw=0.5, color='b')

ax.set_title('Lorenz System Phase-Space Trajectory', fontsize=14, pad=20)
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)
ax.set_zlabel('Z Axis', fontsize=12)

ax.view_init(elev=25, azim=-45)
plt.tight_layout()

plt.savefig('lorenz_phase_space.png', dpi=300, bbox_inches='tight')
plt.close()