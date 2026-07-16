import numpy as np
import matplotlib.pyplot as plt


def rk4_integrate(deriv, state0, dt, T):
    n_steps = int(T / dt)
    state0 = np.array(state0, dtype=float)
    dim = len(state0)
    traj = np.empty((n_steps + 1, dim))
    traj[0] = state0
    t = np.linspace(0, n_steps * dt, n_steps + 1)

    x = state0.copy()
    for i in range(n_steps):
        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj[i + 1] = x

    return t, traj


def lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


# ==================== C.1  rho-sweep (bifurcation-style plot) ====================

sigma, beta = 10.0, 8/3
IC = [1, 1, 1]
dt_sweep = 0.01
T_sweep = 50            # same length as B.2
transient_cutoff = 20   # discard first 20 time units

rho_values = np.arange(0, 30.01, 0.25)   # within doc's 0.1-0.5 step range
rho_list = []
z_maxima = []

for rho in rho_values:
    deriv = lambda state, rho=rho: lorenz_deriv(state, sigma=sigma, rho=rho, beta=beta)
    t, traj = rk4_integrate(deriv, IC, dt=dt_sweep, T=T_sweep)
    z = traj[:, 2]

    mask = t > transient_cutoff
    z_kept = z[mask]

    # local maxima: bigger than both neighbours
    is_peak = (z_kept[1:-1] > z_kept[:-2]) & (z_kept[1:-1] > z_kept[2:])
    peaks = z_kept[1:-1][is_peak]

    rho_list.extend([rho] * len(peaks))
    z_maxima.extend(peaks)

plt.figure(figsize=(10, 6))
plt.plot(rho_list, z_maxima, ',k', alpha=0.5)
plt.xlabel("ρ")
plt.ylabel("local maxima of z(t)")
plt.title("Lorenz system: bifurcation diagram (z maxima vs ρ)")
plt.tight_layout()
plt.savefig("lorenz_rho_sweep.png", dpi=150)
plt.show()


# ==================== C.2  step-size sensitivity ====================

IC = [1, 1, 1]
T = 50
dts = [0.001, 0.01, 0.05]
results = {}

for dt in dts:
    t, traj = rk4_integrate(lambda s: lorenz_deriv(s), IC, dt=dt, T=T)
    results[dt] = (t, traj)

# (i) overlay x(t)
plt.figure(figsize=(10, 5))
for dt in dts:
    t, traj = results[dt]
    plt.plot(t, traj[:, 0], label=f"dt={dt}", lw=0.8)
plt.xlabel("t"); plt.ylabel("x(t)")
plt.title("Lorenz: x(t) at different step sizes")
plt.legend()
plt.tight_layout()
plt.savefig("stepsize_xt_overlay.png", dpi=150)
plt.show()

# (ii) overlay 3-D phase-space trajectories
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection='3d')
for dt in dts:
    t, traj = results[dt]
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"dt={dt}", lw=0.6)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Lorenz: phase-space overlay at different dt")
ax.legend()
plt.tight_layout()
plt.savefig("stepsize_3d_overlay.png", dpi=150)
plt.show()

# (iii) divergence |x_dt(t) - x_baseline(t)| vs t
baseline_dt = 0.01
t_base, traj_base = results[baseline_dt]
x_base = traj_base[:, 0]

plt.figure(figsize=(10, 5))
for dt in dts:
    if dt == baseline_dt:
        continue
    t_dt, traj_dt = results[dt]
    x_interp = np.interp(t_base, t_dt, traj_dt[:, 0])
    divergence = np.abs(x_interp - x_base)
    plt.plot(t_base, divergence, label=f"|x_{dt} - x_{baseline_dt}|")
plt.xlabel("t"); plt.ylabel("divergence")
plt.yscale("log")
plt.title("Lorenz: divergence from baseline dt over time")
plt.legend()
plt.tight_layout()
plt.savefig("stepsize_divergence.png", dpi=150)
plt.show()