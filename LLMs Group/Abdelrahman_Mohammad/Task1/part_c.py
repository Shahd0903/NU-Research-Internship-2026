"""
Part C - Sensitivity analysis: rho-sweep of Lorenz system, and step-size
sensitivity of Lorenz trajectories.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.signal import argrelextrema

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = "figs"
SIGMA, BETA = 10.0, 8 / 3


def lorenz_rhs(state, rho, sigma=SIGMA, beta=BETA):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def rk4_step(state, rho, dt):
    k1 = lorenz_rhs(state, rho)
    k2 = lorenz_rhs(state + 0.5 * dt * k1, rho)
    k3 = lorenz_rhs(state + 0.5 * dt * k2, rho)
    k4 = lorenz_rhs(state + dt * k3, rho)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_integrate(state0, rho, dt, T):
    n_steps = int(round(T / dt))
    states = np.empty((n_steps + 1, 3))
    states[0] = state0
    s = np.array(state0, dtype=float)
    for i in range(n_steps):
        s = rk4_step(s, rho, dt)
        states[i + 1] = s
    t = np.arange(n_steps + 1) * dt
    return t, states


# ---------------------------------------------------------------
# C.1 rho-sweep: local maxima of z(t) vs rho
# ---------------------------------------------------------------
rho_vals = np.arange(0.0, 30.0 + 1e-9, 0.1)
dt = 0.01
T = 60.0
transient_t = 20.0
state0 = np.array([1.0, 1.0, 1.0])

rho_points = []
z_points = []

for rho in rho_vals:
    t, states = rk4_integrate(state0, rho, dt, T)
    mask = t >= transient_t
    z = states[mask, 2]
    # local maxima
    idx = argrelextrema(z, np.greater)[0]
    if len(idx) == 0:
        # steady state (fixed point): record the settled value once
        z_points.append(z[-1])
        rho_points.append(rho)
    else:
        z_points.extend(z[idx])
        rho_points.extend([rho] * len(idx))

rho_points = np.array(rho_points)
z_points = np.array(z_points)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(rho_points, z_points, ",", color="black", alpha=0.4, markersize=0.3)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("local maxima of $z(t)$")
ax.set_title(r"Lorenz sensitivity to $\rho$ ($\sigma=10$, $\beta=8/3$)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/lorenz_rho_sweep.png")
plt.close(fig)
print("rho-sweep done.")

# ---------------------------------------------------------------
# C.2 Step-size sensitivity
# ---------------------------------------------------------------
rho_std = 28.0
T2 = 50.0
state0_2 = np.array([1.0, 1.0, 1.0])
dts = [0.001, 0.01, 0.05]
labels = ["dt=0.001 (fine)", "dt=0.01 (baseline)", "dt=0.05 (coarse)"]
colors = ["#1f77b4", "#2ca02c", "#d62728"]

results = {}
for dt_i in dts:
    t_i, s_i = rk4_integrate(state0_2, rho_std, dt_i, T2)
    results[dt_i] = (t_i, s_i)

# (i) overlay x(t)
fig, ax = plt.subplots(figsize=(8, 3.5))
for dt_i, lab, col in zip(dts, labels, colors):
    t_i, s_i = results[dt_i]
    ax.plot(t_i, s_i[:, 0], lw=0.7, label=lab, color=col, alpha=0.85)
ax.set_xlabel("t")
ax.set_ylabel("x(t)")
ax.set_title("Step-size comparison: x(t) overlay")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/stepsize_x_overlay.png")
plt.close(fig)

# (ii) overlay 3D phase trajectories
fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_subplot(111, projection="3d")
for dt_i, lab, col in zip(dts, labels, colors):
    t_i, s_i = results[dt_i]
    ax.plot(s_i[:, 0], s_i[:, 1], s_i[:, 2], lw=0.4, label=lab, color=col, alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Step-size comparison: phase-space overlay")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/stepsize_phase_overlay.png")
plt.close(fig)

# (iii) pairwise divergence |x_dt(t) - x_baseline(t)| vs t
t_base, s_base = results[0.01]
fig, ax = plt.subplots(figsize=(8, 3.5))
for dt_i, lab, col in zip([0.001, 0.05], [labels[0], labels[2]], [colors[0], colors[2]]):
    t_i, s_i = results[dt_i]
    x_interp = np.interp(t_base, t_i, s_i[:, 0])
    diff = np.abs(x_interp - s_base[:, 0])
    ax.semilogy(t_base, diff + 1e-16, lw=0.8, label=f"|{lab.split(' ')[0]} - baseline|", color=col)
ax.set_xlabel("t")
ax.set_ylabel(r"$|x_{dt}(t)-x_{0.01}(t)|$ (log scale)")
ax.set_title("Pairwise divergence from baseline (dt=0.01)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/stepsize_divergence.png")
plt.close(fig)

print("Step-size sensitivity done.")
print("Part C complete.")
