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


# ---------- B.2 Lorenz system ----------

def lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

t_lorenz, traj_lorenz = rk4_integrate(lorenz_deriv, [1, 1, 1], dt=0.01, T=50)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(traj_lorenz[:, 0], traj_lorenz[:, 1], traj_lorenz[:, 2], lw=0.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Lorenz system: phase-space trajectory")
plt.tight_layout()
plt.savefig("lorenz_3d.png", dpi=150)
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
labels = ["x(t)", "y(t)", "z(t)"]
for i in range(3):
    axs[i].plot(t_lorenz, traj_lorenz[:, i], lw=0.6)
    axs[i].set_ylabel(labels[i])
axs[-1].set_xlabel("t")
fig.suptitle("Lorenz system: time series")
plt.tight_layout()
plt.savefig("lorenz_timeseries.png", dpi=150)
plt.show()


# ---------- B.3 Rössler system ----------

def rossler_deriv(state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

t_rossler, traj_rossler = rk4_integrate(rossler_deriv, [1, 1, 1], dt=0.01, T=250)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(traj_rossler[:, 0], traj_rossler[:, 1], traj_rossler[:, 2], lw=0.4)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Rössler system: phase-space trajectory")
plt.tight_layout()
plt.savefig("rossler_3d.png", dpi=150)
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for i in range(3):
    axs[i].plot(t_rossler, traj_rossler[:, i], lw=0.5)
    axs[i].set_ylabel(labels[i])
axs[-1].set_xlabel("t")
fig.suptitle("Rössler system: time series")
plt.tight_layout()
plt.savefig("rossler_timeseries.png", dpi=150)
plt.show()


# ---------- B.4 Chen system ----------

def chen_deriv(state, a=35.0, b=3.0, c=28.0):
    x, y, z = state
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz])

t_chen, traj_chen = rk4_integrate(chen_deriv, [-0.1, 0.5, -0.6], dt=0.002, T=50)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(traj_chen[:, 0], traj_chen[:, 1], traj_chen[:, 2], lw=0.4)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Chen system: phase-space trajectory")
plt.tight_layout()
plt.savefig("chen_3d.png", dpi=150)
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for i in range(3):
    axs[i].plot(t_chen, traj_chen[:, i], lw=0.5)
    axs[i].set_ylabel(labels[i])
axs[-1].set_xlabel("t")
fig.suptitle("Chen system: time series")
plt.tight_layout()
plt.savefig("chen_timeseries.png", dpi=150)
plt.show()


# ---------- B.5 Hyperchaotic Rössler system ----------

def hyperchaotic_rossler_deriv(state, a=0.25, b=3.0, c=0.5, d=0.05):
    x, y, z, w = state
    dx = -y - z
    dy = x + a * y + w
    dz = b + x * z
    dw = -c * z + d * w
    return np.array([dx, dy, dz, dw])

t_hr, traj_hr = rk4_integrate(hyperchaotic_rossler_deriv, [-10, -6, 0, 10], dt=0.01, T=250)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
projections = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 3, "y", "w")]
for ax, (i, j, xl, yl) in zip(axs, projections):
    ax.plot(traj_hr[:, i], traj_hr[:, j], lw=0.3)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{xl}-{yl} projection")
fig.suptitle("Hyperchaotic Rössler: 2D projections")
plt.tight_layout()
plt.savefig("hyperchaotic_rossler_projections.png", dpi=150)
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
labels_hr = ["x(t)", "y(t)", "z(t)", "w(t)"]
for i in range(4):
    axs[i].plot(t_hr, traj_hr[:, i], lw=0.5)
    axs[i].set_ylabel(labels_hr[i])
axs[-1].set_xlabel("t")
fig.suptitle("Hyperchaotic Rössler: time series")
plt.tight_layout()
plt.savefig("hyperchaotic_rossler_timeseries.png", dpi=150)
plt.show()