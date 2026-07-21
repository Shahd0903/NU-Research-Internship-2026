"""
Part A - Discrete chaotic systems: Logistic map and Henon map.
Generates: time series, attractor plot, and bifurcation diagrams.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = "figs"

# A.2 Logistic map
def logistic_timeseries(r, x0=0.5, n_iter=1000, transient=300):
    x = np.empty(n_iter)
    x[0] = x0
    for n in range(n_iter - 1):
        x[n + 1] = r * x[n] * (1 - x[n])
    return x

r_demo = 3.9
x_series = logistic_timeseries(r_demo, x0=0.5, n_iter=1000, transient=300)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(np.arange(len(x_series)), x_series, lw=0.8, color="#1f77b4")
ax.set_xlabel("iteration $n$")
ax.set_ylabel("$x_n$")
ax.set_title(f"Logistic map time series ($r={r_demo}$, $x_0=0.5$)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/logistic_timeseries.png")
plt.close(fig)

# ---------------------------------------------------------------
# A.4 Logistic bifurcation diagram
# ---------------------------------------------------------------
r_vals = np.arange(2.5, 4.0, 0.002)
n_iter = 1000
transient = 500
n_keep = n_iter - transient

x = np.full_like(r_vals, 0.5)
r_plot = []
x_plot = []
for n in range(n_iter):
    x = r_vals * x * (1 - x)
    if n >= transient:
        r_plot.append(r_vals)
        x_plot.append(x.copy())
r_plot = np.concatenate(r_plot)
x_plot = np.concatenate(x_plot)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(r_plot, x_plot, ",", color="black", alpha=0.35, markersize=0.1)
ax.set_xlabel("$r$")
ax.set_ylabel("$x_n$ (retained)")
ax.set_title("Logistic map bifurcation diagram")
ax.set_xlim(2.5, 4.0)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/logistic_bifurcation.png")
plt.close(fig)

print("Logistic map done.")

# ---------------------------------------------------------------
# A.3 Henon map attractor
# ---------------------------------------------------------------
def henon_iterate(a, b, x0=0.0, y0=0.0, n_iter=5000, transient=500):
    x = np.empty(n_iter)
    y = np.empty(n_iter)
    x[0], y[0] = x0, y0
    for n in range(n_iter - 1):
        x[n + 1] = 1 - a * x[n] ** 2 + y[n]
        y[n + 1] = b * x[n]
    return x[transient:], y[transient:]

a_c, b_c = 1.4, 0.3
hx, hy = henon_iterate(a_c, b_c, n_iter=5000, transient=500)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(hx, hy, s=0.3, color="#d62728", alpha=0.6)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(f"Henon attractor ($a={a_c}$, $b={b_c}$)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/henon_attractor.png")
plt.close(fig)

# ---------------------------------------------------------------
# A.4 Henon bifurcation diagram (sweep a, b fixed)
# ---------------------------------------------------------------
a_vals = np.arange(1.0, 1.4, 0.002)
b_fixed = 0.3
n_iter = 1000
transient = 500

x = np.zeros_like(a_vals)
y = np.zeros_like(a_vals)
a_plot = []
x_plot2 = []
for n in range(n_iter):
    x_new = 1 - a_vals * x ** 2 + y
    y_new = b_fixed * x
    x, y = x_new, y_new
    if n >= transient:
        a_plot.append(a_vals)
        x_plot2.append(x.copy())
a_plot = np.concatenate(a_plot)
x_plot2 = np.concatenate(x_plot2)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(a_plot, x_plot2, ",", color="black", alpha=0.35, markersize=0.1)
ax.set_xlabel("$a$")
ax.set_ylabel("$x_n$ (retained)")
ax.set_title(f"Henon map bifurcation diagram ($b={b_fixed}$ fixed)")
ax.set_xlim(1.0, 1.4)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/henon_bifurcation.png")
plt.close(fig)

print("Henon map done.")
print("Part A complete.")
