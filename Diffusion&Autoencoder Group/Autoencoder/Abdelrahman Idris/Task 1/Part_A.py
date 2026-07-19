import numpy as np
import matplotlib.pyplot as plt

# ---------- Logistic map: time series ----------

def logistic_map(r, x0, n_iter):
    x = np.empty(n_iter)
    x[0] = x0
    for i in range(1, n_iter):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x

r_chaotic = 3.9
x0 = 0.5
n_iter = 1000
transient = 300

x_series = logistic_map(r_chaotic, x0, n_iter)
n = np.arange(n_iter)

plt.figure(figsize=(9, 4))
plt.plot(n[transient:], x_series[transient:], lw=0.8)
plt.xlabel("n")
plt.ylabel("x_n")
plt.title(f"Logistic map time series (r = {r_chaotic})")
plt.tight_layout()
plt.savefig("logistic_timeseries.png", dpi=150)
plt.show()


# ---------- Hénon map: attractor ----------

def henon_map(a, b, x0, y0, n_iter):
    x = np.empty(n_iter)
    y = np.empty(n_iter)
    x[0], y[0] = x0, y0
    for i in range(1, n_iter):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x, y

a, b = 1.4, 0.3
hx0, hy0 = 0.0, 0.0
n_iter_h = 5000
transient_h = 500

x_h, y_h = henon_map(a, b, hx0, hy0, n_iter_h)

plt.figure(figsize=(6, 6))
plt.scatter(x_h[transient_h:], y_h[transient_h:], s=0.5, color="black")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Hénon attractor (a={a}, b={b})")
plt.tight_layout()
plt.savefig("henon_attractor.png", dpi=150)
plt.show()


# ---------- Bifurcation diagram: logistic map ----------

n_r = 1500
r_vals = np.linspace(2.5, 4.0, n_r)
x = np.full(n_r, 0.5)

n_iter_bif = 1000
transient_bif = 300

r_plot, x_plot = [], []

for i in range(n_iter_bif):
    x = r_vals * x * (1 - x)
    if i >= transient_bif:
        r_plot.append(r_vals)
        x_plot.append(x.copy())

r_plot = np.concatenate(r_plot)
x_plot = np.concatenate(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(r_plot, x_plot, ',k', alpha=0.25)
plt.xlabel("r")
plt.ylabel("x_n (retained)")
plt.title("Bifurcation diagram: Logistic map")
plt.tight_layout()
plt.savefig("logistic_bifurcation.png", dpi=200)
plt.show()


# ---------- Bifurcation diagram: Hénon map ----------

n_a = 600
a_vals = np.linspace(1.0, 1.4, n_a)
b_fixed = 0.3
xa = np.zeros(n_a)
ya = np.zeros(n_a)

n_iter_bif_h = 2000
transient_bif_h = 500

a_plot, xa_plot = [], []

for i in range(n_iter_bif_h):
    xa_new = 1 - a_vals * xa ** 2 + ya
    ya_new = b_fixed * xa
    xa, ya = xa_new, ya_new
    if i >= transient_bif_h:
        a_plot.append(a_vals)
        xa_plot.append(xa.copy())

a_plot = np.concatenate(a_plot)
xa_plot = np.concatenate(xa_plot)

plt.figure(figsize=(10, 6))
plt.plot(a_plot, xa_plot, ',k', alpha=0.25)
plt.xlabel("a")
plt.ylabel("x_n (retained)")
plt.title("Bifurcation diagram: Hénon map (b = 0.3)")
plt.tight_layout()
plt.savefig("henon_bifurcation.png", dpi=200)
plt.show()