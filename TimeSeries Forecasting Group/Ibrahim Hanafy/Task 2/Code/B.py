import os
import numpy as np
import matplotlib.pyplot as plt
from lyapunov import (lyapunov_wolf_ode, AttractorODEConfig, WolfODEConfig,
                      wolf_mle, WolfConfig)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load saved data from Part A
params = np.load(os.path.join(os.path.dirname(__file__), 'params.npz'))
dt = float(params['dt'])
tau_opt = int(params['tau'])
m_opt = int(params['m'])
x_series = np.load(os.path.join(os.path.dirname(__file__), 'lorenz_x.npy'))

print(f"Loaded from Part A: dt={dt}, tau={tau_opt}, m={m_opt}")
print(f"Time series length: {len(x_series)}")


# ============================================================
# METHOD 1: Lyapunov spectrum from the ODE (ground truth)
# ============================================================
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return np.array([sigma*(y - x),
                     x*(rho - z) - y,
                     x*y - beta*z])

def lorenz_jacobian(state, t, sigma, rho, beta):
    x, y, z = state
    return np.array([[-sigma,  sigma,    0],
                     [rho - z,    -1,   -x],
                     [y,           x, -beta]])

attractor_cfg = AttractorODEConfig(
    ode=lorenz,
    jacobian=lorenz_jacobian,
    x0=np.array([1.0, 1.0, 1.0]),
    params=(10, 28, 8/3),
    dt=dt,
    transient_steps=1000,
    n_steps=30000,
    solver='RK45'
)

wolf_ode_cfg = WolfODEConfig(ortho_steps=20, log_base='e')

print("\nComputing Lyapunov spectrum from ODE...")
spectrum = lyapunov_wolf_ode(attractor_cfg, wolf_cfg=wolf_ode_cfg)

print(f"\nFull Lyapunov spectrum (nats/time):")
for i, lam in enumerate(spectrum):
    print(f"  λ{i+1} = {lam:.4f}")

lambda1_ode = spectrum[0]
print(f"\nClassification: {'Chaotic' if lambda1_ode > 0 else 'Not chaotic'}")


# ============================================================
# METHOD 2: Largest Lyapunov exponent from time series
# ============================================================
wolf_ts_cfg = WolfConfig(
    min_dist_scale=1e-3,
    max_dist_scale=1e-1,
    evolve_cap=50,
    max_replacements=500,
    angle_weight=0.5
)

print("\nComputing λ1 from time series...")
mle, debug = wolf_mle(x_series, dt=dt, tau=tau_opt, m=m_opt,
                       cfg=wolf_ts_cfg, return_debug=True)

mle_nats = mle * np.log(2)

print(f"\nλ1 from time series: {mle:.4f} bits/time")
print(f"λ1 converted to nats/time: {mle_nats:.4f}")
print(f"Segments tracked: {debug['replacements']}")
print(f"Physical time used: {debug['physical_time_used']:.2f} seconds")


# ============================================================
# COMPARISON
# ============================================================
rel_error = abs(lambda1_ode - mle_nats) / abs(lambda1_ode) * 100

print("\n" + "=" * 55)
print("PART B — COMPARISON")
print("=" * 55)
print(f"  λ1 from ODE (ground truth):  {lambda1_ode:.4f} nats/time")
print(f"  λ1 from time series:         {mle_nats:.4f} nats/time")
print(f"  Difference:                  {abs(lambda1_ode - mle_nats):.4f} nats/time")
print(f"  Relative error:              {rel_error:.1f}%")
print("=" * 55)


# ============================================================
# RESULTS FIGURE
# ============================================================
fig = plt.figure(figsize=(10, 4.5))

# Left: spectrum bar chart
ax_bar = fig.add_axes([0.08, 0.15, 0.4, 0.75])
labels = ['λ₁', 'λ₂', 'λ₃']
colors = ['red' if v > 0 else ('gray' if abs(v) < 0.01 else 'blue') for v in spectrum]
ax_bar.bar(labels, spectrum, color=colors, edgecolor='black', width=0.5)
ax_bar.axhline(0, color='black', lw=0.8)
ax_bar.set_ylabel('Exponent (nats/time)')
ax_bar.set_title('Lyapunov Spectrum (ODE)')
ax_bar.grid(True, alpha=0.3, axis='y')

# Right: comparison table
ax_table = fig.add_axes([0.55, 0.15, 0.42, 0.75])
ax_table.axis('off')
table_data = [
    ['Method', 'λ₁', 'Units'],
    ['ODE (ground truth)', f'{lambda1_ode:.4f}', 'nats/time'],
    ['Time series', f'{mle:.4f}', 'bits/time'],
    ['Time series', f'{mle_nats:.4f}', 'nats/time'],
    ['Relative error', f'{rel_error:.1f}%', ''],
]
table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(3):
    table[0, j].set_facecolor('#d4e6f1')

plt.savefig(os.path.join(FIGURES_DIR, 'B_lyapunov_results.png'), dpi=150, bbox_inches='tight')
plt.show()