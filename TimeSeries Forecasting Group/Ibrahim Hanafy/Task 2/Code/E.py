import os
import numpy as np
import matplotlib.pyplot as plt
from psr import reconstruct_matrix

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

params = np.load(os.path.join(os.path.dirname(__file__), 'params.npz'))
tau = int(params['tau'])
m = int(params['m'])
x_series = np.load(os.path.join(os.path.dirname(__file__), 'lorenz_x.npy'))

X = np.load(os.path.join(os.path.dirname(__file__), 'lorenz_full_state.npy'))
print(f"Attractor: {X.shape[0]} points in {X.shape[1]}D")


def box_counting(X, epsilons):
    N_boxes = []
    X_shifted = X - X.min(axis=0)

    for eps in epsilons:
        cell_indices = (X_shifted / eps).astype(int)
        unique_cells = set(map(tuple, cell_indices))
        N_boxes.append(len(unique_cells))

    return np.array(N_boxes)


# Range of box sizes
attractor_diameter = np.max(X.max(axis=0) - X.min(axis=0))
epsilons = np.logspace(np.log10(attractor_diameter * 0.5),
                       np.log10(attractor_diameter * 0.002), 30)

print("Computing box counts...")
N_boxes = box_counting(X, epsilons)

# Plot ln N(ε) vs ln(1/ε)
ln_inv_eps = np.log(1.0 / epsilons)
ln_N = np.log(N_boxes.astype(float))

plt.figure(figsize=(8, 5))
plt.plot(ln_inv_eps, ln_N, 'bo-', markersize=5)
plt.xlabel('ln(1/ε)')
plt.ylabel('ln N(ε)')
plt.title('Box-Counting — Lorenz')
plt.grid(True, alpha=0.3)

# Fit slope over scaling region (middle portion)
n = len(ln_inv_eps)
i_start = n // 4
i_end = 3 * n // 4
slope, intercept = np.polyfit(ln_inv_eps[i_start:i_end], ln_N[i_start:i_end], 1)
fit_line = slope * ln_inv_eps + intercept
plt.plot(ln_inv_eps, fit_line, 'r--', lw=2, label=f'D₀ = {slope:.3f}')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'E_boxcounting.png'), dpi=150)
plt.show()

# Load D2 for comparison
corr_data = np.load(os.path.join(os.path.dirname(__file__), 'correlation_data.npz'))
D2_values = corr_data['D2_values']
D2_saturated = D2_values[-1]

print(f"\n{'='*55}")
print(f"PART E — RESULTS")
print(f"{'='*55}")
print(f"  Box-counting dimension D₀ = {slope:.3f}")
print(f"  Correlation dimension  D₂ = {D2_saturated:.3f}")
print(f"  D₀ ≥ D₂? {'✓ Yes' if slope >= D2_saturated - 0.1 else '✗ Check'}")
print(f"  Expected: D₀ ≈ 2.05+, D₂ ≈ 2.05 for Lorenz")
print(f"{'='*55}")