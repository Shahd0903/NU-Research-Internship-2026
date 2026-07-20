"""
Part A - Phase Space Reconstruction

1. Estimate the optimal time delay using Average Mutual Information (AMI)
2. Estimate the optimal embedding dimension using False Nearest Neighbours (FNN)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from psr import (
    estimate_delay_ami,
    estimate_dimension_fnn,
    AMIConfig,
    FNNConfig,
)

# --------------------------------------------------
# Load Lorenz time series
# --------------------------------------------------

data = np.load("lorenz_series.npz")

x = data["x"]
dt = float(data["dt"])

# --------------------------------------------------
# Estimate orbital period (Theiler window)
# --------------------------------------------------

peaks, _ = find_peaks(x, distance=5)

orbital_period = int(np.median(np.diff(peaks)))

print(f"Mean orbital period : {orbital_period} samples")

# --------------------------------------------------
# Part A.1 : Average Mutual Information
# --------------------------------------------------

ami_cfg = AMIConfig(
    max_lag=200,
    n_bins=32,
    criterion="first_local_min",
)

tau_opt, lags, ami_vals = estimate_delay_ami(
    x,
    cfg=ami_cfg,
    standardize=True,
)

print(f"Optimal time delay = {tau_opt}")

plt.figure(figsize=(7,4))

plt.plot(lags, ami_vals)

plt.axvline(
    tau_opt,
    color="red",
    linestyle="--",
    label=f"tau = {tau_opt}",
)

plt.xlabel("Lag")

plt.ylabel("AMI")

plt.title("Average Mutual Information")

plt.legend()

plt.tight_layout()

plt.savefig("ami_curve.png")

plt.close()

# --------------------------------------------------
# Part A.2 : False Nearest Neighbours
# --------------------------------------------------

fnn_cfg = FNNConfig(
    max_dim=15,
    R_tol=10,
    A_tol=2,
    threshold_percent=1,
    theiler=orbital_period,
)

m_opt, dims, fnn_pct = estimate_dimension_fnn(
    x,
    tau=tau_opt,
    cfg=fnn_cfg,
    standardize=True,
)

print(f"Optimal embedding dimension = {m_opt}")

plt.figure(figsize=(7,4))

plt.plot(dims, fnn_pct, "o-")

plt.axhline(
    fnn_cfg.threshold_percent,
    color="red",
    linestyle="--",
    label="1% Threshold",
)

plt.axvline(
    m_opt,
    color="black",
    linestyle=":",
    label=f"m = {m_opt}",
)

plt.xlabel("Embedding Dimension")

plt.ylabel("FNN (%)")

plt.title("False Nearest Neighbours")

plt.legend()

plt.tight_layout()

plt.savefig("fnn_curve.png")

plt.close()

# --------------------------------------------------
# Save results
# --------------------------------------------------

np.savez(
    "part_a_results.npz",
    tau_opt=tau_opt,
    m_opt=m_opt,
    orbital_period_samples=orbital_period,
    lags=lags,
    ami_vals=ami_vals,
    dims=dims,
    fnn_pct=fnn_pct,
)

print("\n===================================")
print("Part A completed successfully.")
print("===================================")
print(f"Time Delay (tau)        : {tau_opt}")
print(f"Embedding Dimension (m) : {m_opt}")
print(f"Theiler Window          : {orbital_period}")