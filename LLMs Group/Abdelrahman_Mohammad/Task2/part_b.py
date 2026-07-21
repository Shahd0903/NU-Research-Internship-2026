"""
Part B - Lyapunov exponents:
  B.2: full spectrum from the known Lorenz ODE + Jacobian (Wolf's GSR method)
  B.3: largest exponent from the scalar x(t) series (Wolf's phase-space
       tracking method), using the (tau, m) found in Part A.
"""
import numpy as np

from generate_series import lorenz_ode, lorenz_jacobian, SIGMA, RHO, BETA, DT
from lyapunov import (lyapunov_wolf_ode, AttractorODEConfig, WolfODEConfig,
                       wolf_mle, WolfConfig)

data = np.load("lorenz_series.npz")
x = data["x"]
dt = float(data["dt"])

a = np.load("part_a_results.npz")
tau_opt = int(a["tau_opt"])
m_opt = int(a["m_opt"])

# ---------------------------------------------------------------
# B.2 Ground-truth spectrum from the known ODE (Wolf GSR)
# ---------------------------------------------------------------
attractor_cfg = AttractorODEConfig(
    ode=lorenz_ode, jacobian=lorenz_jacobian,
    x0=np.array([1.0, 1.0, 1.0]), params=(SIGMA, RHO, BETA),
    dt=DT, transient_steps=1000, n_steps=30000, solver="RK45",
)
wolf_ode_cfg = WolfODEConfig(ortho_steps=20, log_base="e")
spectrum_nats = lyapunov_wolf_ode(attractor_cfg, wolf_cfg=wolf_ode_cfg)
spectrum_sorted = np.sort(spectrum_nats)[::-1]
print("Lyapunov spectrum (nats/time), largest first:", spectrum_sorted)

# A flow always has one Lyapunov exponent numerically close to zero (the
# direction along the trajectory itself), so a small absolute threshold is
# used rather than "> 0" to avoid misclassifying that null exponent as
# positive/chaotic.
POS_THRESHOLD = 0.05  # nats/time
n_positive = int(np.sum(spectrum_sorted > POS_THRESHOLD))
classification = ("hyperchaotic" if n_positive >= 2 else
                   "chaotic" if n_positive == 1 else "non-chaotic")
print(f"Positive exponents (threshold={POS_THRESHOLD}): {n_positive} -> classification: {classification}")
print(f"(second exponent = {spectrum_sorted[1]:.5f} nats/time, consistent with the "
      f"theoretical null exponent along the flow direction)")

lambda1_ode_nats = spectrum_sorted[0]
lambda1_ode_bits = lambda1_ode_nats / np.log(2)

# ---------------------------------------------------------------
# B.3 Largest exponent from the scalar time series (Wolf MLE)
# ---------------------------------------------------------------
wolf_cfg = WolfConfig(
    theiler_window=None,       # defaults to tau*m
    min_dist_scale=1e-3, max_dist_scale=1e-1,
    evolve_cap=50, max_replacements=200, angle_weight=0.3,
)
mle_bits, debug = wolf_mle(x, dt=dt, tau=tau_opt, m=m_opt, cfg=wolf_cfg, return_debug=True)
mle_nats = mle_bits * np.log(2)

print(f"wolf_mle lambda1 = {mle_bits:.4f} bits/time  ({mle_nats:.4f} nats/time)")
print(f"segments tracked: {debug['replacements']}, "
      f"physical time used: {debug['physical_time_used']:.2f}")

rel_diff_pct = 100 * abs(mle_nats - lambda1_ode_nats) / abs(lambda1_ode_nats)
print(f"Relative difference (nats): {rel_diff_pct:.1f}%")

np.savez("part_b_results.npz",
         spectrum_nats=spectrum_sorted, n_positive=n_positive,
         lambda1_ode_nats=lambda1_ode_nats, lambda1_ode_bits=lambda1_ode_bits,
         mle_bits=mle_bits, mle_nats=mle_nats,
         replacements=debug["replacements"],
         physical_time_used=debug["physical_time_used"],
         rel_diff_pct=rel_diff_pct)
print("Part B complete.")
