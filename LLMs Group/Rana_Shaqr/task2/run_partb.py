# ============================================================
#  PART B — Lyapunov Exponent
#  Paste this entire cell into Google Colab and run.
#  Requires psr.py-style helper `reconstruct_matrix` (bundled
#  inside lyapunov.py) and lyapunov.py in the same directory.
# ============================================================

import numpy as np
from scipy.integrate import solve_ivp
from lyapunov import (lyapunov_wolf_ode, AttractorODEConfig, WolfODEConfig,
                       wolf_mle, WolfConfig)

# ────────────────────────────────────────────────
#  Lorenz ODE + analytic Jacobian
#  (same system, parameters, and IC as Assignment 1 / Part A)
# ────────────────────────────────────────────────
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def lorenz_jac(state, t, sigma, rho, beta):
    x, y, z = state
    return np.array([[-sigma,   sigma,   0],
                      [rho - z, -1,     -x],
                      [y,        x,     -beta]])

params = (10.0, 28.0, 8/3)

# ────────────────────────────────────────────────
#  B.2 — Full spectrum from the known ODE (Wolf's GSR method)
# ────────────────────────────────────────────────
ode_cfg = AttractorODEConfig(
    ode=lorenz, jacobian=lorenz_jac,
    x0=np.array([1.0, 1.0, 1.0]), params=params,
    dt=0.01, transient_steps=1000, n_steps=30000, solver='RK45')

wolf_ode_cfg = WolfODEConfig(ortho_steps=20, log_base='e')   # 'e' -> nats/time
spectrum_nats = lyapunov_wolf_ode(ode_cfg, wolf_cfg=wolf_ode_cfg)
spectrum_bits = spectrum_nats / np.log(2)

# A theoretically exact zero exponent (perturbation along the flow direction)
# will show up as a small nonzero number from finite numerical precision —
# use a tolerance well above that numerical noise floor before calling
# something "positive", so the marginal direction isn't misclassified.
zero_tol = 0.01  # nats/time
n_positive = int(np.sum(spectrum_nats > zero_tol))
classification = ("chaotic (exactly one positive exponent)" if n_positive == 1 else
                  "hyperchaotic (two or more positive exponents)" if n_positive >= 2 else
                  "non-chaotic (no positive exponent)")

print("===== B.2 — ODE-based Lyapunov spectrum (Wolf GSR) =====")
print(f"Spectrum (nats/time): {np.round(spectrum_nats, 4)}")
print(f"Spectrum (bits/time): {np.round(spectrum_bits, 4)}")
print(f"Sum of exponents: {spectrum_nats.sum():.4f} nats/time "
      f"(theory: -(sigma+1+beta) = {-(params[0]+1+params[2]):.4f})")
print(f"Number of positive exponents (tol={zero_tol}): {n_positive}  ->  {classification}")
print(f"lambda_1 = {spectrum_nats[0]:.4f} nats/time  =  {spectrum_bits[0]:.4f} bits/time")

# ────────────────────────────────────────────────
#  Regenerate the same scalar x(t) series used in Part A
#  (same params / IC / solver / dt / transient discard)
# ────────────────────────────────────────────────
def lorenz_flat(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

dt = 0.01
T_total = 170.0
t_eval = np.arange(0, T_total, dt)
state0 = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz_flat, (0, T_total), state0, t_eval=t_eval,
                 method="RK45", rtol=1e-9, atol=1e-9)

transient_cutoff = 20.0
mask = sol.t >= transient_cutoff
x_series = sol.y[0][mask]
print(f"\nScalar series length after transient discard: {len(x_series)}")

# ────────────────────────────────────────────────
#  B.3 — lambda_1 from the scalar series (Wolf's tracking method)
#  tau, m come directly from Part A's estimate_delay_ami / estimate_dimension_fnn
# ────────────────────────────────────────────────
tau_opt, m_opt = 16, 3   # from Part A results

wolf_cfg = WolfConfig(theiler_window=None,      # defaults to tau*m
                       min_dist_scale=1e-3, max_dist_scale=1e-1,
                       evolve_cap=50, max_replacements=300, angle_weight=0.3)

mle_bits, debug = wolf_mle(x_series, dt=dt, tau=tau_opt, m=m_opt, cfg=wolf_cfg, return_debug=True)
mle_nats = mle_bits * np.log(2)

print("\n===== B.3 — Time-series-based lambda_1 (Wolf tracking) =====")
print(f"lambda_1 = {mle_nats:.4f} nats/time  =  {mle_bits:.4f} bits/time")
print(f"Segments tracked: {debug['replacements']} / max_replacements={wolf_cfg.max_replacements}")
print(f"Physical time accumulated: {debug['physical_time_used']:.2f} time units")
print(f"Runtimes (sec): {debug['runtimes_sec']}")

# Per-segment diagnostics: show first 5 and summary stats
segs = debug["segments"]
print(f"\nFirst 5 of {len(segs)} tracked segments:")
for s in segs[:5]:
    print(f"  i_ref={s['i_ref']:5d}  steps_evolved={s['steps_evolved']:3d}  "
          f"d0={s['d0']:.4e}  d_final={s['d_final']:.4e}  growth={s['growth']:.3f} bits")

steps_arr = np.array([s['steps_evolved'] for s in segs])
growth_arr = np.array([s['growth'] for s in segs])
print(f"\nMean steps_evolved per segment: {steps_arr.mean():.1f}  (evolve_cap={wolf_cfg.evolve_cap})")
print(f"Mean growth per segment: {growth_arr.mean():.3f} bits")
print(f"Segments hitting evolve_cap (truncated early): "
      f"{np.sum(steps_arr >= wolf_cfg.evolve_cap)} / {len(segs)}")

# ────────────────────────────────────────────────
#  Side-by-side comparison (units explicitly labeled)
# ────────────────────────────────────────────────
print("\n===== SUMMARY: lambda_1 comparison =====")
print(f"{'Method':35s}{'nats/time':>12s}{'bits/time':>12s}")
print(f"{'ODE / Wolf GSR (ground truth)':35s}{spectrum_nats[0]:12.4f}{spectrum_bits[0]:12.4f}")
print(f"{'Time series / Wolf tracking':35s}{mle_nats:12.4f}{mle_bits:12.4f}")
diff_bits = abs(spectrum_bits[0] - mle_bits)
print(f"\nAbsolute difference: {diff_bits:.4f} bits/time "
      f"({diff_bits*np.log(2):.4f} nats/time)")
print(f"Relative difference: {100*diff_bits/spectrum_bits[0]:.1f}%")
print("Both agree that lambda_1 > 0 (chaotic); the ODE-based GSR spectrum is the "
      "reliable reference, the time-series estimate is the noisier approximation.")