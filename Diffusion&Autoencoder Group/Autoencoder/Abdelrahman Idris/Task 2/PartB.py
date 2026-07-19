import numpy as np

from lyapunov import (
    lyapunov_wolf_ode, wolf_mle,
    AttractorODEConfig, WolfODEConfig, WolfConfig
)

# --- 1. Lorenz system + Jacobian ---

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

def lorenz_ode(state, t, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def lorenz_jacobian(state, t, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    return np.array([
        [-sigma,      sigma,   0.0],
        [rho - z,     -1.0,   -x  ],
        [y,            x,     -beta]
    ])

# --- 2. Part B.2 - Full spectrum via Wolf's GSR (ODE method) ---

attractor_cfg = AttractorODEConfig(
    ode=lorenz_ode,
    jacobian=lorenz_jacobian,
    x0=np.array([1.0, 1.0, 1.0]),
    params=(SIGMA, RHO, BETA),
    dt=0.01,
    transient_steps=5000,
    n_steps=100000,
    solver='RK45',
    solver_options={'rtol': 1e-9, 'atol': 1e-9}
)

wolf_ode_cfg = WolfODEConfig(ortho_steps=50, log_base='e')

print("Running Wolf GSR ODE method (this integrates the variational equations)...")
spectrum = lyapunov_wolf_ode(attractor_cfg, wolf_ode_cfg)
spectrum_sorted = np.sort(spectrum)[::-1]

print("\n--- Part B.2: Lyapunov spectrum (ODE method, Wolf GSR) ---")
print(f"Raw output:    {spectrum}")
print(f"Sorted (desc): {spectrum_sorted}  [units: nats / time unit]")
print(f"lambda_1 (largest) = {spectrum_sorted[0]:.4f} nats/time")
print(f"Sum of spectrum     = {np.sum(spectrum_sorted):.4f}  "
      f"(should be close to -(sigma+1+beta) = {-(SIGMA+1+BETA):.4f} for Lorenz)")

# --- 3. Part B.3 - LLE from scalar time series via Wolf tracking ---

data = np.load('lorenz_data.npz')
x_series = data['x_series']
dt = float(data['dt'])
tau = int(data['tau'])
m = int(data['m'])

print(f"\nUsing tau={tau}, m={m} from Part A on a series of length {len(x_series)}")

wolf_ts_cfg = WolfConfig(
    theiler_window=tau * m,
    min_dist_scale=1e-3,
    max_dist_scale=1e-1,
    evolve_cap=50,
    max_replacements=200,
    angle_weight=0.3
)

mle_bits, debug = wolf_mle(x_series, dt=dt, tau=tau, m=m, cfg=wolf_ts_cfg, return_debug=True)
mle_nats = mle_bits * np.log(2)

print("\n--- Part B.3: LLE from time series (Wolf tracking) ---")
print(f"lambda_1 = {mle_bits:.4f} bits/time unit")
print(f"lambda_1 = {mle_nats:.4f} nats/time unit  (converted, for comparison with B.2)")
print(f"Replacements (tracked segments) used: {debug['replacements']}")
print(f"Physical time accumulated: {debug['physical_time_used']:.3f}")

print("\n--- Part B Summary: lambda_1 comparison ---")
print(f"ODE method (Wolf GSR):        lambda_1 = {spectrum_sorted[0]:.4f} nats/time")
print(f"Time-series method (Wolf MLE): lambda_1 = {mle_nats:.4f} nats/time  "
      f"(originally {mle_bits:.4f} bits/time)")
print(f"Absolute difference: {abs(spectrum_sorted[0] - mle_nats):.4f} nats/time")

if debug['replacements'] < 30:
    print("\nWARNING: low replacement count - MLE estimate may be noisy.")

np.savez('lyapunov_results.npz', spectrum=spectrum_sorted, mle_nats=mle_nats, mle_bits=mle_bits)