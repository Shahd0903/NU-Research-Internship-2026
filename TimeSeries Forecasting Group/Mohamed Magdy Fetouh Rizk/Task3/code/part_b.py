import numpy as np
from scipy.integrate import solve_ivp
from lyapunov import lyapunov_wolf_ode, AttractorODEConfig, WolfODEConfig
from lyapunov import wolf_mle, WolfConfig

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

def lorenz_jac(state, t, sigma, rho, beta):
    x, y, z = state
    return [[-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta]]

sigma, rho, beta = 10.0, 28.0, 8/3
dt = 0.01
state0 = np.array([1.0, 1.0, 1.0])

cfg_ode = AttractorODEConfig(
    ode=lorenz,
    jacobian=lorenz_jac,
    x0=state0,
    params=(sigma, rho, beta),
    dt=dt,
    transient_steps=2000,
    n_steps=50000,
    solver='RK45'
)

wolf_ode_cfg = WolfODEConfig(ortho_steps=20, log_base='2')

print("Calculating full Lyapunov spectrum from ODE...")
spectrum = lyapunov_wolf_ode(cfg_ode, wolf_cfg=wolf_ode_cfg)
print(f"Lyapunov Spectrum (ODE): {spectrum} bits/time")
print(f"Largest Lyapunov Exponent (ODE): {spectrum[0]:.4f} bits/time")

t_span = (0, 150)
t_eval = np.arange(t_span[0], t_span[1], dt)
sol = solve_ivp(lambda t, state: lorenz(state, t, sigma, rho, beta), t_span, state0, t_eval=t_eval, method='RK45')
x_series = sol.y[0][5000:]

tau_opt = 17
m_opt = 3

print(f"\nCalculating Largest Lyapunov Exponent from time series (tau={tau_opt}, m={m_opt})...")
mle_cfg = WolfConfig(
    min_dist_scale=1e-3, 
    max_dist_scale=1e-1, 
    evolve_cap=50, 
    max_replacements=200, 
    angle_weight=0.3
)

mle, debug = wolf_mle(x_series, dt=dt, tau=tau_opt, m=m_opt, cfg=mle_cfg, return_debug=True)

print(f"Largest Lyapunov Exponent (Time Series): {mle:.4f} bits/time")
print(f"Segments tracked: {debug['replacements']}")