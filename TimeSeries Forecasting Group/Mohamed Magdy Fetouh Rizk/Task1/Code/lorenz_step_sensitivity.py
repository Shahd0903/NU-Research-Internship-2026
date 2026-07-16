import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def rk4_fixed_step(fun, t_span, y0, dt, args=()):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(y0), len(t)))
    y[:, 0] = y0
    
    for i in range(1, len(t)):
        h = dt
        t_prev = t[i-1]
        y_prev = y[:, i-1]
        
        k1 = fun(t_prev, y_prev, *args)
        k2 = fun(t_prev + h/2, y_prev + (h/2)*k1, *args)
        k3 = fun(t_prev + h/2, y_prev + (h/2)*k2, *args)
        k4 = fun(t_prev + h, y_prev + h*k3, *args)
        
        y[:, i] = y_prev + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, y

def generate_sensitivity_plots():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    initial_condition = [1.0, 1.0, 1.0]
    t_span = (0, 50)
    
    dt_fine = 0.001
    dt_baseline = 0.01
    dt_coarse = 0.05
    
    t_fine, y_fine = rk4_fixed_step(
        lorenz_system, t_span, initial_condition, dt_fine, args=(sigma, rho, beta)
    )
    t_base, y_base = rk4_fixed_step(
        lorenz_system, t_span, initial_condition, dt_baseline, args=(sigma, rho, beta)
    )
    t_coarse, y_coarse = rk4_fixed_step(
        lorenz_system, t_span, initial_condition, dt_coarse, args=(sigma, rho, beta)
    )
    
    x_fine, x_base, x_coarse = y_fine[0], y_base[0], y_coarse[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_base, x_base, label=f'Baseline (dt={dt_baseline})', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(t_fine, x_fine, label=f'Fine (dt={dt_fine})', color='blue', linewidth=1.0, alpha=0.7)
    plt.plot(t_coarse, x_coarse, label=f'Coarse (dt={dt_coarse})', color='red', linewidth=1.0, linestyle='--', alpha=0.7)
    
    plt.title('Lorenz System $x(t)$ Sensitivity to Step Size', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('x(t)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('step_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(y_base[0], y_base[1], y_base[2], label=f'Baseline (dt={dt_baseline})', color='black', lw=0.8, alpha=0.6)
    ax.plot(y_fine[0], y_fine[1], y_fine[2], label=f'Fine (dt={dt_fine})', color='blue', lw=0.5, alpha=0.6)
    ax.plot(y_coarse[0], y_coarse[1], y_coarse[2], label=f'Coarse (dt={dt_coarse})', color='red', lw=0.5, linestyle='--', alpha=0.6)
    
    ax.set_title('Phase-Space Trajectories by Step Size', fontsize=14, pad=15)
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.legend(loc='upper right')
    ax.view_init(elev=25, azim=-45)
    
    plt.tight_layout()
    plt.savefig('step_phase.png', dpi=300, bbox_inches='tight')
    plt.close()

    interp_fine = interp1d(t_fine, x_fine, kind='cubic', fill_value="extrapolate")
    interp_coarse = interp1d(t_coarse, x_coarse, kind='cubic', fill_value="extrapolate")
    
    x_fine_interp = interp_fine(t_base)
    x_coarse_interp = interp_coarse(t_base)
    
    div_fine = np.abs(x_fine_interp - x_base)
    div_coarse = np.abs(x_coarse_interp - x_base)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_base, div_coarse, label='|Coarse - Baseline|', color='red', linewidth=1.2)
    plt.semilogy(t_base, div_fine, label='|Fine - Baseline|', color='blue', linewidth=1.2)
    
    plt.title('Pairwise Divergence of $x(t)$ Over Time (Log Scale)', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Absolute Error $|x_{dt}(t) - x_{baseline}(t)|$', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('step_divergence.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_sensitivity_plots()