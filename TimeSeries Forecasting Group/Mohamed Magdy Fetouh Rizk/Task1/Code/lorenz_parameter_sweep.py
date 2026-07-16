import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_bifurcation_diagram():
    sigma = 10.0
    beta = 8.0 / 3.0
    
    rho_start = 0.0
    rho_end = 30.0
    rho_step = 0.1
    rho_values = np.arange(rho_start, rho_end + rho_step, rho_step)
    
    initial_condition = [1.0, 1.0, 1.0]
    t_span = (0, 100)
    dt = 0.01
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    transient_time = 20.0
    transient_index = int(transient_time / dt)
    
    recorded_rho = []
    recorded_z_maxima = []
    
    for rho in rho_values:
        solution = solve_ivp(
            lorenz_system,
            t_span,
            initial_condition,
            args=(sigma, rho, beta),
            t_eval=t_eval,
            method='RK45'
        )
        
        z_trajectory = solution.y[2]
        z_steady_state = z_trajectory[transient_index:]
        
        peaks, _ = find_peaks(z_steady_state)
        local_maxima = z_steady_state[peaks]
        
        for maximum in local_maxima:
            recorded_rho.append(rho)
            recorded_z_maxima.append(maximum)
            
    plt.figure(figsize=(10, 6))
    plt.scatter(recorded_rho, recorded_z_maxima, s=1.5, c='black', alpha=0.6, marker='.')
    
    plt.title('Bifurcation Diagram of the Lorenz System', fontsize=14)
    plt.xlabel(r'Control Parameter ($\rho$)', fontsize=12)
    plt.ylabel(r'Local Maxima of $z(t)$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('lorenz_rho_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_bifurcation_diagram()