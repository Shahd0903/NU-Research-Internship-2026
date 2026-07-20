"""
A script containing core Lyapunov exponent estimation functions (Wolf's ODE
and Wolf's time-series methods).
"""

import time
import warnings
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from sklearn.neighbors import NearestNeighbors

# --- 1. Helper Functions (Dependencies) ---

def reconstruct_matrix(x: np.ndarray, tau: int, d: int) -> np.ndarray:
    """
    Creates the time-delay embedding matrix using highly efficient numpy striding.
    This is an O(1) memory operation instead of copying arrays.

    Args:
        x (np.ndarray): The 1D time series.
        tau (int): The time delay.
        d (int): The embedding dimension.

    Returns:
        np.ndarray: The reconstructed phase space matrix of shape (M, d).
    """
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError("Time series too short for the requested dimension and delay.")
    
    # Efficient rolling window utilizing numpy's stride tricks
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# --- 2. Wolf's Method for ODEs (Direct Method) ---

@dataclass
class AttractorODEConfig:
    """
    Configuration for an ODE system to be used with the Wolf ODE solver.
    This provides all necessary functions and parameters for the integration.
    """
    ode: callable
    jacobian: callable
    x0: np.ndarray
    params: tuple
    dt: float
    transient_steps: int
    n_steps: int
    solver: str = 'RK45'
    solver_options: dict = field(default_factory=dict)

@dataclass
class WolfODEConfig:
    """
    Configuration for Wolf's Gram-Schmidt ODE algorithm.
    """
    ortho_steps: int = 50      # How many integration steps between Gram-Schmidt reorthogonalizations
    log_base: str = 'e'        # '2' for bits/sec (as in Wolf's paper), 'e' for nats/sec

WOLF_ODE_CFG = WolfODEConfig()

def lyapunov_wolf_ode(attractor_cfg: AttractorODEConfig, wolf_cfg: WolfODEConfig = WOLF_ODE_CFG):
    """
    Calculates the full Lyapunov spectrum using Wolf's Gram-Schmidt Reorthogonalization (GSR)
    method for systems of Ordinary Differential Equations (Section 3 of Wolf et al. 1985).

    Args:
        attractor_cfg (AttractorODEConfig): Configuration object containing the ODE system's
                                            dynamics, jacobian, and integration parameters.
        wolf_cfg (WolfODEConfig, optional): Configuration for the Wolf algorithm itself.
                                            Defaults to WOLF_ODE_CFG.

    Returns:
        np.ndarray: An array containing the full Lyapunov spectrum.
    """
    x = attractor_cfg.x0.copy()
    solver = getattr(attractor_cfg, 'solver', None)
    solver_options = getattr(attractor_cfg, 'solver_options', {})
    dt = attractor_cfg.dt
    p = attractor_cfg.params
    n = len(x)
    
    # 1. Run transient steps to settle onto the attractor
    t = 0.0
    if solver:
        t_span_transient = [0, attractor_cfg.transient_steps * dt]
        sol = solve_ivp(lambda t, y: attractor_cfg.ode(y, t, *p), t_span_transient, x, method=solver, dense_output=False, **solver_options)
        x = sol.y[:, -1]
        t = sol.t[-1]
    else: # Manual RK4 integration if no solver is specified
        for _ in range(attractor_cfg.transient_steps):
            k1 = attractor_cfg.ode(x, t, *p)
            k2 = attractor_cfg.ode(x + k1*dt/2, t + dt/2, *p)
            k3 = attractor_cfg.ode(x + k2*dt/2, t + dt/2, *p)
            k4 = attractor_cfg.ode(x + k3*dt, t + dt, *p)
            x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt

    # Initialize a set of n orthogonal vectors (Identity matrix)
    V = np.eye(n)
    lyap_sum = np.zeros(n)
    
    log_conv = 1.0 if wolf_cfg.log_base == 'e' else 1.0 / np.log(2.0)
    total_ortho_passes = 0
    
    if solver:
        num_passes = attractor_cfg.n_steps // wolf_cfg.ortho_steps

        def combined_system(t, y_combined):
            state = y_combined[:n]
            vectors = y_combined[n:].reshape((n, n))
            d_state = attractor_cfg.ode(state, t, *p)
            J = attractor_cfg.jacobian(state, t, *p)
            d_vectors = (J @ vectors).flatten()
            return np.concatenate((d_state, d_vectors))

        for _ in range(num_passes):
            y0_combined = np.concatenate((x, V.flatten()))
            t_start, t_end = t, t + wolf_cfg.ortho_steps * dt
            
            sol = solve_ivp(combined_system, [t_start, t_end], y0_combined, method=solver, dense_output=False, **solver_options)
            
            if sol.status != 0:
                warnings.warn(f"solve_ivp failed with status {sol.status}: {sol.message}. Stopping early.")
                break

            final_state = sol.y[:, -1]
            x = final_state[:n]
            V_evolved = final_state[n:].reshape((n, n))
            t = sol.t[-1]

            # --- Gram-Schmidt Reorthogonalization Step ---
            Q, R = np.linalg.qr(V_evolved)
            diag_R = np.abs(np.diag(R))
            diag_R[diag_R == 0] = 1e-15 # Avoid log(0)
            lyap_sum += np.log(diag_R) * log_conv
            V = Q
            total_ortho_passes += 1
    else: # Manual RK4 integration for state and variational equations
        steps_since_ortho = 0
        for _ in range(attractor_cfg.n_steps):
            k1_x = attractor_cfg.ode(x, t, *p)
            k1_V = attractor_cfg.jacobian(x, t, *p) @ V
            x_k2 = x + k1_x * dt / 2
            k2_x = attractor_cfg.ode(x_k2, t + dt/2, *p)
            k2_V = attractor_cfg.jacobian(x_k2, t + dt/2, *p) @ (V + k1_V * dt / 2)
            x_k3 = x + k2_x * dt / 2
            k3_x = attractor_cfg.ode(x_k3, t + dt/2, *p)
            k3_V = attractor_cfg.jacobian(x_k3, t + dt/2, *p) @ (V + k2_V * dt / 2)
            x_k4 = x + k3_x * dt
            k4_x = attractor_cfg.ode(x_k4, t + dt, *p)
            k4_V = attractor_cfg.jacobian(x_k4, t + dt, *p) @ (V + k3_V * dt)
            x = x + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            V = V + (dt/6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)
            t += dt
            steps_since_ortho += 1
            
            if steps_since_ortho >= wolf_cfg.ortho_steps:
                Q, R = np.linalg.qr(V)
                diag_R = np.abs(np.diag(R))
                diag_R[diag_R == 0] = 1e-15
                lyap_sum += np.log(diag_R) * log_conv
                V = Q
                steps_since_ortho = 0
                total_ortho_passes += 1
            
    total_time = (total_ortho_passes * wolf_cfg.ortho_steps) * attractor_cfg.dt
    lyapunov_exponents = lyap_sum / total_time if total_time > 0 else np.full(n, np.nan)
    
    return lyapunov_exponents


# --- 3. Wolf's Method for Time Series (Approximation) ---

@dataclass
class WolfConfig:
    """Configuration for Wolf's time-series algorithm."""
    theiler_window: int = None      # If None, defaults to tau * m
    min_dist_scale: float = 1e-3    # Minimum distance scale for valid neighbors
    max_dist_scale: float = 1e-1    # Maximum distance scale for valid neighbors
    evolve_cap: int = 50            # Maximum steps to evolve before forcing a replacement
    max_replacements: int = 200     # Cap on total replacements/tracking steps
    angle_weight: float = 0.3       # Weight (0 to 1) for preserving vector angle during replacement
    eps: float = 1e-12              # Small epsilon to prevent division by zero

MLE_CFG_WOLF = WolfConfig()

def wolf_mle(x: np.ndarray, dt: float, tau: int, m: int, cfg: WolfConfig = MLE_CFG_WOLF, return_debug: bool = True):
    """
    Estimates the Largest Lyapunov Exponent (LLE) using Wolf's phase space tracking algorithm.
    
    Args:
        x (np.ndarray): 1D numpy array containing the scalar time series.
        dt (float): Sampling interval of the time series.
        tau (int): Time delay for phase space reconstruction.
        m (int): Embedding dimension.
        cfg (WolfConfig, optional): Configuration dataclass. Defaults to MLE_CFG_WOLF.
        return_debug (bool, optional): Whether to return internal tracking statistics.
        
    Returns:
        tuple or float: 
            - If return_debug is True: (mle, debug_dict)
            - If return_debug is False: mle (float)
    """
    total_t0 = time.perf_counter()

    rec_t0 = time.perf_counter()
    X = reconstruct_matrix(x, tau=tau, d=m)
    reconstruction_runtime_sec = time.perf_counter() - rec_t0

    M = len(X)
    theiler_window = cfg.theiler_window if cfg.theiler_window is not None else (tau * m)

    scale_t0 = time.perf_counter()
    sample_size = min(2000, M)
    pair_dists = np.linalg.norm(X[:sample_size, None, :] - X[None, :sample_size, :], axis=2)
    pair_dists = pair_dists[np.triu_indices_from(pair_dists, k=1)]
    scale = np.median(pair_dists[np.isfinite(pair_dists)]) if len(pair_dists) else 1.0
    distance_scale_runtime_sec = time.perf_counter() - scale_t0

    min_dist, max_dist = cfg.min_dist_scale * scale, cfg.max_dist_scale * scale

    nn_fit_t0 = time.perf_counter()
    nbrs = NearestNeighbors(n_neighbors=min(500, M), algorithm='kd_tree').fit(X)
    nn_fit_runtime_sec = time.perf_counter() - nn_fit_t0

    def find_valid_replacement(i_ref, old_vector=None):
        distances, indices = nbrs.kneighbors(X[i_ref].reshape(1, -1))
        distances, indices = distances[0], indices[0]
        best_j, best_penalty, best_dist = None, np.inf, None

        for d, j in zip(distances[1:], indices[1:]):
            if abs(j - i_ref) <= theiler_window or d < min_dist or d > max_dist:
                continue
            
            if old_vector is not None:
                new_vector = X[j] - X[i_ref]
                cos_theta = np.dot(old_vector, new_vector) / (np.linalg.norm(old_vector) * d + cfg.eps)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                penalty = (1 - cfg.angle_weight) * (d / max_dist) + cfg.angle_weight * (theta / np.pi)
            else:
                penalty = d 

            if penalty < best_penalty:
                best_penalty, best_j, best_dist = penalty, j, d
        return best_j, best_dist

    i_ref, log_sum, physical_time_sum, replacements = 0, 0.0, 0.0, 0
    segments, current_evolved_vector = [], None
    loop_t0 = time.perf_counter()

    while i_ref < M - cfg.evolve_cap - 1 and replacements < cfg.max_replacements:
        j_nb, d0 = find_valid_replacement(i_ref, old_vector=current_evolved_vector)

        if j_nb is None:
            i_ref += 1
            current_evolved_vector = None
            continue

        k, last_valid_k, last_dist = 1, None, None
        while i_ref + k < M and j_nb + k < M and k <= cfg.evolve_cap:
            dk = np.linalg.norm(X[i_ref + k] - X[j_nb + k])
            if dk > scale * 0.4: break
            if dk > cfg.eps: last_valid_k, last_dist = k, dk
            k += 1

        if last_valid_k is None or d0 < cfg.eps or last_dist is None:
            i_ref += 1
            current_evolved_vector = None
            continue

        dt_physical = last_valid_k * dt
        growth = np.log2(last_dist / d0)
        log_sum += growth
        physical_time_sum += dt_physical
        
        current_evolved_vector = X[j_nb + last_valid_k] - X[i_ref + last_valid_k]

        if return_debug:
            segments.append({"i_ref": i_ref, "j_nb": j_nb, "steps_evolved": last_valid_k, "d0": d0, "d_final": last_dist, "growth": growth})

        i_ref += last_valid_k
        replacements += 1

    tracking_runtime_sec = time.perf_counter() - loop_t0
    total_runtime_sec = time.perf_counter() - total_t0
    mle = log_sum / physical_time_sum if physical_time_sum > 0 else np.nan

    if not return_debug:
        return mle
        
    debug = {
        "mle_estimate": mle, "replacements": replacements, "physical_time_used": physical_time_sum,
        "segments": segments,
        "runtimes_sec": {
            "reconstruction": reconstruction_runtime_sec, "distance_scale": distance_scale_runtime_sec,
            "nn_fit": nn_fit_runtime_sec, "tracking": tracking_runtime_sec, "total": total_runtime_sec
        }
    }
    return mle, debug