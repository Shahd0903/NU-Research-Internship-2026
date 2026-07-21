"""
A script containing core Phase Space Reconstruction (PSR) functions
for estimating time delay and embedding dimension.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

# --- 1. Helper Functions (Dependencies for both AMI and FNN) ---

def maybe_standardize(x: np.ndarray, standardize: bool) -> np.ndarray:
    """Standardizes the time series to zero mean and unit variance if requested."""
    if standardize:
        return (x - np.mean(x)) / np.std(x)
    return x

def reconstruct_matrix(x: np.ndarray, tau: int, d: int) -> np.ndarray:
    """
    Creates the time-delay embedding matrix using highly efficient numpy striding.
    O(1) memory operation instead of copying arrays.
    """
    N = len(x)
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError("Time series too short for the requested dimension and delay.")
    
    # Efficient rolling window utilizing stride tricks
    shape = (M, d)
    strides = (x.strides[0], x.strides[0] * tau)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def first_local_minimum(arr: np.ndarray) -> int:
    """
    Finds the index of the first local minimum in a 1D array.
    A local minimum is found when the derivative transitions from negative to positive.
    """
    diff = np.diff(arr)
    for i in range(len(diff) - 1):
        if diff[i] < 0 and diff[i+1] > 0:
            return i + 1
    # Fallback to absolute minimum if no local minimum exists in the window
    return int(np.argmin(arr))


# --- 2. AMI (Average Mutual Information) for Time Delay ---

@dataclass
class AMIConfig:
    """Configuration for the AMI algorithm."""
    max_lag: int = 100
    n_bins: int = 32
    criterion: str = "first_local_min"

PSR_CFG_AMI = AMIConfig()

def estimate_delay_ami(x, cfg=PSR_CFG_AMI, standardize=True):
    """
    Calculates the Average Mutual Information (AMI) for a range of time delays.
    Fully vectorized entropy calculation to bypass nested for-loops.

    Args:
        x (np.ndarray): The 1D time series.
        cfg (AMIConfig, optional): Configuration for the AMI calculation. 
                                   Defaults to PSR_CFG_AMI.
        standardize (bool, optional): Whether to standardize the time series. 
                                      Defaults to True.

    Returns:
        tuple: A tuple containing:
            - tau_opt (int): The optimal time delay.
            - lags (np.ndarray): The array of lags tested.
            - ami_vals (np.ndarray): The AMI values for each lag.
    """
    x = maybe_standardize(x, standardize)
    
    # Pre-compute bin edges once on the whole dataset to keep probability bins strictly consistent
    _, bin_edges = np.histogram(x, bins=cfg.n_bins)

    ami_vals = np.zeros(cfg.max_lag)
    lags = np.arange(1, cfg.max_lag + 1)
    eps = 1e-15

    for idx, tau in enumerate(lags):
        # Shift arrays by tau
        x1 = x[:-tau]
        x2 = x[tau:]

        # 2D Joint Histogram and 1D Marginal Histograms
        hxy, _, _ = np.histogram2d(x1, x2, bins=[bin_edges, bin_edges])
        px, _ = np.histogram(x1, bins=bin_edges)
        py, _ = np.histogram(x2, bins=bin_edges)

        # Convert frequencies to probabilities
        pxy = hxy / np.sum(hxy)
        px = px / np.sum(px)
        py = py / np.sum(py)

        # --- Vectorized Entropy Summation ---
        # Reshape px and py to broadcast into a 2D matrix representing P(x)*P(y)
        # px becomes a column vector (bins, 1), py becomes a row vector (1, bins)
        px_py = px[:, None] * py[None, :]
        
        # Create a boolean mask to isolate valid probabilities (avoiding log(0) and division by zero)
        mask = (pxy > eps) & (px_py > eps)
        
        # Apply the Shannon entropy formula only to valid elements and sum the result
        ami = np.sum(pxy[mask] * np.log(pxy[mask] / px_py[mask]))
        ami_vals[idx] = ami

    # Identify optimal tau based on the selected criterion
    if cfg.criterion == "first_local_min":
        optimal_idx = first_local_minimum(ami_vals)
    else:
        optimal_idx = int(np.argmin(ami_vals))
        
    tau_opt = int(lags[optimal_idx])
    return tau_opt, lags, ami_vals


# --- 3. FNN (False Nearest Neighbors) for Embedding Dimension ---

@dataclass
class FNNConfig:
    """Configuration for the FNN algorithm."""
    max_dim: int = 15
    R_tol: float = 10.0   # Kennel's Criterion 1
    A_tol: float = 2.0    # Kennel's Criterion 2
    threshold_percent: float = 1.0
    theiler: int = 50     # Increased to ensure we skip temporally correlated points
    distance_metric: str = "euclidean"

PSR_CFG_FNN = FNNConfig()

def estimate_dimension_fnn(x, tau, cfg=PSR_CFG_FNN, standardize=True):
    """
    Calculates the percentage of False Nearest Neighbors for dimensions 1 to max_dim.
    Fully vectorized for O(N log N) computational efficiency.

    Args:
        x (np.ndarray): The 1D time series.
        tau (int): The time delay for phase space reconstruction.
        cfg (FNNConfig, optional): Configuration for the FNN calculation. 
                                   Defaults to PSR_CFG_FNN.
        standardize (bool, optional): Whether to standardize the time series. 
                                      Defaults to True.

    Returns:
        tuple: A tuple containing:
            - optimal_m (int): The optimal embedding dimension.
            - dims (np.ndarray): The array of dimensions tested.
            - fnn_pct (np.ndarray): The FNN percentage for each dimension.
    """
    x = maybe_standardize(x, standardize)
    Ra = np.std(x)
    fnn_pct = []
    dims = np.arange(1, cfg.max_dim + 1)

    for d in dims:
        Yd = reconstruct_matrix(x, tau, d)
        M = len(Yd)

        # To find 1 valid neighbor outside the Theiler window, we must search 
        # at least 2*theiler + 2 neighbors to guarantee a hit.
        k_search = min(M, 2 * cfg.theiler + 2)
        nbrs = NearestNeighbors(n_neighbors=k_search, metric=cfg.distance_metric, algorithm='kd_tree').fit(Yd)
        dist, idx = nbrs.kneighbors(Yd)

        # Vectorized Theiler window exclusion
        i_arr = np.arange(M)[:, None]
        valid_mask = np.abs(idx - i_arr) > cfg.theiler
        
        # Find the first index in each row that falls outside the Theiler window
        first_valid_idx = np.argmax(valid_mask, axis=1)
        
        # Extract the correct neighbor indices and distances
        chosen_j = idx[np.arange(M), first_valid_idx]
        chosen_rd = dist[np.arange(M), first_valid_idx]

        # Map current indices to the d+1 dimension
        i1 = np.arange(M) + d * tau
        j1 = chosen_j + d * tau

        # Filter points that exist in dimension d but can't expand to d+1
        valid_bounds = (i1 < len(x)) & (j1 < len(x))
        valid_i1 = i1[valid_bounds]
        valid_j1 = j1[valid_bounds]
        valid_rd = chosen_rd[valid_bounds]

        # Calculate distances in the newly added dimension
        delta_new = np.abs(x[valid_i1] - x[valid_j1])
        rd1 = np.sqrt(valid_rd**2 + delta_new**2)

        # Vectorized Criterion Checks
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(valid_rd < 1e-15, np.inf, delta_new / valid_rd)

        is_false = (rel > cfg.R_tol) | ((rd1 / Ra) > cfg.A_tol)
        
        total_count = np.sum(valid_bounds)
        false_count = np.sum(is_false)

        pct = 100 * false_count / total_count if total_count > 0 else np.nan
        fnn_pct.append(pct)

    fnn_pct = np.array(fnn_pct)
    below = np.where(fnn_pct <= cfg.threshold_percent)[0]
    optimal_m = int(dims[below[0]]) if len(below) else int(cfg.max_dim)
    
    return optimal_m, dims, fnn_pct