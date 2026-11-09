import numpy as np
from cilia.DataName import TANGENT_ANGLE_PCA_SERIES, TANGENT_ANGLE_PCA_TRACKS
import matplotlib.pyplot as plt

def leadingModes(val: np.ndarray, n_modes: int = 10)-> dict:
    """
    Perform leading mode decomposition on a time-series dataset.

    Parameters:
    - val (np.ndarray): Array of shape (n_t, n_s) where n_t is number of time steps,
                        and n_s is the number of spatial/feature components.
    - n_modes (int): Number of leading eigenmodes to compute (default: 10)

    Returns:
    - dict with:
        - 'projected': val projected onto leading eigenvectors (shape: n_t x n_modes)
        - 'eigenvectors': leading eigenvectors (shape: n_s x n_modes)
        - 'eigenvalues': corresponding eigenvalues (length n_modes)
    """
    if val.ndim != 2:
        raise ValueError("Input must be a 2D array (n_t, n_s)")
    n_t, n_s = val.shape
    if n_modes <= 0 or n_modes > n_s:
        raise ValueError(f"n_modes must be in range (0, {n_s}]")
    
    cov = val.T @ val/ (n_t - 1)  # shape (n_s, n_s)
    eigvals, eigvecs = np.linalg.eigh(cov)  # returns in ascending order
    eigvals /= np.sum(eigvals)
    # Sort descending
    idx = np.argsort(eigvals)[::-1][:n_modes]
    leading_vals = eigvals[idx]
    leading_vecs = eigvecs[:, idx]  # shape (n_s, n_modes)

    projected = val @ leading_vecs  # shape (n_t, n_modes)

    return {
        "projected": projected,
        "eigenvectors": leading_vecs,
        "eigenvalues": leading_vals
    }

def center_data(val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Centers data by subtracting the mean across time axis.

    Parameters:
    - val (np.ndarray): Data of shape (n_t, n_s)

    Returns:
    - tuple[np.ndarray, np.ndarray]:
        - mean: Mean vector of shape (n_s,)
        - centered: Mean-centered data of shape (n_t, n_s)
    """
    if val.ndim != 2:
        raise ValueError("Input must be a 2D array (n_t, n_s)")

    mean_per_time = np.mean(val, axis=1)
    val = val - mean_per_time[:, np.newaxis]  # Center each time step
    
    mean = np.mean(val, axis=0)
    centered = val - mean

    return mean, centered


def process_tangent_angle_series(input_dict: dict) -> dict:
    """
    Process a dictionary containing tangent angle series data.
    This function extracts the tangent angles, times, and arclength,
    and applies leading mode decomposition and explained variance calculations.
    Parameters:
    - input_dict (dict): Dictionary containing 'tangent_angles', 'times', and 'arclength'.
    Returns:
    - dict: Processed data with leading modes and explained variance.
    """
    if not all(key in input_dict for key in ['tangent_angles', 'times', 'arclength']):
        raise ValueError("Input dictionary must contain 'tangent_angles', 'times', and 'arclength' keys.")
    
    print(input_dict.keys())
    tangent_angles=input_dict['tangent_angles']
    times = input_dict['times']
    arclength = input_dict['arclength']
    if tangent_angles.ndim != 2:
        raise ValueError("tangent_angles must be a 2D array (n_t, n_s)")
    n_t, n_s = tangent_angles.shape
    if times.ndim != 1 or arclength.ndim != 1:
        raise ValueError("times and arclength must be 1D arrays")
    if times.shape[0] != n_t or arclength.shape[0] != n_s:
        raise ValueError("Shape mismatch: times must match n_t and arclength must match n_s")

    
    # Center the data
    mean_shape, centered = center_data(tangent_angles)

    # Perform leading mode decomposition
    leading_modes_result = leadingModes(centered)


    
    res={TANGENT_ANGLE_PCA_SERIES.key: {
        "tangent_angles": tangent_angles,
        "times": times,
        "arclength": arclength,
        "mean_shape": mean_shape,
        "leading_modes_projected": leading_modes_result['projected'],
        "leading_modes_eigenvectors": leading_modes_result['eigenvectors'],
        "leading_modes_eigenvalues": leading_modes_result['eigenvalues']
    }}
    return res


def process_tangent_angle_tracks(input_dict: dict) -> dict:
    """
    Process a dictionary containing tangent angle track data.
    This function extracts the tangent angles, and arclength,
    and applies leading mode decomposition and explained variance calculations.
    Parameters:
    - input_dict (dict): Dictionary containing 'theta' and 's'.
    Returns:
    - dict: Processed data with leading modes and explained variance.
    """
    if not all(key in input_dict for key in ['theta', 's']):
        raise ValueError("Input dictionary must contain 'theta', and 's' keys.")
    
    print(input_dict.keys())
    theta=input_dict['theta']

    s = input_dict['s']
    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array (n_t, n_s)")
    
    # Center the data
    mean_shape, centered = center_data(theta)

    # Perform leading mode decomposition
    leading_modes_result = leadingModes(centered)
     
    res={TANGENT_ANGLE_PCA_TRACKS.key: {
        "tangent_angles": theta,
        "arclength": s,
        "mean_shape": mean_shape,
        "leading_modes_projected": leading_modes_result['projected'],
        "leading_modes_eigenvectors": leading_modes_result['eigenvectors'],
        "leading_modes_eigenvalues": leading_modes_result['eigenvalues']
    }}
    return res