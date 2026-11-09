import numpy as np

import matplotlib.pyplot as plt

def compute_phase_from_protophi(protophi: np.ndarray, nharm: int = 10) -> np.ndarray:
    """
    Compute a proper phase from a given protophase using harmonic correction.

    Parameters:
    - protophi (np.ndarray): Real-valued 1D array representing protophase angles (in radians).
    - nharm (int): Number of harmonics to use in correction (default: 10).

    Returns:
    - np.ndarray: corrected phase of the same shape as protophi. NOT UNWRAPED
    """
    if not isinstance(protophi, np.ndarray) or protophi.ndim != 1:
        raise ValueError("protophi must be a 1D NumPy array")
    if not np.issubdtype(protophi.dtype, np.floating):
        raise ValueError("protophi must be a real-valued array")

    k = np.arange(1, nharm + 1)
    exp_kphi = np.exp(-1j * np.outer(k, protophi))  # shape: (nharm, len(protophi))
    Sn_pos = np.mean(exp_kphi, axis=1)  # shape: (nharm,)

    phi = protophi.astype(np.complex128)
    correction_terms = Sn_pos[:, None] * (np.exp(1j * np.outer(k, protophi)) - 1) / (1j * k[:, None])
    phi += 2 * np.sum(correction_terms, axis=0)

    return np.real(phi)#np.unwrap(np.real(phi))



def compute_protophase_leadingProjection(projection: np.ndarray) -> np.ndarray:
    """
    Placeholder function for computing protophase from leading projection.
    This function is intended to compute the protophase from leading projection data.
    Currently, it does not implement any functionality and serves as a placeholder.
    """
    
    if not isinstance(projection, np.ndarray) or projection.ndim != 2:
        raise ValueError("projection must be a 2D NumPy array")

    # plt.plot(projection[:, 0], projection[:, 1], label='Leading Projection')
    # plt.xlabel('Leading Projection X')
    # plt.ylabel('Leading Projection Y')
    # plt.title('Leading Projection in 2D')
    # plt.legend()
    # plt.show()

    angle = np.arctan2(projection[:, 1], projection[:, 0]) 
    protophase = np.unwrap(angle) 
    return protophase

from scipy.signal import hilbert

def compute_protophase_from_Hilbert(y: np.ndarray) -> np.ndarray:
    """
    Compute protophase from a Hilbert transform of a signal.

    Parameters:
    - y (np.ndarray): 1D array representing the signal.

    Returns:
    - np.ndarray: Protophase of the signal.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y must be a 1D NumPy array")
    
    hilbert_transform = hilbert(y)
    protophase = np.unwrap(np.angle(hilbert_transform))  # Compute the phase and unwrap it
    if protophase.ndim != 1:
        raise ValueError("The result of the Hilbert transform must be a 1D array")
    if protophase.shape[0] != y.shape[0]:
        raise ValueError("The shape of the protophase must match the input signal shape")
    if not np.issubdtype(protophase.dtype, np.floating):
        raise ValueError("protophase must be a real-valued array")
    
    return protophase

from cilia.DataName import PHASE_SERIES, PHASE_TRACKS

def phase_from_tangent_angle_pca_series(input_dict: dict) -> dict:
    """
    Compute the phase series from a tangent angle PCA series.

    Parameters:
    - input_dict (dict): Dictionary containing the tangent angle PCA series data.

    Returns:
    - dict: Dictionary containing the computed phase series.
    """
    if not isinstance(input_dict, dict):
        raise ValueError("input_dict must be a dictionary")

    leading_modes_projected = input_dict.get('leading_modes_projected')
    if leading_modes_projected is None or not isinstance(leading_modes_projected, np.ndarray):
        raise ValueError("Missing or invalid 'leading_modes_projected' in input_dict")
    
    protophase = compute_protophase_leadingProjection(leading_modes_projected)
    
    phase = compute_phase_from_protophi(protophase)
    if phase[-1]-phase[0] < 0:
        phase = -phase

    # plt.plot(phase, label='phase')
    # plt.xlabel('Time (s)')
    # plt.ylabel('phase (radians)')

    # plt.legend()
    # plt.show()
    
    return {PHASE_SERIES.key:{PHASE_SERIES.key: phase}}

def phase_from_tangent_angle_pca_tracks(pca_dict: dict) -> dict:
    """
    Compute the phase tracks from tangent angle PCA data and related sperm track information.

    Parameters:
    - pca_dict (dict): Dictionary containing the tangent angle PCA tracks data.
                       Must contain key 'leading_modes_projected' with a 2D NumPy array.
    Returns:
    - dict: Dictionary containing the computed phase tracks (currently returns nothing; development state).
    """
    if not isinstance(pca_dict, dict):
        raise ValueError("Both pca_dict and guasto_dict must be dictionaries")

    print(pca_dict.keys())
    print(pca_dict['leading_modes_projected'].shape)
    # print(guasto_dict.keys())
    # print(guasto_dict['SpermTracks']['run_idx'].shape)

    leading_modes_projected = pca_dict.get('leading_modes_projected')
    if leading_modes_projected is None or not isinstance(leading_modes_projected, np.ndarray):
        raise ValueError("Missing or invalid 'leading_modes_projected' in input_dict")
    
    protophase = compute_protophase_leadingProjection(leading_modes_projected)
    
    phase = compute_phase_from_protophi(protophase) #todo: correct?
    if phase[-1]-phase[0] < 0:
        phase = -phase

    # if protophase[-1]-protophase[0] < 0:
    #     protophase = -protophase
    # plt.plot(protophase, label='protophase')
    # plt.plot(phase, label='phase')
    # plt.xlabel('Time (s)')
    # plt.ylabel('phase (radians)')

    # plt.legend()
    # plt.show()
    
    return {PHASE_TRACKS.key:{PHASE_TRACKS.key: phase}}