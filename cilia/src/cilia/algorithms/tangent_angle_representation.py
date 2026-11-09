import numpy as np
from powersmooth.powersmooth import upsample_with_mask,powersmooth_general



def tangent_angle_from_xy(x, y):
    """
    Calculate the tangent angle from x and y coordinates.

    Parameters:
    x (float or np.ndarray): x coordinate(s)
    y (float or np.ndarray): y coordinate(s)

    Returns:
    float: Tangent angle in radians
    """
    psi0=np.arctan2(np.gradient(y), np.gradient(x)) if isinstance(x, np.ndarray) else np.arctan2(np.gradient(y), np.gradient(x)).item()
    psi0 = np.unwrap(psi0)  # Unwrap the angle to avoid discontinuities
    return psi0 - np.mean(psi0)  # Center the angle around zero

from cilia.DataName import TANGENT_ANGLE_TRACKS
from scipy.interpolate import interp1d

def create_tangent_angle_series(input_dict: dict):
    xy=input_dict['xy']
    res=tangent_angle_from_xy(xy[:,:,0],xy[:,:,1])
    print(res)
    raise

def create_tangent_angle_tracks(input_dict: dict, target_N: int = 100) -> dict:
    """
    Given input_dict containing raw tail coordinates under input_dict['SpermTracks']['tail'],
    resample each tail to fixed arclength spacing and compute tangent angles.

    Returns a dict with keys:
      's'     : 1D array of length target_N giving arclength positions,
      'x'     : array (N_tracked, target_N) of x-coordinates,
      'y'     : array (N_tracked, target_N) of y-coordinates,
      'theta' : array (N_tracked, target_N) of tangent angles (atan2(dy/ds, dx/ds)).
    """
    TailTracks = input_dict['SpermTracks']['tail']  # assumed shape (N_tracked, 2, n_raw)
    N_tracked = TailTracks.shape[0]

    # Pre-allocate arrays
    x_resampled = np.zeros((N_tracked, target_N))
    y_resampled = np.zeros((N_tracked, target_N))
    theta = np.zeros((N_tracked, target_N))

    for i in range(N_tracked):
        x_raw = np.array(TailTracks[i][:,0],dtype=np.double)
        y_raw = np.array(TailTracks[i][:,1],dtype=np.double)
        dx_raw = np.diff(x_raw)
        dy_raw = np.diff(y_raw)

        ds_raw = np.sqrt(dx_raw*dx_raw + dy_raw*dy_raw)
        s_raw = np.concatenate(([0.0], np.cumsum(ds_raw)))  # length = n_raw

        total_length = s_raw[-1]
        if total_length == 0:
            # Degenerate case: all points identical → fill with constant values
            s_target = np.linspace(0, 1, target_N)
            x_resampled[i, :] = x_raw[0]
            y_resampled[i, :] = y_raw[0]
            theta[i, :] = 0.0
            continue

        dx = total_length / (target_N - 1)
        s_up_x, x_up, mask_up_x = upsample_with_mask(s_raw, x_raw, dx=dx)
        smooth_x_up = powersmooth_general(s_up_x, x_up,
                                          {1: 0.0, 2: 1e+0, 3: 1e-0},
                                          mask_up_x)

        s_up_y, y_up, mask_up_y = upsample_with_mask(s_raw, y_raw, dx=dx)
        smooth_y_up = powersmooth_general(s_up_y, y_up,
                                          {1: 0.0, 2: 1e0, 3: 1e-0},
                                          mask_up_y)

        s_target = np.linspace(0.0, total_length, target_N)
        interp_x = interp1d(s_up_x, smooth_x_up, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(s_up_y, smooth_y_up, kind='linear', fill_value="extrapolate")
        x_eq = interp_x(s_target)
        y_eq = interp_y(s_target)

        x_resampled[i, :] = x_eq
        y_resampled[i, :] = y_eq

        dx_ds = np.gradient(x_eq, s_target)
        dy_ds = np.gradient(y_eq, s_target)
        theta[i, :] = np.arctan2(dy_ds, dx_ds)

    return {TANGENT_ANGLE_TRACKS.key:{
        's': s_target,           # common arclength positions (1D, length=target_N)
        'x': x_resampled,        # (N_tracked, target_N)
        'y': y_resampled,        # (N_tracked, target_N)
        'theta': theta           # (N_tracked, target_N)
    }}

