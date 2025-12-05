import os
import numpy as np
from pymatreader import read_mat
from pathlib import Path
import pandas as pd

from utils_rdf.config import SHARMA_DATA_DIR, N_SLIDING_WINDOWS_VARIANCE, MIN_LENGTH_AFTER_TRIM, MIN_LENGTH_Q,\
REMOVE_HILBERT_EXPERIMENT,N_HARMONICS_GLOBAL_PHASE_EXPERIMENT, OUTPUT_DATA_SHARME_DIR
from utils_rdf.flagella_algorythms import  center_data, get_quantities, get_global_phase_amplitude, get_local_phase_amplitude

def filter_first_column_by_peak_psd(centered, dt, multiplier=3.5):
    """
    Low-pass filter centered[:, 0] based on PSD peak.
    
    Steps:
    1. FFT of the signal
    2. PSD estimate
    3. Detect dominant frequency peak
    4. Low-pass filter at multiplier * peak frequency
    5. iFFT back to time domain
    
    Parameters
    ----------
    centered : ndarray
        Array of shape (T, S). Only column 0 is filtered.
    dt : float
        Sampling interval in seconds.
    multiplier : float
        Cutoff = multiplier * peak frequency.
    
    Returns
    -------
    x_filt : ndarray
        Filtered version of centered[:, 0] with shape (T,).
    """
    x = centered[:, 0]
    T = len(x)

    # FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(T, d=dt)

    # PSD
    PSD = np.abs(X)**2

    # Peak frequency (ignore DC)
    if len(PSD) > 1:
        peak_idx = np.argmax(PSD[1:]) + 1
    else:
        return x.copy()  # degenerate case

    f_peak = freqs[peak_idx]
    f_cut = multiplier * f_peak

    # Low-pass mask
    mask = freqs <= f_cut

    # Apply mask
    X_filt = X * mask

    # Inverse FFT
    x_filt = np.fft.irfft(X_filt, n=T)

    return x_filt

def determineQuantities(tangent_angles, dt_in_second, data_name, do_phi_omega_a=True):
    mean_shape, centered = center_data(tangent_angles)
    # --- compute quantities for different representations ---
    res  = get_quantities(centered, dt_in_second,N_HARMONICS_GLOBAL_PHASE_EXPERIMENT, REMOVE_HILBERT_EXPERIMENT, Q_adaptive=True)
    
    if do_phi_omega_a:
        x0_filt = filter_first_column_by_peak_psd(centered, dt_in_second)
        centered_minus = centered - x0_filt[:, None]

        phi_global, omega_global ,amp_global = get_global_phase_amplitude(centered_minus, nharm=N_HARMONICS_GLOBAL_PHASE_EXPERIMENT)
        global_phi_omega_a=np.vstack((phi_global, omega_global ,amp_global))
        
        # --- local time dependent phase and amplitude
        phi_local, omega_local ,amp_local = get_local_phase_amplitude(centered_minus,nharm=N_HARMONICS_GLOBAL_PHASE_EXPERIMENT, ds=1, dt=3)
        local_phi_omega_a = np.stack((phi_local, omega_local, amp_local), axis=0)


        out_folder = Path(OUTPUT_DATA_SHARME_DIR) / data_name
        out_folder.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_folder / "phi_omega_a.npz",
            global_phi_omega_a=global_phi_omega_a,
            local_phi_omega_a=local_phi_omega_a,
            centered=centered,
        )

    return res

def _sliding_var_mean_across_axis1(X, n):
    """
    X: (T, S) array; variance is computed in sliding windows along axis=0 (time),
       per column, then averaged across columns -> (T-n+1,)
    Uses cumulative sums (O(T*S)) and is fast.
    """
    T, S = X.shape
    if n < 1 or n > T:
        raise ValueError(f"Invalid window size n={n} for T={T}")
    # cumulative sums for sum(x) and sum(x^2)
    csum = np.cumsum(X, axis=0)
    csum2 = np.cumsum(X*X, axis=0)
    # pad with a zero row to make window sums easy
    csum = np.vstack([np.zeros((1, S), dtype=X.dtype), csum])
    csum2 = np.vstack([np.zeros((1, S), dtype=X.dtype), csum2])
    win_sum  = csum[n:, :]  - csum[:-n, :]
    win_sum2 = csum2[n:, :] - csum2[:-n, :]
    var_cols = win_sum2/n - (win_sum/n)**2          # shape (T-n+1, S), ddof=0
    return var_cols.mean(axis=1)                    # shape (T-n+1,)

def _windows_to_time_segments(good_win, n, T):
    """
    Convert a boolean mask over windows (length T-n+1) into contiguous good
    time segments over [0..T-1] by unioning the covered windows.
    Returns list of (start, end) inclusive indices.
    """
    W = good_win.shape[0]
    if W != T - n + 1:
        raise ValueError("good_win length mismatch")
    # difference array trick to mark coverage of all good windows
    diff = np.zeros(T + 1, dtype=np.int32)
    idx = np.flatnonzero(good_win)
    for i in idx:
        diff[i] += 1
        diff[i + n] -= 1
    covered = np.cumsum(diff[:-1]) > 0  # length T, True where any good window covers time t

    # extract contiguous True runs
    segments = []
    in_seg = False
    start = None
    for t, ok in enumerate(covered):
        if ok and not in_seg:
            start = t
            in_seg = True
        elif not ok and in_seg:
            segments.append((start, t - 1))
            in_seg = False
    if in_seg:
        segments.append((start, T - 1))
    return segments

def _trim_segments(segments, T, n):
    """
    Trim n//2 samples from each segment's start and end for safety,
    EXCEPT when the segment touches the overall start (0) or end (T-1).
    Drops segments that become empty after trimming.
    """
    trim = n // 2
    trimmed = []
    for s, e in segments:
        s2 = s if s == 0 else s + trim
        e2 = e if e == T - 1 else e - trim
        if s2 <= e2:
            trimmed.append((s2, e2))
    return trimmed

def get_segments(X, T, N_STUCKED=2):
    # Detect stuck frames (duplicate frames)
    eps = 1e-5
    dX = np.max(np.abs(X[1:] - X[:-1]), axis=1)   # shape (T-1,)
    stuck_mask = dX < eps                         # True where frame t & t+1 are stuck

    # Identify contiguous stuck-frame regions
    stuck_segments = []
    in_seg = False
    start = None
    for t, st in enumerate(stuck_mask):
        if st and not in_seg:
            start = t
            in_seg = True
        elif not st and in_seg:
            stuck_segments.append((start, t))  # region from start..t
            in_seg = False
    if in_seg:
        stuck_segments.append((start, len(stuck_mask)))

   
    #  keep single stuck frames, remove only longer stuck segments
    
    remove_mask = np.zeros(T, dtype=bool)

    for s, e in stuck_segments:
        seg_len = e - s + 1
        if seg_len > N_STUCKED:
            # remove all frames involved in the stuck region
            # frames s..e+1 must be removed (because dX relates t <-> t+1)
            remove_mask[s:e+1] = True
        # else: keep single stuck frames, do not remove

    # Indices of "good" non-removed frames
    good_idx = np.flatnonzero(~remove_mask)

    if len(good_idx) < N_SLIDING_WINDOWS_VARIANCE:
        print('No good frames after removing multi-frame stuck segments')
        return []

    # Split good_idx into contiguous time blocks
    nonstuck_segments = []
    start = good_idx[0]
    prev = good_idx[0]
    for idx in good_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        nonstuck_segments.append((start, prev))
        start = idx
        prev = idx
    nonstuck_segments.append((start, prev))
    
    # Your variance-based segmentation (unchanged)

    kept_segments = []

    for seg_s, seg_e in nonstuck_segments:
        seg_len = seg_e - seg_s + 1
        if seg_len < N_SLIDING_WINDOWS_VARIANCE:
            continue

        Xseg = X[seg_s:seg_e+1]

        result = _sliding_var_mean_across_axis1(Xseg, N_SLIDING_WINDOWS_VARIANCE)
        thr = 0.1 * np.mean(result)
        good_win = result >= thr

        seg_inner = _windows_to_time_segments(
            good_win, N_SLIDING_WINDOWS_VARIANCE, seg_len
        )
        seg_inner = _trim_segments(
            seg_inner, seg_len, N_SLIDING_WINDOWS_VARIANCE
        )

        for s_local, e_local in seg_inner:
            s_global = seg_s + s_local
            e_global = seg_s + e_local
            if (e_global - s_global + 1) >= MIN_LENGTH_AFTER_TRIM:
                kept_segments.append((s_global, e_global))

    return kept_segments

def main():
    file_list = ['WT_KCl_master.mat']

    results = []

    for mat_file in file_list:
        data = read_mat(os.path.join(SHARMA_DATA_DIR, mat_file))
        master = data['Master']
        genotype_from_name = mat_file.split('_')[0]

        for i in range(len(master)):
            for j in range(len(master[i])):
                entry = master[i][j]
                if not isinstance(entry, dict):
                    continue

                file_tag = entry.get('File', np.nan)
                tangent_angles = entry['tangent_angle_psi_in_rad']
                dt_in_second   = entry['dt_in_second']
                ds_in_micron   = entry['ds_in_micron']
                
                ATP            = entry.get('ATP_in_uM', np.nan)
                KCL            = entry.get('KCl_in_mM', np.nan)
                if ATP !=750:
                    continue
                T, S = tangent_angles.shape
                if T < N_SLIDING_WINDOWS_VARIANCE:
                    continue

                # --- compute mean-centered X ---
                X = tangent_angles - np.mean(tangent_angles, axis=1)[:, None] \
                                    - np.mean(tangent_angles, axis=0)[None, :]

                kept_segments = get_segments(X,T)
        
                # Final check
                if not kept_segments:
                    print('WARNING:', KCL)
                    continue

                # --- collect weighted results dynamically ---
                seg_results = None
                seg_weights = None

                for i_segment,(s, e) in enumerate(kept_segments):
                    seg = tangent_angles[s:e+1, :]
                    weight = e - s + 1  # segment length

                    quantities = determineQuantities(seg, dt_in_second, f'{file_tag}_{i_segment}')

                    # initialize dictionaries dynamically based on first result
                    if seg_results is None:
                        seg_results = {k: [] for k in quantities.keys()}
                        seg_weights = {k: [] for k in quantities.keys()}

                    for key, value in quantities.items():
                        # handle 'Q' values separately if too short
                        if key.endswith('Q') and (e - s + 1) < MIN_LENGTH_Q:
                            continue
                        seg_results[key].append(value)
                        seg_weights[key].append(weight)

                # skip if no valid results
                if not seg_results or all(len(v) == 0 for v in seg_results.values()):
                    continue

                # --- weighted average ---
                averaged = {}
                for k, values in seg_results.items():
                    if len(values) == 0:
                        averaged[k] = np.nan
                    else:
                        w = np.array(seg_weights[k])
                        v = np.array(values)
                        averaged[k] = float(np.average(v, weights=w))


                # calculate L
                L = ds_in_micron * tangent_angles.shape[1]

                
                sexp     = entry['sexp']
                A_given  = entry['A']
                Q_given  = entry['Q']
                f0_in_Hz = entry['f0_in_Hz']

                results.append({
                    **averaged,
                    'ATP': ATP,
                    'genotype': genotype_from_name,
                    'File': file_tag,
                    'L': L,
                    'KCL': KCL,
                    'sexp': sexp,
                    'A_given': A_given,
                    'Q_given': Q_given,
                    'f0_in_Hz': f0_in_Hz
                })
                
    df = pd.DataFrame(results)
    out_csv = 'all_sharma_results.csv'
    Path(OUTPUT_DATA_SHARME_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATA_SHARME_DIR / out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    main()
    
