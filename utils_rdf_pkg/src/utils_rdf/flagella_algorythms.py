import numpy as np
from scipy.fft import fft, fftfreq
# import warnings
from numpy.polynomial.legendre import legvander
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
# from sklearn.decomposition import TruncatedSVD
from scipy.signal import hilbert


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

def compute_psd(signal, dt):
    n_time = signal.shape[0]
    fft_vals = fft(signal, axis=0)
    freqs = fftfreq(n_time, d=dt)
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]

    fs=1/dt
    power = (np.abs(fft_vals[pos_mask, :]) ** 2) * (2.0 / (fs * n_time))
    if n_time % 2 == 0:
        power[0, :] /= 2
        power[-1, :] /= 2
    else:
        power[0, :] /= 2

    psd = np.mean(power, axis=1)
    return freqs_pos, psd

def get_peak_frequency(freqs, psd, min_freq=0.2):
    mask = freqs >= min_freq
    peak_idx = np.argmax(psd[mask])
    return freqs[mask][peak_idx]

def integrate_power_in_band(freqs, psd, peak_freq, band):
    mask = (freqs >= peak_freq / (1 + band)) & (freqs <= peak_freq * (1 + band))
    df = freqs[1] - freqs[0]
    return np.sum(psd[mask]) * df

def power_to_amplitude(power):
    return np.sqrt(2 * power)

def variance_based_amplitude(centered):
    return np.sqrt(np.mean(centered**2)*2)

def periodic_average_tangent_angle_series(
    tangent_angles,
    phase,
    n_periodic_t: int = 1000,
    sigma: float = 0.3,
) -> dict:
    """
    Periodic average of tangent angle series y(phase) using:
      - local first-order (local linear) regression in phase,
      - circular (von Mises) kernel,
      - adaptive bandwidth (Abramson's rule) with base bandwidth selected by CV.

    Parameters
    ----------
   
    - 'tangent_angles': ndarray (n_t, n_s)
    - 'times': ndarray (n_t,)
    - 'phase_series': ndarray (n_t,)
    n_periodic_t : int
        Number of phase points to evaluate the averaged waveform on.
    sigma : float
        Initial guess for the base bandwidth in radians (used to seed CV search).

    Returns
    -------
    dict
        With key PERIODIC_TANGENT_ANGLE_SERIES.key and value:
          - 'tangent_avg': (n_periodic_t, n_s) averaged waveform over one period
          - 'phase_uniform': (n_periodic_t,) phases in [0, 2π)
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    eps = 1e-18

    def wrap_diff(a, b):
        """Smallest signed circular difference a - b into (-π, π]."""
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def vm_weights(dphi, kappa):
        """Von Mises kernel weights (no normalization)."""
        return np.exp(kappa * np.cos(dphi))

    def local_linear_predict(phase_eval, phase_train, Y_train, sigma0,
                             adaptive=True,
                             sigma_mult_bounds=(0.3, 3.0),
                             pilot_sigma=None):
        """
        Local linear prediction at phase_eval using (phase_train, Y_train).
        - Von Mises kernel
        - Optional adaptive bandwidth via Abramson's rule
        Shapes:
          phase_eval: (n_p,)
          phase_train: (n_t_sub,)
          Y_train: (n_t_sub, n_s_sub)
        Returns:
          Y_hat: (n_p, n_s_sub)
        """
        n_p = phase_eval.shape[0]
        dphi = wrap_diff(phase_eval[:, None], phase_train[None, :])  # (n_p, n_t_sub)

        # Base bandwidth per eval point
        if adaptive:
            sig_pilot = pilot_sigma if pilot_sigma is not None else sigma0
            kappa_pilot = 1.0 / (sig_pilot**2 + eps)
            pilot_w = vm_weights(dphi, kappa_pilot)                     # (n_p, n_t_sub)
            f_hat = pilot_w.sum(axis=1) + eps                            # density proxy at eval points
            g = np.exp(np.mean(np.log(f_hat)))                           # geometric mean
            sigma_eval = sigma0 * np.sqrt(np.clip(g / f_hat,
                                                  sigma_mult_bounds[0]**2,
                                                  sigma_mult_bounds[1]**2))
        else:
            sigma_eval = np.full(n_p, sigma0)

        kappa_eval = 1.0 / (sigma_eval**2 + eps)                         # (n_p,)
        w = np.exp(kappa_eval[:, None] * np.cos(dphi))                   # (n_p, n_t_sub)

        Δ = dphi
        S0 = np.sum(w, axis=1, keepdims=True) + eps
        S1 = np.sum(w * Δ, axis=1, keepdims=True)
        S2 = np.sum(w * Δ * Δ, axis=1, keepdims=True)
        D = S0 * S2 - S1 * S1 + eps

        alpha = (S2 - Δ * S1) / D                                        # (n_p, n_t_sub)
        W_eff = w * alpha                                                # (n_p, n_t_sub)
        Y_hat = W_eff @ Y_train                                          # (n_p, n_s_sub)
        return Y_hat

    def choose_sigma0_cv(phase, Y, sigma_seed,
                         n_candidates=7,
                         scale_low=0.1, scale_high=10.0,
                         cv_max_points=300,
                         cv_folds=5,
                         cv_n_s=8,
                         random_state=0):
        """
        Pick base sigma0 by K-fold CV (predict left-out samples at their phases).
        Uses local linear + adaptive bandwidth (computed on train split) and von Mises kernel.
        To keep compute bounded, uses a subset of time points and a subset of spatial columns.
        """
        rng = np.random.default_rng(random_state)

        n_t, n_s = Y.shape
        # choose time subset for CV
        if n_t > cv_max_points:
            idx_all = rng.choice(n_t, size=cv_max_points, replace=False)
        else:
            idx_all = np.arange(n_t)

        # choose spatial subset
        if n_s > cv_n_s:
            # evenly spaced columns to be representative across arclength
            s_idx = np.linspace(0, n_s - 1, cv_n_s, dtype=int)
        else:
            s_idx = np.arange(n_s)

        phi_sub = np.mod(phase[idx_all], 2*np.pi)
        Y_sub = Y[idx_all][:, s_idx]

        # candidate sigmas (log-spaced around seed)
        if sigma_seed <= 0:
            sigma_seed = 0.3
        sigmas = np.exp(np.linspace(np.log(sigma_seed * scale_low),
                                    np.log(sigma_seed * scale_high),
                                    n_candidates))
        # avoid too tiny/too large
        sigmas = np.clip(sigmas, 0.05, np.pi)

        # build folds
        rng.shuffle(idx_all)
        folds = np.array_split(np.arange(phi_sub.shape[0]), min(cv_folds, phi_sub.shape[0]))

        best_sigma = sigmas[0]
        best_err = np.inf

        for sigma0 in sigmas:
            err_acc = 0.0
            cnt = 0
            for fold in folds:
                val_idx = fold
                train_mask = np.ones(phi_sub.shape[0], dtype=bool)
                train_mask[val_idx] = False
                phi_tr = phi_sub[train_mask]
                phi_va = phi_sub[val_idx]
                Y_tr = Y_sub[train_mask, :]
                Y_va = Y_sub[val_idx, :]

                if phi_tr.size == 0 or phi_va.size == 0:
                    continue

                Y_hat = local_linear_predict(
                    phase_eval=phi_va,
                    phase_train=phi_tr,
                    Y_train=Y_tr,
                    sigma0=sigma0,
                    adaptive=True,              # adaptive in CV, per request
                    sigma_mult_bounds=(0.3, 3.0),
                    pilot_sigma=None
                )
                # MSE on validation subset
                diff = Y_hat - Y_va
                err_acc += float(np.mean(diff * diff))
                cnt += 1

            if cnt > 0 and err_acc / cnt < best_err:
                best_err = err_acc / cnt
                best_sigma = float(sigma0)

        return best_sigma

    # ----------------------------
    # Inputs
    # ----------------------------
    Y = tangent_angles.copy()   # (n_t, n_s)
    phase = phase.copy()  # (n_t,)

    n_t, n_s = Y.shape

    # Remove global shift per time
    Y = Y - np.mean(Y, axis=1, keepdims=True)

    # phases mod 2π (keep original order)
    phase_mod = np.mod(phase, 2*np.pi)

    # ----------------------------
    # Bandwidth selection (base sigma0) by CV on a subset
    # ----------------------------
    sigma0 = choose_sigma0_cv(
        phase=phase_mod,
        Y=Y,
        sigma_seed=float(sigma),
        n_candidates=7,
        scale_low=0.6,
        scale_high=2.0,
        cv_max_points=300,
        cv_folds=5,
        cv_n_s=min(12, n_s),
        random_state=0,
    )

    # ----------------------------
    # Final evaluation on uniform phase grid with adaptive bandwidth
    # ----------------------------
    phase_uniform = np.linspace(0, 2*np.pi, n_periodic_t, endpoint=False)  # (n_p,)
    tangent_avg = local_linear_predict(
        phase_eval=phase_uniform,
        phase_train=phase_mod,
        Y_train=Y,
        sigma0=sigma0,
        adaptive=True,                      # adaptive bandwidth across phase
        sigma_mult_bounds=(0.3, 3.0),
        pilot_sigma=None
    )  # (n_p, n_s)

    return {
            "tangent_avg": tangent_avg,
            "phase_uniform": phase_uniform,
        }
    
def compute_phase_from_protophi(protophi: np.ndarray, nharm: int = 10) -> np.ndarray:
    """
    Compute a proper phase from a given protophase using harmonic correction.
    We follow the method described in Kralemann et al., Phys. Rev. E 77, 066205 (2008).
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

    return np.real(phi)


def get_phase_by_hilbert(leading_projected, nharm=20, REMOVE_HILBERT=1000):
    trajectory_H = hilbert(leading_projected)
  
    norms_H = np.absolute(trajectory_H)

    p1_H, p10_H, p50_H = np.percentile(norms_H, [1, 10, 50])

    angles_H = np.unwrap(np.angle(trajectory_H))[REMOVE_HILBERT:-REMOVE_HILBERT]
    angles_corrected_H=np.unwrap(compute_phase_from_protophi(angles_H,nharm=nharm))
    return angles_corrected_H,p1_H, p10_H, p50_H

def fit_shapeGeyer(input_dict):
    """
    Fits the Geyer shape model to the tangent‐averaged data.
    
    input_dict must contain:
      - 'tangent_avg': array‐like, shape (T, S), tangent‐averaged data
      - 'phase_uniform': array‐like, shape (T,), uniform phase sampling (one full period)
      - 'arclength': array‐like, shape (S,), spatial positions along the cilium
    Returns a dict under SHAPE_GEYER_FIT.key with:
      - 'a':       array (S,), amplitude envelope a(s)
      - 'phi':     array (S,), local phase shift φ(s) (with φ(0)=0)
      - 'a_coefs': array (6,), Legendre coefficients of a(s) on [0,L], orders 0–5
      - 'phi_coefs': array (6,), Legendre coefficients of φ(s) on [0,L], orders 0–5
      - 'wavelength': scalar, λ determined from φ₁ = −π·L/λ
    """
    # Unpack
    X = np.asarray(input_dict['tangent_avg'])      # shape (T, S)
    phase = np.asarray(input_dict['phase_uniform'])  # shape (T,)
    s   = np.linspace(0,1,X.shape[1])      # shape (S,)
    
    X = X - X[:,0][:,np.newaxis] #assume fixed point at s=0
    
    T, S = X.shape
    # First Fourier mode per s-position
    #    C = (2/T) ∑ X(t,s)·cos(phase(t)), 
    #    S_coef = (2/T) ∑ X(t,s)·sin(phase(t))
    C      = (2.0/T) * (X * np.cos(phase)[:, None]).sum(axis=0)  # shape (S,)
    S_coef = (2.0/T) * (X * np.sin(phase)[:, None]).sum(axis=0)  # shape (S,)
    # amplitude
    a = np.sqrt(C**2 + S_coef**2)
    thresh = 1e-4 * np.nanmean(a)   # threshold for bad amplitude
    bad    = a < thresh

    if bad.any():
        inds = np.arange(S)
        good = ~bad
        C[good] /= a[good]  # normalize C by amplitude
        S_coef[good] /= a[good]  # normalize S_coef by amplitude
        # define extrapolating interpolators
        C_interp = interp1d(inds[good], C[good], kind='linear', fill_value='extrapolate')
        S_interp = interp1d(inds[good], S_coef[good], kind='linear', fill_value='extrapolate')

        # apply to bad indices
        C[bad] = C_interp(inds[bad])
        S_coef[bad] = S_interp(inds[bad])

    # phase shift
    phi = np.arctan2(-S_coef, C)  # in (−π, π]
    phi = np.unwrap(phi)
    
    # Legendre‐polynomial decomposition on [0, L]
    L = s[-1] - s[0]
    # map s ∈ [s0, sN] → x ∈ [−1, 1]
    x = 2*(s - s[0]) / L - 1
    deg = 5
    V = legvander(x, deg)   # shape (S, deg+1), columns = P0..P5

    # least‐squares fit for amplitude and phase
    a_coefs, *_   = np.linalg.lstsq(V, a,   rcond=None)  # length 6
    phi_coefs, *_ = np.linalg.lstsq(V, phi, rcond=None)  # length 6

    a_recon = V[:, :deg+1] @ a_coefs[:deg+1]
    phi_recon = V[:, :deg+1] @ phi_coefs[:deg+1]

  

    X_rec = a_recon[None, :] * np.cos(phase[:, None] + phi_recon[None, :])

    # compute R^2 of the reconstruction ---
    ss_res = np.sum((X - X_rec)**2)
    ss_tot = np.sum((X - X.mean())**2)
    r2 = 1.0 - ss_res/ss_tot
 
    # enforce φ₀ = 0 exactly
    phi_coefs[0] = 0.0

    # 3) Determine wavelength λ from φ₁ = −π·L/λ
    phi1 = phi_coefs[1]
    wavelength = np.abs(np.pi * L / phi1)

    return {
            'a': a,
            'phi': phi,
            'a_legendre_coefs': a_coefs,
            'phi_legendre_coefs': phi_coefs,
            'wavelength': wavelength,
            'r2': r2
    }

def getDQ_fixed_tau(angles,fraction=0.1):
    N = len(angles)
    tau = int(fraction * N)
    omega = (angles[-1] - angles[0]) / N

    delta = angles[tau:] - angles[:-tau]
    expected = omega * tau
    var = np.mean((delta - expected) ** 2)
    D=var/2/tau
    Q=omega/2/D
    return D,Q

def getDQ_adaptive(angles):
    N = len(angles)

    # range of fractions to test
    fractions = np.logspace(max(np.log(1/N),-5), -1.5, 15)
    omega = (angles[-1] - angles[0]) / N  # mean frequency

    var_mean = []
    var_std = []
    taus = []

    for frac in fractions:
        tau = int(frac * N)
        if tau < 1 or tau >= N:
            continue
        delta = angles[tau:] - angles[:-tau]
        expected = omega * tau
        displace = delta - expected
        vals = displace**2
        var_mean.append(np.mean(vals))
        var_std.append(np.std(vals))
        taus.append(tau)

    var_mean = np.array(var_mean)
    var_std = np.array(var_std)
    taus = np.array(taus)
    fractions = taus / N

    Q_tau = omega / var_mean * taus

    Q=np.min(Q_tau)
    return omega/2/Q, Q

def get_quantities(centered,dt_in_second,nharm, remove_hilbert, Q_adaptive=True, ):
    leading_modes_result = leadingModes(centered, n_modes=3)

    n_time = centered.shape[0]
    fft_vals = fft(centered-centered[:,[0]], axis=0)
    freqs = fftfreq(n_time, d=dt_in_second)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]

    fs=1/dt_in_second
    power = (np.abs(fft_vals[pos_mask, :]) ** 2) * (2.0 / (fs * n_time))
    if n_time % 2 == 0:
        power[0, :] /= 2
        power[-1, :] /= 2
    else:
        power[0, :] /= 2

    psd = np.mean(power, axis=1)
    
    mask = freqs >= 0.2
    peak_idx = np.argmax(psd[mask])
    peak_freq_hz =  freqs[mask][peak_idx]
    
    power = integrate_power_in_band(freqs, psd, peak_freq_hz, 0.1)
    amplitude_band_10pct = power_to_amplitude(power)

    phase,p1, p10, p50=get_phase_by_hilbert(leading_modes_result["projected"][:,0],nharm=nharm, REMOVE_HILBERT=remove_hilbert)

    num_rotations_H = (phase[-1] - phase[0]) / (2 * np.pi)
    f_phase=num_rotations_H/len(phase)/dt_in_second

    if Q_adaptive:
        _,Q = getDQ_adaptive(phase)
    else:
        _,Q = getDQ_fixed_tau(phase)

    periodic_avg=periodic_average_tangent_angle_series(centered[remove_hilbert:-remove_hilbert,:], phase)
    res_geyer=fit_shapeGeyer(periodic_avg)
    
    return {
        'p50': p50,
        'p1': p1,
        'wavelength_rel': res_geyer['wavelength'],
        'Q': Q ,
        'peak_freq_hz': peak_freq_hz,
        'f_phase': f_phase,
        'amplitude_band_10pct': amplitude_band_10pct,
    }

def _extract_patches_2d(arr, dt: int, ds: int):
    """
    Return sliding-window patches centered at each valid (t, s):
      shape = (T-2*dt, S-2*ds, 2*dt+1, 2*ds+1)
    Zero-copy view via strides.
    """
    T, S = arr.shape
    t_win = 2 * dt + 1
    s_win = 2 * ds + 1
    if T <= 2 * dt or S <= 2 * ds:
        raise ValueError("gamma too small for given dt/ds")

    out_shape = (T - 2 *dt, S - 2 *ds, t_win, s_win)
    strides = (arr.strides[0], arr.strides[1], arr.strides[0], arr.strides[1])
    return as_strided(arr, shape=out_shape, strides=strides, writeable=False)

def local_phase_pca2(
    gamma: np.ndarray,
    nharm,
    dt: int = 3,
    ds: int = 3,
):
    """
    Local phase/amplitude from spatiotemporal neighborhoods via 2D PCA.
    Returns phase and amplitude.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    patches = _extract_patches_2d(gamma, dt, ds)
    Tprime, Sprime, t_win, s_win = patches.shape
    P = t_win * s_win

    phi = np.empty((Tprime, Sprime))
    amp = np.empty((Tprime, Sprime))

    for s in tqdm(range(Sprime), desc="PCA2 columns", leave=False):
        X = patches[:, s].reshape(Tprime, P)
       
       
        cov = X.T @ X/ (Tprime - 1)  # shape (P, P)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:2]
        leading_vecs = eigvecs[:, idx]  # shape (n_s, n_modes)
        projected = X @ leading_vecs

        a1, a2 = projected[:,0],projected[:,1]

        protophi = np.arctan2(a2, a1)
        protoamp = np.hypot(a1, a2)

        k = np.arange(1, nharm + 1)
        Sn_pos = np.mean(np.exp(-1j * np.outer(k, protophi)), axis=1)  # (nharm,)
        phi_corr = protophi.astype(np.complex128)     # start from protophi
        exp_term = np.exp(1j * np.outer(k, protophi)) - 1
        denom = (1j * k)[:, None]
        correction_terms = Sn_pos[:, None] * exp_term / denom
        phi_corr += 2 * np.sum(correction_terms, axis=0)  # result shape (n_t,)
        phi_corr = np.real(phi_corr)
        
        phi_un=np.unwrap(phi_corr)
        if phi_un[-1]-phi_un[0]<0:
            phi_corr = -phi_corr
       
        k = np.arange(0, nharm + 1) 
        An = np.mean(protoamp * np.exp(-1j * np.outer(k, phi_corr)), axis=1)    
        k_pos = np.arange(1, nharm + 1)
        An_pos = An[1:]
        # expand to negatives
        k_all = np.concatenate((-k_pos[::-1], [0], k_pos))
        An_all = np.concatenate((np.conj(An_pos[::-1]), [An[0]], An_pos))
        # reconstruct
        Aphi = np.sum(An_all[:, None] * np.exp(1j * np.outer(k_all, phi_corr)), axis=0).real

        norm_amp = protoamp / Aphi

        phi[:, s] = phi_corr
        amp[:, s] = norm_amp

    return phi, amp

def phi_amp_from_two_signals(s1: np.ndarray, s2, nharm: int) -> np.ndarray:
    if s1.ndim != 1:
        raise ValueError("signal must be a 1D NumPy array")
    k = np.arange(1, nharm + 1)              # shape (nharm,)
    ht=s1+1j*s2
    protophi = np.angle(ht)
    # Sn_pos[k] = < e^{-i k φ} >_t
    Sn_pos = np.mean(np.exp(-1j * np.outer(k, protophi)), axis=1)  # (nharm,)
    phi = protophi.astype(np.complex128)     # start from protophi

    # build the numerator:  e^{i k φ} - 1, shape (nharm, n_times)
    exp_term = np.exp(1j * np.outer(k, protophi)) - 1
    # reshape denominator to (nharm,1) so it divides down each row
    denom = (1j * k)[:, None]
    # now shape(Sn_pos[:,None]) = (nharm,1), exp_term=(nharm,n_t),
    # denom=(nharm,1) ⇒ correction_terms=(nharm,n_t)
    correction_terms = Sn_pos[:, None] * exp_term / denom
    # sum over harmonics and add
    phi += 2 * np.sum(correction_terms, axis=0)  # result shape (n_t,)
    phi = np.real(phi)

    phi_un=np.unwrap(phi)
    if phi_un[-1]-phi_un[0]<0:
        phi = -phi
    
    protoamp = np.abs(ht)

    k = np.arange(0, nharm + 1) 
    An = np.mean(protoamp * np.exp(-1j * np.outer(k, phi)), axis=1)    
    k_pos = np.arange(1, nharm + 1)
    An_pos = An[1:]
    # expand to negatives
    k_all = np.concatenate((-k_pos[::-1], [0], k_pos))
    An_all = np.concatenate((np.conj(An_pos[::-1]), [An[0]], An_pos))

    # reconstruct
    Aphi = np.sum(An_all[:, None] * np.exp(1j * np.outer(k_all, phi)), axis=0).real


    #  normalize protoamp
    norm_amp = protoamp / Aphi
    return phi, norm_amp



def get_global_phase_amplitude(arr, nharm):
    leading_modes_result = leadingModes(arr, n_modes=3)
    phi_corr,amp_corr=phi_amp_from_two_signals(leading_modes_result["projected"][:,0],leading_modes_result["projected"][:,1], nharm=nharm)

    z = np.exp(1j * phi_corr)
    dzdt = np.gradient(z, axis=0)  # central finite diff
    omega = np.imag(dzdt / z)       # dphi/dt
    omega=omega/np.mean(omega)

    return phi_corr, omega ,amp_corr

def get_local_phase_amplitude(arr, nharm, dt=3,ds=3):
    phi_corr, amp_corr=local_phase_pca2(arr,nharm,dt,ds)
    
    z = np.exp(1j * phi_corr)
    dzdt = np.gradient(z, axis=0)  # central finite diff
    omega = np.imag(dzdt / z)       # dphi/dt
    omega=omega/np.mean(omega)


    return phi_corr, omega ,amp_corr

    
