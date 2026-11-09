import numpy as np

def estimate_phase_from_protophase_local(protophi, nharm=5):
    """
    Vectorized version for (Nt, Ns) protophase.

    Parameters
    ----------
    protophi : array (Nt, Ns)
    nharm : int

    Returns
    -------
    phase : array (Nt, Ns)
    """
    protophi = np.asarray(protophi)

    if protophi.ndim != 2:
        raise ValueError("protophi must be a 2D array (Nt, Ns)")
    if not np.issubdtype(protophi.dtype, np.floating):
        raise ValueError("protophi must be real-valued")

    Nt, Ns = protophi.shape
    k = np.arange(1, nharm + 1)  # (K,)

    # shape: (K, Nt, Ns)
    exp_neg = np.exp(-1j * k[:, None, None] * protophi[None, :, :])
    # average over time axis (Nt)
    Sn_pos = np.mean(exp_neg, axis=1)  # (K, Ns)
    # base phase (complex)
    phi = protophi.astype(np.complex128)
    # shape helpers
    exp_pos = np.exp(1j * k[:, None, None] * protophi[None, :, :])  # (K, Nt, Ns)
    correction_terms = (
        Sn_pos[:, None, :] * (exp_pos - 1.0) / (1j * k[:, None, None])
    )  # (K, Nt, Ns)
    phi = phi + 2.0 * np.sum(correction_terms, axis=0)  # (Nt, Ns)

    return np.real(phi)

def _dominant_period_from_psd(x: np.ndarray, fmin: float = 1e-6) -> int:
    """
    Estimate dominant period (in samples) from PSD peak of a 1D signal x.
    Assumes unit sampling interval (dt=1 frame).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return max(1, n)

    x = x - np.mean(x)
    # rFFT PSD
    X = np.fft.rfft(x)
    psd = (X.real**2 + X.imag**2)

    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles / sample
    # ignore DC and extremely low frequencies
    valid = freqs >= fmin
    valid[0] = False  # force DC off

    if not np.any(valid):
        return max(1, n)

    k = np.argmax(psd[valid])
    f_peak = freqs[valid][k]
    if f_peak <= 0:
        return max(1, n)

    T = int(np.round(1.0 / f_peak))
    return int(np.clip(T, 1, n))


import matplotlib.pyplot as plt

def _plot_kymographs(tangent_angles, phase, amplitude, *, wrap_phase=True):
    """
    Debug plot: tangent angle, phase (cyclic cmap, NO interpolation), amplitude.

    Parameters
    ----------
    tangent_angles, phase, amplitude : (Nt, Ns) arrays
    wrap_phase : bool
        If True, wrap phase to [0, 2π) before plotting.
    """
    tangent_angles = np.asarray(tangent_angles)
    phase = np.asarray(phase)
    amplitude = np.asarray(amplitude)

    if wrap_phase:
        phase_show = np.mod(phase, 2*np.pi)
        vmin, vmax = 0.0, 2*np.pi
    else:
        phase_show = phase
        vmin, vmax = -np.pi, np.pi

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Tangent angle
    im0 = axes[0].imshow(
        tangent_angles,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        resample=False,  # remove if mpl complains
    )
    axes[0].set_title("Tangent Angle")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    # Phase (cyclic, no smoothing)
    im1 = axes[1].imshow(
        phase_show,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        resample=False,  # remove if mpl complains
    )
    axes[1].set_title("Phase")
    axes[1].set_xlabel("s")
    fig.colorbar(im1, ax=axes[1])

    # Amplitude
    im2 = axes[2].imshow(
        amplitude,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        resample=False,  # remove if mpl complains
    )
    axes[2].set_title("Amplitude")
    axes[2].set_xlabel("s")
    fig.colorbar(im2, ax=axes[2])

    plt.show()

def subtract_gauge_keep_harmonics_fft(
    A: np.ndarray,
    *,
    s0: int = 0,
    n_harm: int = 5,
    fmin: float = 1e-6,
):
    """
    Smooth gauge A[:, s0] by keeping harmonics up to n_harm of the
    fundamental beat frequency. Everything above (n_harm + 0.5)*k0 is removed.

    Returns
    -------
    A_gauged : (Nt, Ns)
    g_smooth : (Nt,)
    aux : dict with k0 and k_cut
    """
    A = np.asarray(A, dtype=float)
    Nt, Ns = A.shape

    g = A[:, s0].astype(float)

    G = np.fft.fft(g)
    psd = (G.real**2 + G.imag**2)

    freqs = np.fft.fftfreq(Nt, d=1.0)

    # find fundamental (ignore DC)
    pos = freqs > fmin
    if not np.any(pos):
        return A.copy(), np.zeros_like(g), {"k0": 0, "k_cut": 0}

    k_pos = np.where(pos)[0]
    k0 = int(k_pos[np.argmax(psd[k_pos])])

    # harmonic cutoff
    k_cut = int((n_harm + 0.5) * k0)

    # build mask
    k_indices = np.arange(Nt)
    k_signed = np.minimum(k_indices, Nt - k_indices)  # |k| for FFT ordering

    keep = k_signed <= k_cut

    G_filt = np.where(keep, G, 0.0)
    g_smooth = np.fft.ifft(G_filt).real

    A_gauged = A - g_smooth[:, None]

    return A_gauged, g_smooth, {"k0": k0, "k_cut": k_cut}

def _align_signs_from_middle(Y: np.ndarray, s_mid: int | None = None) -> np.ndarray:
    """
    Align sign of each column Y[:, s] to make it consistent across s.
    Starts at s_mid and propagates left/right.
    Uses dot-product with neighbor to decide sign.

    Y: (Nt, Ns)
    returns: aligned Y (Nt, Ns)
    """
    Y = np.asarray(Y, dtype=float).copy()
    Nt, Ns = Y.shape
    if s_mid is None:
        s_mid = Ns // 2

    # normalize columns for stable dot products (optional but helps)
    Yn = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-12)

    # go left: s_mid-1, ..., 0
    for s in range(s_mid - 1, -1, -1):
        # choose sign to match right neighbor
        if np.dot(Yn[:, s], Yn[:, s + 1]) < 0:
            Y[:, s] *= -1
            Yn[:, s] *= -1

    # go right: s_mid+1, ..., Ns-1
    for s in range(s_mid + 1, Ns):
        # choose sign to match left neighbor
        if np.dot(Yn[:, s], Yn[:, s - 1]) < 0:
            Y[:, s] *= -1
            Yn[:, s] *= -1

    return Y

def correct_amplitude_from_phase(protoamp, phase, nharm=5, eps=1e-12):
    """
    Vectorized amplitude correction for (Nt, Ns).

    protoamp : (Nt, Ns)
    phase    : (Nt, Ns)

    returns normalized amplitude
    """
    Nt, Ns = phase.shape

    k = np.arange(0, nharm + 1)  # (K,)

    # Fourier coefficients
    exp_neg = np.exp(-1j * k[:, None, None] * phase[None, :, :])  # (K, Nt, Ns)
    An = np.mean(protoamp[None, :, :] * exp_neg, axis=1)          # (K, Ns)

    # positive harmonics
    k_pos = np.arange(1, nharm + 1)
    An_pos = An[1:]  # (K, Ns)

    # build symmetric spectrum
    k_all = np.concatenate((-k_pos[::-1], [0], k_pos))  # (2K+1,)
    An_all = np.concatenate(
        (np.conj(An_pos[::-1]), An[[0]], An_pos),
        axis=0
    )  # (2K+1, Ns)

    # reconstruct A(phi)
    exp_pos = np.exp(1j * k_all[:, None, None] * phase[None, :, :])
    Aphi = np.sum(An_all[:, None, :] * exp_pos, axis=0).real  # (Nt, Ns)

    return protoamp / (Aphi + eps)

def get_local_phase(
    tangent_angles: np.ndarray,
    *,
    max_wt: int = 4,
    max_ws: int = 4,
    eps: float = 1e-12,
):
    A = np.asarray(tangent_angles, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"tangent_angles must be 2D (Nt,Ns); got shape {A.shape}")

    Nt, Ns = A.shape

    # --- early return if trivially too small ---
    if Nt < 20 or Ns < 15:
        phase = np.full((Nt, Ns), np.nan, dtype=float)
        amp = np.full((Nt, Ns), np.nan, dtype=float)
        aux = {
            "T": 1,
            "wt": 0,
            "ws": 0,
            "too_small": True,
            "amp_p01_s": np.zeros((Ns,), dtype=float),
            "msd_dphi_norm_s": np.zeros((Ns,), dtype=float),
        }
        return phase, amp, aux

    # ---Determine dominant period T from PSD peak ---
    A0 = A - A.mean(axis=0)
    U, S, VT = np.linalg.svd(A0, full_matrices=False)
    x = U[:, 0]
    T = _dominant_period_from_psd(x)

    too_small = not (Ns > 15 and Nt >= 5 * T)
    if too_small:
        phase = np.full((Nt, Ns), np.nan, dtype=float)
        amp = np.full((Nt, Ns), np.nan, dtype=float)
        aux = {
            "T": 1,
            "wt": 0,
            "ws": 0,
            "too_small": True,
            "amp_p01_s": np.zeros((Ns,), dtype=float),
            "msd_dphi_norm_s": np.zeros((Ns,), dtype=float),
        }
        return phase, amp, aux

    # --- Window sizes ---
    wt = int(min(max(T // 10, 1), max_wt))
    ws = int(min(max(Ns // 5, 1), max_ws))
    Lt = 2 * wt + 1
    Ls = 2 * ws + 1
    D = Lt * Ls

    Ap = np.pad(A0, pad_width=((wt, wt), (ws, ws)), mode="reflect")

    # --- main projections ---
    Ysym = np.zeros((Nt, Ns), dtype=float)
    Yanti = np.zeros((Nt, Ns), dtype=float)

    for s in range(Ns):
        sp = s + ws

        V = np.empty((Nt, D), dtype=float)
        for t in range(Nt):
            tp = t + wt
            patch = Ap[tp - wt: tp + wt + 1, sp - ws: sp + ws + 1]  # (Lt, Ls)
            V[t, :] = patch.reshape(-1)

        V3 = V.reshape(Nt, Lt, Ls)
        V3_rev = V3[:, ::-1, :]
        Vsym = 0.5 * (V3 + V3_rev).reshape(Nt, D)
        Vanti = 0.5 * (V3 - V3_rev).reshape(Nt, D)

        Vsym_c = Vsym - Vsym.mean(axis=0, keepdims=True)
        Vanti_c = Vanti - Vanti.mean(axis=0, keepdims=True)

        def leading_pc(Xc):
            if np.allclose(Xc, 0.0):
                v = np.zeros(D, dtype=float)
                v[0] = 1.0
                return v
            _, _, VTloc = np.linalg.svd(Xc, full_matrices=False)
            v = VTloc[0].copy()
            j = np.argmax(np.abs(v))
            if v[j] < 0:
                v *= -1.0
            return v / (np.linalg.norm(v) + eps)

        v_sym = leading_pc(Vsym_c)
        v_anti = leading_pc(Vanti_c)

        y_sym = Vsym_c @ v_sym
        y_anti = Vanti_c @ v_anti

        # normalize to unit variance (as you do)
        y_sym /= (np.std(y_sym) + eps)
        y_anti /= (np.std(y_anti) + eps)

        Ysym[:, s] = y_sym
        Yanti[:, s] = y_anti

    # align signs across s
    Ysym = _align_signs_from_middle(Ysym, s_mid=Ns // 2)
    Yanti = _align_signs_from_middle(Yanti, s_mid=Ns // 2)
    phase = np.arctan2(Yanti, Ysym) 
    amplitude = np.hypot(Ysym, Yanti)
    
    # enforce positive temporal winding
    phase_u = np.unwrap(phase, axis=0)
    if np.sum(phase_u[-1,:]-phase_u[0,:]) < 0:
        phase = -phase

    phase = estimate_phase_from_protophase_local(phase)
    amplitude = correct_amplitude_from_phase(amplitude, phase)

    #  1st percentile of amplitude per s
    amp_p01_s = np.percentile(amplitude, 1.0, axis=0)  # shape (Ns,)

    # MSD of consecutive phase differences, normalized by frequency
    # Use unwrapped phase in time to make dphi meaningful.
    phase_u = np.unwrap(phase, axis=0)  # (Nt, Ns)
    dphi = np.diff(phase_u, axis=0)     # (Nt-1, Ns)

    # normalize by angular frequency ω = 2π/T (rad per frame)
    omega = 2.0 * np.pi / max(T, 1)  # avoid div0
    dphi_norm = dphi / (omega + eps)
    dphi_norm_centered = dphi_norm - np.mean(dphi_norm, axis=0, keepdims=True)
    msd_dphi_norm_s = np.mean(dphi_norm_centered**2, axis=0)  # (Ns,)
    aux = {
        "T": int(T),
        "wt": int(wt),
        "ws": int(ws),
        "too_small": bool(too_small),
        "amp_p01_s": amp_p01_s,
        "msd_dphi_norm_s": msd_dphi_norm_s,
    }

    # optional debugging plot
    # _plot_kymographs(amplitude, phase, amplitude)

    return phase, amplitude, aux


