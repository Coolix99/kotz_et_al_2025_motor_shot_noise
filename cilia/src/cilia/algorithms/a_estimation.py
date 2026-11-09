import numpy as np

def estimate_amplitude_mean_square(tangent_angles):
    Y = np.asarray(tangent_angles, float)

    # local mean square over time
    ms_s = np.mean(Y**2, axis=0)

    # local amplitude squared per s
    A2_s = 2.0 * ms_s

    # spatial RMS of amplitude (averaging A² over s, then sqrt)
    A = np.sqrt(np.mean(A2_s))
  
    return float(A)

def estimate_amplitude_psd(tangent_angles, band_fraction=0.10):
    Y = np.asarray(tangent_angles, dtype=float)

    if Y.ndim == 1:
        Y = Y[:, None]

    Nt, Ns = Y.shape
    if Nt < 4:
        return np.nan

    fft = np.fft.rfft(Y, axis=0)
    psd = (np.abs(fft) ** 2) / (Nt ** 2)
    psd[0, :] = 0.0

    psd_mean = np.mean(psd, axis=1)
    k0 = int(np.argmax(psd_mean))
    if k0 == 0:
        return np.nan

    dk = max(1, int(round(band_fraction * k0)))
    k_lo = max(1, k0 - dk)
    k_hi = min(psd.shape[0] - 1, k0 + dk)

    P_s = np.sum(psd[k_lo:k_hi+1, :], axis=0)

    # local amplitude squared per s
    A2_s = 4.0 * P_s

    # spatial RMS (averaging A² over s, then sqrt)
    A = np.sqrt(np.mean(A2_s))

    return float(A)


def amplitude_first_harmonic_cycle(
    psi_avg: np.ndarray,
):
    """
    Extract amplitude from first Fourier component of a periodic waveform
    sampled uniformly over phase.

    Parameters
    ----------
    psi_avg : ndarray, shape (Nphi, Ns)
        Periodic average Ψ̄(φ,s), uniformly sampled over one cycle

    Returns
    -------
    A : float
        RMS amplitude: sqrt(<A(s)^2>_s)
    """
    psi = np.asarray(psi_avg, float)

    Nphi = psi.shape[0]
    phi = 2.0 * np.pi * np.arange(Nphi) / Nphi

    # --- remove static shape ψ0(s) ---
    psi = psi - np.mean(psi, axis=0, keepdims=True)

    # First Fourier mode
    c1c = 2.0 * np.mean(psi * np.cos(phi[:, None]), axis=0)
    c1s = 2.0 * np.mean(psi * np.sin(phi[:, None]), axis=0)

    # amplitude squared per s
    A2_s = c1c**2 + c1s**2

    # spatial RMS: average A² over s, then sqrt
    return float(np.sqrt(np.mean(A2_s)))


def amplitude_mean_square_cycle(psi_avg):
    psi = np.asarray(psi_avg, float)

    psi = psi - np.mean(psi, axis=0, keepdims=True)

    ms_s = np.mean(psi**2, axis=0)

    # amplitude squared per s
    A2_s = 2.0 * ms_s

    # spatial RMS: average A² over s, then sqrt
    A = np.sqrt(np.mean(A2_s))

    return float(A)


def main():
    import numpy as np

    # -------------------------------
    # parameters
    # -------------------------------
    A_true = 1.5
    f = 1.1
    dt = 0.021
    T = 120.0

    t = np.arange(0, T, dt)
    s = np.linspace(0.0, 1.0, 50)

    # -------------------------------
    # time-domain signal
    # -------------------------------
    y = A_true * np.sin(2 * np.pi * f * t)
    tangent_angles = np.tile(y[:, None], (1, s.size))

    A_ms_time = estimate_amplitude_mean_square(tangent_angles)
    A_psd_time = estimate_amplitude_psd(tangent_angles)

    # -------------------------------
    # build perfect periodic average
    # -------------------------------
    Nphi = 400
    phi = 2 * np.pi * np.arange(Nphi) / Nphi

    a_s = A_true * np.ones_like(s)
    psi_avg = a_s[None, :] * np.sin(phi[:, None])

    A_ms_cycle = amplitude_mean_square_cycle(psi_avg)
    A_harm_cycle = amplitude_first_harmonic_cycle(psi_avg)

    # -------------------------------
    # report (all using RMS: sqrt of averaged A²(s))
    # -------------------------------
    print("\n=== AMPLITUDE ESTIMATOR TEST (RMS-based) ===")
    print(f"True A                  = {A_true:.6f}")
    print(f"Time MSD-based A (RMS)  = {A_ms_time:.6f}")
    print(f"Time PSD-based A (RMS)  = {A_psd_time:.6f}")
    print(f"Cycle MSD-based A (RMS) = {A_ms_cycle:.6f}")
    print(f"Cycle 1st-harmonic A (RMS) = {A_harm_cycle:.6f}")

    # ========================================
    # TEST 2: Non-uniform amplitude A(s)
    # ========================================
    print("\n" + "="*70)
    print("=== TEST 2: LINEARLY VARYING AMPLITUDE A(s) ===")
    print("="*70)

    A_min = 0.5
    A_max = 2.5
    a_s_varying = np.linspace(A_min, A_max, len(s))

    # Theoretical RMS amplitude: sqrt(<A(s)^2>_s)
    A2_theory = a_s_varying ** 2
    A_theory_rms = np.sqrt(np.mean(A2_theory))

    # Generate time-domain signal with spatially-varying amplitude
    y_varying = a_s_varying[None, :] * np.sin(2 * np.pi * f * t[:, None])
    A_ms_time_varying = estimate_amplitude_mean_square(y_varying)
    A_psd_time_varying = estimate_amplitude_psd(y_varying)

    # Generate periodic average with spatially-varying amplitude
    psi_avg_varying = a_s_varying[None, :] * np.sin(phi[:, None])
    A_ms_cycle_varying = amplitude_mean_square_cycle(psi_avg_varying)
    A_harm_cycle_varying = amplitude_first_harmonic_cycle(psi_avg_varying)

    print(f"\nAmplitude range: A(s) ∈ [{A_min:.3f}, {A_max:.3f}]")
    print(f"\nTheoretical RMS amplitude: {A_theory_rms:.6f}")
    print(f"  (via: sqrt(1/L * ∫ A²(s) ds))")
    print(f"\nEstimated amplitudes:")
    print(f"  Time MSD-based (RMS)      = {A_ms_time_varying:.6f} (error: {abs(A_ms_time_varying - A_theory_rms):.6f})")
    print(f"  Time PSD-based (RMS)      = {A_psd_time_varying:.6f} (error: {abs(A_psd_time_varying - A_theory_rms):.6f})")
    print(f"  Cycle MSD-based (RMS)     = {A_ms_cycle_varying:.6f} (error: {abs(A_ms_cycle_varying - A_theory_rms):.6f})")
    print(f"  Cycle 1st-harmonic (RMS)  = {A_harm_cycle_varying:.6f} (error: {abs(A_harm_cycle_varying - A_theory_rms):.6f})")


if __name__ == "__main__":
    main()
