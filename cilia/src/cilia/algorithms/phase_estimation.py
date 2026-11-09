import numpy as np
from scipy.signal import hilbert

def estimate_phase_from_protophase(protophi, nharm=10):
    """
    Compute a proper phase from a protophase using harmonic correction.
    Returns a wrapped (not unwrapped) phase.
    """
    protophi = np.asarray(protophi)
    if protophi.ndim != 1:
        raise ValueError("protophi must be a 1D array")
    if not np.issubdtype(protophi.dtype, np.floating):
        raise ValueError("protophi must be a real-valued array")

    k = np.arange(1, nharm + 1)
    exp_kphi = np.exp(-1j * np.outer(k, protophi))
    Sn_pos = np.mean(exp_kphi, axis=1)

    phi = protophi.astype(np.complex128)
    correction_terms = Sn_pos[:, None] * (np.exp(1j * np.outer(k, protophi)) - 1) / (1j * k[:, None])
    phi += 2 * np.sum(correction_terms, axis=0)
    # phi=np.real(phi)
    # import matplotlib.pyplot as plt
    # plt.plot(np.gradient(np.unwrap(protophi)), label="proto")
    # # print(phi)
    # plt.plot(np.gradient(np.unwrap(phi)), label="corrected")
    # plt.legend()
    # plt.show()
    # # raise
    return np.real(phi)

def _enforce_positive_slope(phi):
    """
    Flip phase sign if it decreases on average.
    Works for any (unwrapped or wrapped then unwrapped) phase.
    """
    phi_u = np.unwrap(phi)
    if phi_u[-1]-phi_u[0] < 0:
        phi = -phi
    return phi

def estimate_relative_amplitude(amplitude, protophi, nharm=10, eps=1e-3):
    """
    Remove phase-dependent amplitude modulation via Fourier series.

    amplitude : array (Nt,)
    protophi  : array (Nt,)
    """

    amplitude = np.asarray(amplitude, dtype=float)
    protophi = np.asarray(protophi, dtype=float)

    k = np.arange(1, nharm + 1)

    # Fourier coefficients of amplitude vs phase
    exp_kphi = np.exp(-1j * np.outer(k, protophi))
    Sn = np.mean(amplitude * exp_kphi, axis=1)   # weighted!

    # reconstruct periodic trend
    A_trend = np.mean(amplitude) * np.ones_like(amplitude)

    A_trend += 2 * np.sum(
        (Sn[:, None] * np.exp(1j * np.outer(k, protophi))).real,
        axis=0
    )

    # avoid division problems
    A_trend = np.maximum(A_trend, eps)

    rel_amplitude = amplitude / A_trend

    return rel_amplitude, A_trend

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
    exp_term = np.exp(1j * np.outer(k, protophi)) - 1 #think about this!
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

def protophase_from_spatial_modes(tangent_angles):
    """
    Uses first two spatial modes (SVD) to extract a global protophase.
    
    Returns:
        dict with keys:
            - 'protophase': the phase values
            - 'percentile_1': 1st percentile of amplitude (distance from origin)
            - 'variance_explained': fraction of total variance explained by first 2 modes
    """
    Y = np.asarray(tangent_angles, dtype=float)

    if Y.ndim == 1:
        raise ValueError("Need spatial dimension for mode-based protophase")

    # subtract temporal mean per s (important!)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    # import matplotlib.pyplot as plt
    # plt.imshow(Yc.T, aspect='auto')
    # plt.show()
    # SVD: time x space
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

    a1 = U[:, 0] * S[0]
    a2 = U[:, 1] * S[1]

    a1 -= np.mean(a1)
    a2 -= np.mean(a2)
    rms1 = np.sqrt(np.mean(a1**2))
    rms2 = np.sqrt(np.mean(a2**2))
    if rms1 > 0:
        a1 /= rms1
    if rms2 > 0:
        a2 /= rms2

    # import matplotlib.pyplot as plt
    # plt.plot(Vt[0,:])
    # plt.plot(Vt[1,:])
    # plt.show()

    # plt.plot(a1)
    # plt.plot(a2)
    # plt.show()

    # plt.scatter(a1,a2,color='grey',s=5, alpha=0.2)
    # plt.show()

    # Compute diagnostics
    amplitude = np.sqrt(a1**2 + a2**2)
    percentile_1 = np.percentile(amplitude, 1)
    
    # Variance explained by first two modes
    variance_explained = (S[0]**2 + S[1]**2) / np.sum(S**2)

    protophi = np.arctan2(a2, a1)
    protophi = _enforce_positive_slope(protophi)
    
    # plt.plot(amplitude)
    # plt.show()

    # plt.scatter(protophi%2*np.pi, amplitude,color='grey',s=5, alpha=0.2)
    # plt.show()

    rel_amplitude, _ = estimate_relative_amplitude(
        amplitude,
        protophi,
        nharm=25
    )
    # plt.scatter(protophi%2*np.pi, rel_amplitude, color='grey',s=5, alpha=0.2)
    # plt.show()

    _, rel_amplitude=phi_amp_from_two_signals(a1,a2, nharm=10)
    return {
        'protophase': protophi,
        'rel_amplitude': rel_amplitude,
        'percentile_1': percentile_1,
        'variance_explained': variance_explained,
    }

def protophase_from_first_mode_hilbert(tangent_angles, pad_frac=0.2):
    """
    Extract protophase from first spatial mode using Hilbert transform.
    
    Returns:
        dict with keys:
            - 'protophase': the phase values
            - 'percentile_1': 1st percentile of |analytic signal| (comparable to SVD amplitude)
            - 'variance_explained': fraction of total variance explained by first mode
    """
    Y = np.asarray(tangent_angles, dtype=float)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)

    # first spatial mode
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    a1 = U[:, 0] * S[0]

    # Normalize a1 (same as in spatial modes case)
    a1 -= np.mean(a1)
    rms1 = np.sqrt(np.mean(a1**2))
    if rms1 > 0:
        a1 /= rms1

    # mirror padding
    N = len(a1)
    pad = int(pad_frac * N)
    a1_pad = np.concatenate([a1[pad:0:-1], a1, a1[-2:-pad-2:-1]])

    z = hilbert(a1_pad)
    phi = np.unwrap(np.angle(z))[pad:pad+N] # type: ignore

    # Compute diagnostics: amplitude of the analytic signal
    z_unpadded = z[pad:pad+N]
    amplitude = np.abs(z_unpadded) # type: ignore
    percentile_1 = np.percentile(amplitude, 1)
    
    # Variance explained by first mode
    variance_explained = S[0]**2 / np.sum(S**2)
    
    phi = _enforce_positive_slope(phi)
    
    return {
        'protophase': phi,
        'percentile_1': percentile_1,
        'variance_explained': variance_explained,
    }


def main():
    # -----------------------------
    # Ground-truth signal
    # -----------------------------
    A = 1.0
    f = 1.01
    omega = 2 * np.pi * f
    T = 50.0
    dt = 0.021

    t = np.arange(0, T, dt)
    Nt = t.size

    Ns = 60
    s = np.linspace(0.0, 1.0, Ns)

    k = 2 * np.pi * 2.0  # 2 spatial waves

    # traveling wave
    true_phase = omega * t
    tangent_angles = A * np.cos(true_phase[:, None] - k * s[None, :])

    # -----------------------------
    # Protophase estimates
    # -----------------------------
    proto_svd = protophase_from_spatial_modes(tangent_angles)
    proto_hilbert = protophase_from_first_mode_hilbert(tangent_angles)

    # Extract protophase and diagnostics
    phi_svd_data = proto_svd
    phi_hilbert_data = proto_hilbert

    # proper phase correction
    phi_svd = estimate_phase_from_protophase(phi_svd_data['protophase'])
    phi_hilbert = estimate_phase_from_protophase(phi_hilbert_data['protophase'])

    # unwrap for comparison
    phi_svd = np.unwrap(phi_svd)
    phi_hilbert = np.unwrap(phi_hilbert)

    # Print diagnostics
    print("\n=== Protophase diagnostics ===")
    print(f"SVD 1st percentile (amplitude)     = {phi_svd_data['percentile_1']:.3e}")
    print(f"SVD variance explained (2 modes)   = {phi_svd_data['variance_explained']:.4f}")
    print(f"Hilbert 1st percentile (amplitude) = {phi_hilbert_data['percentile_1']:.3e}")
    print(f"Hilbert variance explained (1 mode) = {phi_hilbert_data['variance_explained']:.4f}")

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def slope(phi):
        return np.polyfit(t, phi, 1)[0]

    print("\n=== Phase slope diagnostics ===")
    print(f"True ω         = {omega:.6f}")
    print(f"SVD ω̂         = {slope(phi_svd):.6f}")
    print(f"Hilbert ω̂     = {slope(phi_hilbert):.6f}")



    print("\n=== RMS phase error (mod 2π) ===")
    def best_signed_phase_error(phi, true_phase):
        """
        Compute RMS error modulo 2π, allowing sign flip and constant shift.
        """
        phi_u = np.unwrap(phi)
        true_u = np.unwrap(true_phase)

        def rms_mod2pi(x):
            d = (x + np.pi) % (2*np.pi) - np.pi
            return np.sqrt(np.mean(d**2))

        # try + and -
        errs = []
        for a in (+1.0, -1.0):
            x = a * phi_u

            # best constant shift b (circular mean of difference)
            diff = true_u - x
            b = np.angle(np.mean(np.exp(1j * diff)))
            errs.append(rms_mod2pi((x + b) - true_u))
        return min(errs)


    print(f"SVD RMS err       = {best_signed_phase_error(phi_svd,true_phase):.3e}")
    print(f"Hilbert RMS err   = {best_signed_phase_error(phi_hilbert,true_phase):.3e}")


    print("\n=== Harmonic perturbation test ===")

    # -------------------------------------------------
    # Add phase distortion (higher harmonic)
    # -------------------------------------------------
    eps = 0.3          # strength of distortion
    m = 3              # harmonic order

    distorted_protophi = true_phase + eps * np.sin(m * true_phase)

    # wrap to [-pi, pi] to mimic real protophase
    distorted_protophi = (distorted_protophi + np.pi) % (2*np.pi) - np.pi

    # enforce monotonic direction
    distorted_protophi = _enforce_positive_slope(distorted_protophi)

    # -------------------------------------------------
    # Apply correction
    # -------------------------------------------------
    corrected_phi = estimate_phase_from_protophase(distorted_protophi, nharm=50)
    corrected_phi = np.unwrap(corrected_phi)

    # -------------------------------------------------
    # Compare slopes
    # -------------------------------------------------
    print(f"True ω              = {omega:.6f}")
    print(f"Distorted ω̂        = {slope(np.unwrap(distorted_protophi)):.6f}")
    print(f"Corrected ω̂        = {slope(corrected_phi):.6f}")

    # -------------------------------------------------
    # RMS errors
    # -------------------------------------------------
    err_distorted = best_signed_phase_error(distorted_protophi, true_phase)
    err_corrected = best_signed_phase_error(corrected_phi, true_phase)

    print(f"Distorted RMS err   = {err_distorted:.3e}")
    print(f"Corrected RMS err   = {err_corrected:.3e}")

    # -------------------------------------------------
    # Optional visualization (very insightful)
    # -------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(t, np.unwrap(distorted_protophi)-true_phase, label="distorted", alpha=0.6)
    plt.plot(t, corrected_phi-true_phase, label="corrected", lw=2)
    plt.plot(t, true_phase-true_phase, "--", label="true", lw=2)
    plt.xlabel("t")
    plt.ylabel("phase")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
