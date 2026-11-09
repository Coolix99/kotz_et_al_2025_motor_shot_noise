import numpy as np

def residual_std_from_cycle(
    phase_array: np.ndarray,
    tangent_angle: np.ndarray,
    periodic_avg: np.ndarray,
) -> float:
    """
    Compute RMS residual between data and reconstructed limit cycle.

    Parameters
    ----------
    phase_array : (Nt,)
    tangent_angle : (Nt, Ns)
    periodic_avg : (n_phase, Ns)

    Returns
    -------
    sigma_hat : float
        RMS deviation around limit cycle.
    """

    phase = np.asarray(phase_array, float)
    Y = np.asarray(tangent_angle, float)
    cycle = np.asarray(periodic_avg, float)

    Nt, Ns = Y.shape
    n_phase = cycle.shape[0]

    # fold phase
    phase_mod = np.mod(phase, 2*np.pi)

    # corresponding phase grid
    phase_uniform = np.linspace(0, 2*np.pi, n_phase, endpoint=False)

    # interpolate cycle at measured phases
    # periodic interpolation
    cycle_interp = np.zeros_like(Y)

    for j in range(Ns):
        cycle_interp[:, j] = np.interp(
            phase_mod,
            phase_uniform,
            cycle[:, j],
            period=2*np.pi,
        )

    residual = Y - cycle_interp

    sigma_hat = np.sqrt(np.mean(residual**2))

    return sigma_hat


def periodic_avg(
    phase_array: np.ndarray,
    tangent_angle: np.ndarray,
    *,
    n_phase: int = 100,
    sigma: float = 0.04,
) -> np.ndarray:
    """
    Phase-locked (periodic) average of tangent-angle series.

    Parameters
    ----------
    phase_array : ndarray, shape (Nt,)
        Phase time series (can be wrapped or unwrapped).
    tangent_angle : ndarray, shape (Nt, Ns)
        Tangent angle field Ψ(t,s).
    n_phase : int
        Number of phase bins over [0, 2π).
    sigma : float
        Gaussian kernel width in radians.

    Returns
    -------
    periodic_avg : ndarray, shape (n_phase, Ns)
        Phase-averaged tangent angle Ψ̄(φ,s).
    """
    phase = np.asarray(phase_array, dtype=float)
    Y = np.asarray(tangent_angle, dtype=float)
    
    if phase.ndim != 1:
        raise ValueError("phase_array must be 1D")
    if Y.ndim != 2:
        raise ValueError("tangent_angle must be 2D (Nt, Ns)")
    if Y.shape[0] != phase.shape[0]:
        raise ValueError("phase_array and tangent_angle must have same Nt")
    Nt, Ns = Y.shape
    # Phase folded to [0, 2π)
    phase_mod = np.mod(phase, 2 * np.pi)

    # Uniform phase grid
    phase_uniform = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)

    # Circular phase differences (n_phase, Nt)
    dphi = (
        phase_mod[None, :]
        - phase_uniform[:, None]
        + np.pi
    ) % (2 * np.pi) - np.pi

    # Gaussian kernel weights
    W = np.exp(-0.5 * (dphi / sigma) ** 2)

    # Normalize weights per phase bin
    W_sum = np.sum(W, axis=1, keepdims=True)
    W_sum[W_sum == 0.0] = 1.0
    W /= W_sum

    # Phase-averaged tangent angles
    periodic_avg = W @ Y   # (n_phase, Ns)

    cycle_interp = np.zeros_like(Y)

    for j in range(Ns):
        cycle_interp[:, j] = np.interp(
            phase_mod,
            phase_uniform,
            periodic_avg[:, j],
            period=2*np.pi,
        )

    residual = Y - cycle_interp
    sigma_hat = np.sqrt(np.mean(residual**2))
    std_phase = sigma_hat / np.sqrt(Nt/10) #heruristic
    uncertainty = float(np.sqrt(np.mean(std_phase**2)))

    # import matplotlib.pyplot as plt
    # plt.imshow(periodic_avg.T, aspect="auto")
    # plt.show()
    return periodic_avg, uncertainty



# import numpy as np


def periodic_avg_fourier(
    phase_array: np.ndarray,
    tangent_angle: np.ndarray,
    *,
    n_phase: int = 100,
    max_mode: int = 5,
) -> np.ndarray:
    """
    Fourier-based phase-locked average using truncated Fourier series.

    Parameters
    ----------
    phase_array : ndarray (Nt,)
        Phase time series (wrapped or unwrapped).
    tangent_angle : ndarray (Nt, Ns)
        Tangent angle field Ψ(t,s).
    n_phase : int
        Number of uniform phase samples in output.
    max_mode : int
        Keep Fourier modes |k| <= max_mode.

    Returns
    -------
    periodic_avg : ndarray (n_phase, Ns)
    """

    phase = np.asarray(phase_array, dtype=float)
    Y = np.asarray(tangent_angle, dtype=float)

    Nt, Ns = Y.shape

    # Fold phase to [0, 2π)
    phase_mod = np.mod(phase, 2*np.pi)

    # --- Build Fourier design matrix ---
    # Columns: [1, cos(φ), sin(φ), cos(2φ), sin(2φ), ...]
    K = max_mode
    n_cols = 1 + 2*K
    A = np.zeros((Nt, n_cols))

    A[:, 0] = 1.0
    for k in range(1, K+1):
        A[:, 2*k-1] = np.cos(k * phase_mod)
        A[:, 2*k]   = np.sin(k * phase_mod)

    # Solve least squares for all s simultaneously
    # Coeff shape: (n_cols, Ns)
    coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)

    # --- Reconstruct on uniform phase grid ---
    phase_uniform = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    A_uniform = np.zeros((n_phase, n_cols))
    A_uniform[:, 0] = 1.0
    for k in range(1, K+1):
        A_uniform[:, 2*k-1] = np.cos(k * phase_uniform)
        A_uniform[:, 2*k]   = np.sin(k * phase_uniform)

    periodic_avg = A_uniform @ coeffs

    Y_fit = A @ coeffs
    residual = Y - Y_fit

    dof = Nt - n_cols
    sigma_hat = np.sqrt(np.sum(residual**2) / (dof * Ns))

    ATA_inv = np.linalg.inv(A.T @ A)

    std_phase = np.zeros(n_phase)

    for i in range(n_phase):
        a_phi = A_uniform[i, :]
        var_phi = sigma_hat**2 * (a_phi @ ATA_inv @ a_phi)
        std_phase[i] = np.sqrt(var_phi)

    # Global scalar uncertainty
    uncertainty = float(np.sqrt(np.mean(std_phase**2)))
    return periodic_avg, uncertainty


def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # --- synthetic traveling wave ---
    Nt = 4000
    Ns = 60
    f = 1.0
    omega = 2 * np.pi * f

    t = np.linspace(0, 20, Nt)
    phase = omega * t

    s = np.linspace(0, 1, Ns)
    a = 1.0 + 0.2 * s
    k = -2 * np.pi * 3.0

    # exact waveform
    psi_true = (
        a[None, :]
        * np.cos(phase[:, None] + k * s[None, :])
    )

    # periodic average
    psi_avg = periodic_avg(
        phase,
        psi_true,
    )

    # expected result
    phase_uniform = np.linspace(0, 2*np.pi, psi_avg.shape[0], endpoint=False)
    psi_expected = (
        a[None, :]
        * np.cos(phase_uniform[:, None] + k * s[None, :])
    )

    # choose a few representative arclength positions
    idx_s = np.linspace(0, psi_avg.shape[1]-1, 6, dtype=int)

    plt.figure(figsize=(6, 4))

    for j in idx_s:
        plt.plot(
            phase_uniform,
            psi_expected[:, j],
            color="k",
            lw=2,
            alpha=0.6,
            label="exact" if j == idx_s[0] else None,
        )
        plt.plot(
            phase_uniform,
            psi_avg[:, j],
            "--",
            lw=2,
            label="periodic avg" if j == idx_s[0] else None,
        )

    plt.xlim(0, 2*np.pi)
    plt.xlabel("phase φ")
    plt.ylabel("tangent angle Ψ")
    plt.legend()
    plt.tight_layout()
    plt.show()

    err = np.sqrt(np.mean((psi_avg - psi_expected) ** 2))

    print("=== PERIODIC AVERAGE TEST ===")
    print(f"RMS error = {err:.3e}")
    

if __name__ == "__main__":
    main()
