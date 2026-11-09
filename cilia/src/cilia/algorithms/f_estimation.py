import numpy as np
# from scipy.signal import welch


def estimate_f_endpoint(phase_wrapped, dt):
    """
    Trivial frequency estimate from endpoints:
    f_hat = (phi[-1] - phi[0]) / T, with T = (N - 1) * dt.
    """
    if phase_wrapped is None or len(phase_wrapped) < 2:
        return np.nan

    phi = np.unwrap(phase_wrapped)
    T = (len(phi) - 1) * dt
    if T == 0:
        return np.nan

    return (phi[-1] - phi[0]) / T /np.pi/2

def estimate_f_from_tangent_angle_power(
    tangent_angles,
    dt_frame,
):
    """
    Estimate dominant frequency from tangent-angle time series using PSD.

    Phase-robust:
      - PSD computed per arclength
      - power averaged over space using mean
        (assumes equidistant positions)

    Parameters
    ----------
    tangent_angles : array, shape (Nt, Ns)
    dt_frame : float
        Time step between frames (seconds)

    Returns
    -------
    f : float
        Estimated frequency in Hz
    """
    Y = np.asarray(tangent_angles, dtype=float)

    if Y.ndim == 1:
        Y = Y[:, None]

    Nt, Ns = Y.shape
    if Nt < 4 or Ns < 2:
        return np.nan

    # FFT along time
    fft = np.fft.rfft(Y, axis=0)
    psd = (np.abs(fft) ** 2) / (Nt ** 2)

    # remove DC
    psd[0, :] = 0.0

    # spatial average via mean
    # psd has shape (Nf, Ns)
    psd_mean = np.mean(psd, axis=1)

    k0 = int(np.argmax(psd_mean))
    if k0 == 0:
        return np.nan

    # frequency grid
    freqs = np.fft.rfftfreq(Nt, d=dt_frame)

    return float(freqs[k0])

def main():
    A_true = 1.5
    f_true = 1.25
    dt = 0.011
    T = 120.0

    t = np.arange(0, T, dt)
    s = np.linspace(0.0, 1.1, 50)

    # Traveling wave with phase shift in s
    Y = A_true * np.sin(2 * np.pi * f_true * t[:, None] + 3.0 * s[None, :])

    f_est = estimate_f_from_tangent_angle_power(
        tangent_angles=Y,
        dt_frame=dt,
    )

    print(f"True f      = {f_true:.6f} Hz")
    print(f"Estimated f = {f_est:.6f} Hz")

if __name__ == "__main__":
    main()