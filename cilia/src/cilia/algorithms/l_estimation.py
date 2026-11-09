import numpy as np


def _trapz_mean(values: np.ndarray, s: np.ndarray) -> float:
    """
    Trapezoidal mean over arclength, robust to non-uniform spacing.
    """
    s = np.asarray(s, dtype=float)
    values = np.asarray(values, dtype=float)

    L = float(s[-1] - s[0])
    if L <= 0:
        raise ValueError("Invalid arclength: s[-1] must be > s[0].")
    return float(np.trapz(values, s) / L)

def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """
    Weighted least squares for y ≈ b + m x.
    Returns (m, b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    if x.size < 2:
        return np.nan, np.nan

    W = np.sum(w)
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan

    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W

    Sxx = np.sum(w * (x - xbar) ** 2)
    if Sxx <= 0:
        return np.nan, np.nan

    Sxy = np.sum(w * (x - xbar) * (y - ybar))
    m = Sxy / Sxx
    b = ybar - m * xbar
    return float(m), float(b)

def estimate_lambda_from_tangent_angle_series(
    tangent_angles: np.ndarray,
    arclength: np.ndarray,
    *,
    power_thresh_fraction: float = 0.01,
    band_fraction: float = 0.0,
) -> dict:
    """
    Estimate spatial wavelength lambda from tangent-angle series.

    Algorithm
    ---------
    1) Compute temporal FFT per arclength position s (axis=0 is time).
    2) Compute PSD per s and trapezoid-average over arclength to get a global PSD.
    3) Pick peak temporal frequency bin (excluding DC).
    4) Extract complex coefficient Z(s) at that frequency; phi(s)=unwrap(angle(Z(s))).
    5) Exclude s where power(s) < power_thresh_fraction * mean_power (mean via trapz).
    6) Fit phi(s) ≈ phi0 + k s (weighted by power).
    7) Return lambda = 2π / |k|.

    Notes
    -----
    - lambda returned in same units as arclength.
    - sign(k) gives wave direction; lambda is always positive.
    - Optional `band_fraction` allows averaging the complex coefficient over a small
      band around the peak to stabilize phase in noise (set 0 for “single-bin”).
    """
    Y = np.asarray(tangent_angles, dtype=float)
    s = np.asarray(arclength, dtype=float)

    if Y.ndim == 1:
        raise ValueError("Need a spatial dimension (Nt, Ns). Got 1D.")
    if Y.ndim != 2:
        raise ValueError("tangent_angles must have shape (Nt, Ns).")

    Nt, Ns = Y.shape
    if s.ndim != 1 or s.size != Ns:
        raise ValueError("arclength must be shape (Ns,) matching tangent_angles second axis.")
    if Nt < 8:
        return {"lambda": np.nan, "k": np.nan, "f_peak": np.nan, "n_used": 0}

    # optional: remove temporal mean per s to kill DC leakage
    Yc = Y - np.mean(Y, axis=0, keepdims=True)

    # FFT along time
    F = np.fft.rfft(Yc, axis=0)  # shape (Nf, Ns)
    freqs = np.fft.rfftfreq(Nt,) 

    psd = (np.abs(F) ** 2) / (Nt ** 2)  # shape (Nf, Ns)
    psd[0, :] = 0.0  # remove DC

    # global PSD: trapz-average over arclength (handles uneven spacing)
    psd_global = np.array([_trapz_mean(psd[k, :], s) for k in range(psd.shape[0])], dtype=float)

    k0 = int(np.argmax(psd_global))
    if k0 <= 0 or not np.isfinite(psd_global[k0]) or psd_global[k0] <= 0:
        return {"lambda": np.nan, "k": np.nan, "f_peak": np.nan, "n_used": 0}

    f_peak = float(freqs[k0])

    # optional: average complex coefficient over a band around peak (stabilizes phase)
    if band_fraction and band_fraction > 0:
        dk = max(1, int(round(band_fraction * k0)))
        k_lo = max(1, k0 - dk)
        k_hi = min(F.shape[0] - 1, k0 + dk)
        Z = np.mean(F[k_lo : k_hi + 1, :], axis=0)
        power_s = np.mean(psd[k_lo : k_hi + 1, :], axis=0)
    else:
        Z = F[k0, :]
        power_s = psd[k0, :]

    mean_power = _trapz_mean(power_s, s)
    if not np.isfinite(mean_power) or mean_power <= 0:
        return np.nan

    # exclude weak positions
    mask = power_s >= (power_thresh_fraction * mean_power)
    if np.count_nonzero(mask) < 3:
        return np.nan

    s_use = s[mask]
    phi_use = np.unwrap(np.angle(Z[mask]))
    w_use = power_s[mask].astype(float)

    # fit phi(s) = b + k*s
    k_hat, b_hat = _weighted_linear_fit(s_use, phi_use, w_use)
    if not np.isfinite(k_hat) or k_hat == 0.0:
        return {"lambda": np.nan, "k": k_hat, "f_peak": f_peak, "n_used": int(s_use.size)}

    lam = -float(2.0 * np.pi / k_hat)
    return lam

def estimate_lambda_from_cycle(
    psi_avg: np.ndarray,
    uncertainty: float,
    arclength: np.ndarray,
    *,
    n_sigma: float = 3.0,
) -> float:
    """
    Estimate wavelength lambda from periodic average Ψ̄(φ,s).

    Parameters
    ----------
    psi_avg : ndarray, shape (Nphi, Ns)
        Periodic average sampled uniformly in phase.
    arclength : ndarray, shape (Ns,)
    power_thresh_fraction : float
        Fraction of mean harmonic power used to mask weak s.

    Returns
    -------
    lambda : float
        Spatial wavelength in arclength units.
    """
    psi = np.asarray(psi_avg, float)
    s = np.asarray(arclength, float)

    if psi.ndim != 2:
        raise ValueError("psi_avg must have shape (Nphi, Ns)")
    if s.ndim != 1 or s.size != psi.shape[1]:
        raise ValueError("arclength must match second axis of psi_avg")

    Nphi, Ns = psi.shape

    # uniform phase grid
    phi = 2.0 * np.pi * np.arange(Nphi) / Nphi

    # remove static shape ψ0(s)
    psi = psi - np.mean(psi, axis=0, keepdims=True)

    # first Fourier coefficient in phase
    c1 = (
        np.mean(psi * np.cos(phi[:, None]), axis=0)
        + 1j * np.mean(psi * np.sin(phi[:, None]), axis=0)
    )

    amp = np.abs(c1)

    # significance mask
    mask = amp > n_sigma * uncertainty

    if np.count_nonzero(mask) < 3:
        return np.nan

    phi_s = np.unwrap(np.angle(c1[mask]))
    s_use = s[mask]

    # weights = amplitude^2 gives phase precision weighting
    w_use = amp[mask]**2

    k_hat, _ = _weighted_linear_fit(s_use, phi_s, w_use)

    if not np.isfinite(k_hat) or k_hat == 0.0:
        return np.nan

    return float(2.0 * np.pi / k_hat)

def main():
    # Ideal traveling wave: y(t,s) = A cos(ω t + k s)
    A = 1.2
    f_true = 1.01
    omega = 2 * np.pi * f_true

    # Choose a target wavelength (in arclength units)
    lambda_true = 0.351
    k_true = 2 * np.pi / lambda_true  # rad / unit

    dt = 0.021
    T = 40.0
    t = np.arange(0.0, T, dt)


    # Uneven arclength sampling (still monotone)
    Ns = 80
    u = np.linspace(0.0, 1.0, Ns)
    s = u**1.6  # non-uniform spacing
    # scale to a physical length L (optional)
    L = 1.01
    s = s / s[-1] * L

    Y = A * np.cos(omega * t[:, None] - k_true * s[None, :])

    lam = estimate_lambda_from_tangent_angle_series(Y, s, power_thresh_fraction=0.01, band_fraction=0.05)

    print("=== Test: ideal traveling wave ===")

    print(f"true lambda  = {lambda_true:.6f} (arclength units)")
    print(f"est  lambda  = {lam:.6f}")

    # Ideal traveling wave: Ψ(φ,s) = A cos(φ + k s)
    A = 1.2
    lambda_true = 0.351
    k_true = 2 * np.pi / lambda_true

    Ns = 80
    s = np.linspace(0.0, 1.0, Ns)

    Nphi = 400
    phi = 2 * np.pi * np.arange(Nphi) / Nphi

    psi_avg = A * np.cos(phi[:, None] - k_true * s[None, :])

    lam = estimate_lambda_from_cycle(psi_avg, s)

    print("=== Test: cycle-based lambda ===")
    print(f"true lambda = {lambda_true:.6f}")
    print(f"est  lambda = {lam:.6f}")

if __name__ == "__main__":
    main()
