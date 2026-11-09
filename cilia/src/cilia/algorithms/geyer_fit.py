import numpy as np
import math
from scipy.integrate import trapezoid

# ============================================================
# Utilities: shifted Legendre polynomials on x in [0,1]
# (unnormalized "shifted Legendre" basis as commonly used)
# ℓ0 = 1
# ℓ1 = 2x - 1
# ℓ2 = 6x^2 - 6x + 1
# Paper uses shifted Legendre with coefficients:
#   g_n = (2n+1) * ∫_0^1 g(x) ℓ_n(x) dx
# (see Methods)  :contentReference[oaicite:2]{index=2}
# ============================================================

def shifted_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """
    Shifted Legendre ℓ_n(x) on x∈[0,1], defined by:
        ℓ_n(x) = P_n(2x - 1)
    where P_n is the standard Legendre polynomial on [-1,1].

    Works for any n >= 0.
    """
    x = np.asarray(x, dtype=float)
    t = 2.0 * x - 1.0  # map [0,1] -> [-1,1]

    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t

    Pnm2 = np.ones_like(t)  # P0
    Pnm1 = t                # P1
    for k in range(2, n + 1):
        Pn = ((2 * k - 1) * t * Pnm1 - (k - 1) * Pnm2) / k
        Pnm2, Pnm1 = Pnm1, Pn
    return Pnm1


def trapz_mean_over_s(values_s: np.ndarray) -> float:
    """Mean over arclength assuming equidistant positions."""
    v = np.asarray(values_s, dtype=float)
    return float(np.mean(v))

def legendre_coeffs_ls(g_s, nmax=2, mask=None):
    """
    Ordinary least squares fit of shifted Legendre basis.

    Fits:
        g(s) ≈ sum_{n=0}^{nmax} c_n ℓ_n(x)

    where x ∈ [0,1] equidistant
    """

    g_s = np.asarray(g_s, float)

    if mask is None:
        mask = np.ones_like(g_s, dtype=bool)
    else:
        mask = np.asarray(mask, bool)

    if np.count_nonzero(mask) < (nmax + 1):
        raise RuntimeError("Not enough points for Legendre fit.")

    x = np.linspace(0, 1, len(g_s))

    xm = x[mask]
    ym = g_s[mask]

    # Design matrix
    X = np.stack(
        [shifted_legendre(n, xm) for n in range(nmax + 1)],
        axis=1
    )

    coeffs, *_ = np.linalg.lstsq(X, ym, rcond=None)

    return {n: float(coeffs[n]) for n in range(nmax + 1)}



# ============================================================
# 1) Frequency estimate from PSD, phase-robust across s
#    - compute PSD per s (FFT along time)
#    - average PSD over s using trapz rule
# ============================================================

def estimate_f_from_tangent_angle_power(
    tangent_angles: np.ndarray,
) -> float:
    Y = np.asarray(tangent_angles, dtype=float)

    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2:
        raise ValueError("tangent_angles must be 1D or 2D (Nt or Nt×Ns).")

    Nt, Ns = Y.shape
    if Nt < 4:
        return float("nan")

    # FFT along time, for each s-column
    X = np.fft.rfft(Y, axis=0)
    psd = (np.abs(X) ** 2) / (Nt ** 2)  # consistent with your earlier convention

    # Remove DC
    psd[0, :] = 0.0

    # Average PSD over s using mean
    psd_mean = np.mean(psd, axis=1)

    k0 = int(np.argmax(psd_mean))
    if k0 == 0:
        return float("nan")

    freqs = np.fft.rfftfreq(Nt)
    return float(freqs[k0])


# ============================================================
# 2) Fundamental mode extraction at the dominant temporal frequency
#    For ψ(t,s) = a(s) cos(ω t + φ(s)):
#    - rFFT coefficient X[k0,s] ≈ (Nt/2) * a(s) * exp(i φ(s))  (bin-centered)
#    => a(s) = 2|X|/Nt,   φ(s)=arg(X)
# ============================================================
def _stringify_coeffs(coeffs: dict[int, float]) -> dict[str, float]:
    return {str(k): float(v) for k, v in coeffs.items()}


def extract_fundamental_mode_profiles(
    tangent_angles: np.ndarray,
    power_threshold_fraction: float = 0.01,  # exclude s where power < 1% of mean power
):
    Y = np.asarray(tangent_angles, dtype=float)

    if Y.ndim == 1:
        raise ValueError("Need spatial dimension (Nt×Ns) to extract a(s), φ(s).")
    if Y.ndim != 2:
        raise ValueError("tangent_angles must be 2D (Nt×Ns).")

    Nt, Ns = Y.shape
    if Nt < 4:
        raise ValueError("Need at least 4 time points.")

    # Optional: remove static (time-mean) mode per s (paper subtracts static mode n=0)
    # This makes the temporal fundamental cleaner. :contentReference[oaicite:3]{index=3}
    psi0_s = np.mean(Y, axis=0)        # <-- ADD THIS
    Yc = Y - psi0_s[None, :]

    # Find peak frequency from spatially-averaged PSD
    f0 = estimate_f_from_tangent_angle_power(Yc)
    if not np.isfinite(f0):
        raise RuntimeError("Failed to estimate frequency.")

    freqs = np.fft.rfftfreq(Nt)
    k0 = int(np.argmin(np.abs(freqs - f0)))
    if k0 == 0:
        raise RuntimeError("Peak landed at DC unexpectedly.")

    # FFT and take the complex coefficient at k0 for each s
    X = np.fft.rfft(Yc, axis=0)
    Xk = X[k0, :]  # shape (Ns,)

    # amplitude + phase profiles (bin-centered scaling)
    a_s = (2.0 / Nt) * np.abs(Xk)
    phi_s = np.angle(Xk)  # wrapped
    phi_s = np.unwrap(phi_s)

    # Exclude weak s locations by temporal power at k0
    power_s = (np.abs(Xk) ** 2) / (Nt ** 2)
    mean_power = trapz_mean_over_s(power_s)
    keep = power_s >= (power_threshold_fraction * mean_power)

    return {
        "f": float(f0),
        "k0": int(k0),
        "a_s": a_s,
        "phi_s": phi_s,
        "psi0_s": psi0_s,
        "keep_mask": keep,
        "power_s": power_s,
    }


# ============================================================
# 3) Legendre decomposition of a(s), φ(s) over normalized arclength x∈[0,1]
#    g_n = (2n+1) ∫_0^1 g(x) ℓ_n(x) dx     :contentReference[oaicite:4]{index=4}
#    For phase, set φ0 = 0 by subtracting its mean over x.
# ============================================================

def legendre_coeffs_trapz(g_s: np.ndarray, nmax: int = 2, mask: np.ndarray | None = None):
    g_s = np.asarray(g_s, dtype=float)

    if mask is None:
        mask = np.ones_like(g_s, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # work on normalized x in [0,1] equidistant
    x = np.linspace(0, 1, len(g_s))

    # apply mask by selecting points; integration uses actual (masked) coordinates
    xm = x[mask]
    gm = g_s[mask]

    coeffs = {}
    for n in range(nmax + 1):
        ln = shifted_legendre(n, xm)
        coeff = (2 * n + 1) * trapezoid(gm * ln, xm)
        coeffs[n] = float(coeff)
    return coeffs


def reconstruct_from_legendre(coeffs: dict[int, float], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for n, cn in coeffs.items():
        out += cn * shifted_legendre(int(n), x)
    return out

def _mean_over_phi_and_s(q_phi_s: np.ndarray) -> float:
    """
    Compute ⟨q⟩_{phi,s} where phi is uniform (simple mean over axis=0),
    and s is equidistant (simple mean over axis=1).
    q_phi_s has shape (Nphi, Ns).
    """
    q = np.asarray(q_phi_s, float)

    # mean over s for each phi
    mean_s_per_phi = np.mean(q, axis=1)  # shape (Nphi,)
    # mean over phi (uniform)
    return float(np.mean(mean_s_per_phi))


def compute_geyer_r2_from_cycle(
    psi_c: np.ndarray,
    *,
    a_s: np.ndarray,
    phi_s: np.ndarray,
    #phi0: float = 0.0,
    a_coeffs: dict[int, float],
    phi_coeffs: dict[int, float],
    matrix_size: int = 3,
):
    """
    Explained-variance R^2 for cycle-based Geyer decomposition,
    using psi_c = psi_avg - psi0_s as reference (static removed).

    Returns:
      - r2_full_profiles: uses full a_s, phi_s
      - r2_matrix: (matrix_size x matrix_size) numpy array using truncated coeffs:
          rows i=1..matrix_size: a uses orders 0..i-1
          cols j=1..matrix_size: phi uses orders 1..j (phi0 fixed to 0)
      - sst: total variance (mean square of psi_c)
    """
    psi_c = np.asarray(psi_c, float)
    a_s = np.asarray(a_s, float)
    phi_s = np.asarray(phi_s, float)

    if psi_c.ndim != 2:
        raise ValueError("psi_c must be (Nphi, Ns)")
    Nphi, Ns = psi_c.shape
    if a_s.shape != (Ns,) or phi_s.shape != (Ns,):
        raise ValueError("a_s / phi_s mismatch")

    # uniform phase grid
    phi = 2.0 * np.pi * np.arange(Nphi) / Nphi

    sst = _mean_over_phi_and_s(psi_c**2)
    if sst <= 0:
        return {
            "r2_full_profiles": float("nan"),
            "r2_matrix": np.full((matrix_size, matrix_size), np.nan, dtype=float),
            "sst": float(sst),
        }

    def r2_for_model(a_model_s, phi_model_s) -> float:
        phi_use = phi_model_s
        psi_hat = a_model_s[None, :] * np.cos(phi[:, None] + phi_use[None, :])
        sse = _mean_over_phi_and_s((psi_c - psi_hat) ** 2)
        return float(1.0 - sse / sst)

    # best: full profiles
    r2_full = r2_for_model(a_s, phi_s)

    # coefficient reconstructions live on x in [0,1]
    x = np.linspace(0, 1, Ns)

    phi_coeffs_all = dict(phi_coeffs)

    r2_mat = np.empty((matrix_size, matrix_size), dtype=float)

    for i in range(1, matrix_size + 1):
        a_trunc = {n: float(a_coeffs.get(n, 0.0)) for n in range(i)}  # 0..i-1
        a_rec = reconstruct_from_legendre(a_trunc, x)

        for j in range(2, matrix_size + 2):
            phi_trunc = {n: float(phi_coeffs_all.get(n, 0.0)) for n in range(j)}  # 0..j-1
            phi_rec = reconstruct_from_legendre(phi_trunc, x)

            r2_mat[i-1, j-2] = r2_for_model(a_rec, phi_rec)
    
    return {
        "r2_full_profiles": r2_full,
        "r2_matrix": r2_mat,
        "sst": float(sst),
    }

# ============================================================
# Full pipeline: from ψ(s,t) -> f, a(s), φ(s), Legendre coeffs, λ
# ============================================================

def geyer_fit_from_tangent_angle_series(
    tangent_angles: np.ndarray,
    power_threshold_fraction: float = 0.01,
    nmax: int = 2,
):
    prof = extract_fundamental_mode_profiles(
        tangent_angles=tangent_angles,
        power_threshold_fraction=power_threshold_fraction,
    )

    a_s = prof["a_s"]
    phi_s = prof["phi_s"]
    keep = prof["keep_mask"]
    psi0_s = prof["psi0_s"]

    # Phase: remove arbitrary constant (set φ0 = 0) using mean over s
    phi0 = trapz_mean_over_s(phi_s[keep])
    phi_centered = phi_s - phi0

    # Legendre coefficients up to nmax
    a_coeffs = legendre_coeffs_trapz(a_s, nmax=nmax, mask=keep)
    
    phi_coeffs = legendre_coeffs_ls(phi_centered, nmax=nmax, mask=keep)
    phi_coeffs[0] = 0.0
    phi1 = phi_coeffs.get(1, np.nan)
    lam = -math.pi / phi1

    psi0_coeffs = legendre_coeffs_trapz(psi0_s, nmax=nmax, mask=keep)

    # enforce φ0 = 0 explicitly (numerical drift)
    phi_coeffs[0] = 0.0

    # --- R^2 metrics (explained variance), ignoring static mode ---
   
    return {
        "psi0_s": psi0_s, 
        "a_s": a_s,
        "phi_s": phi_s,                 # wrapped
        "phi_centered": phi_centered,   # wrapped-ish but mean removed
        "keep_mask": keep,
        "psi0_coeffs": _stringify_coeffs(psi0_coeffs),
        "a_coeffs": _stringify_coeffs(a_coeffs),
        "phi_coeffs": _stringify_coeffs(phi_coeffs),
        "lambda": lam,
    }

def geyer_fit_from_cycle(
    psi_avg: np.ndarray,
    uncertainty: float,
    *,
    n_sigma: float = 3.0,
    nmax: int = 3,
):
    """
    Geyer-style decomposition from periodic average Ψ̄(φ,s).

    Parameters
    ----------
    psi_avg : ndarray, shape (Nphi, Ns)
        Periodic average over phase.
    uncertainty : float
        Uncertainty in psi_avg for masking.
    nmax : int
        Maximum Legendre order.

    Returns
    -------
    dict with same structure as geyer_fit_from_tangent_angle_series
    """
    psi = np.asarray(psi_avg, float)

    if psi.ndim != 2:
        raise ValueError("psi_avg must be (Nphi, Ns)")
    if psi.ndim != 2:
        raise ValueError("psi_avg must be (Nphi, Ns)")

    Nphi, Ns = psi.shape

    # uniform phase grid
    phi = 2.0 * np.pi * np.arange(Nphi) / Nphi
    # --- static mode ψ0(s) ---
    psi0_s = np.mean(psi, axis=0)
    # remove static mode
    psi_c = psi - psi0_s[None, :]
    # --- first harmonic in phase ---
    c1c = 2.0 * np.mean(psi_c * np.cos(phi[:, None]), axis=0)
    c1s = 2.0 * np.mean(psi_c * np.sin(phi[:, None]), axis=0)
    a_s = np.sqrt(c1c**2 + c1s**2)
    phi_s = -np.unwrap(np.arctan2(c1s, c1c))
    keep = a_s > (n_sigma * uncertainty)

    if np.count_nonzero(keep) < 3:
        raise RuntimeError("Too few points for Geyer fit (cycle-based).")

    # --- Legendre coefficients ---
    a_coeffs = legendre_coeffs_trapz(a_s, nmax=nmax, mask=keep)
    phi_coeffs = legendre_coeffs_ls(
        phi_s, nmax=nmax, mask=keep
    )

    psi0_coeffs = legendre_coeffs_trapz(psi0_s, nmax=nmax, mask=keep)

    # wavelength from φ1
    phi1 = phi_coeffs.get(1, float("nan"))
    lam = -math.pi / phi1

    # --- R^2 metrics (explained variance), ignoring static mode ---
    r2 = compute_geyer_r2_from_cycle(
        psi_c=psi_c,
        a_s=a_s,
        phi_s=phi_s,
        a_coeffs=a_coeffs,
        phi_coeffs=phi_coeffs,
        matrix_size=3,
    )

    return {
        "psi0_s": psi0_s,
        "a_s": a_s,
        "phi_s": phi_s,
        "keep_mask": keep,
        "psi0_coeffs": _stringify_coeffs(psi0_coeffs),
        "a_coeffs": _stringify_coeffs(a_coeffs),
        "phi_coeffs": _stringify_coeffs(phi_coeffs),
        "lambda": lam,
        "r2_full_profiles": r2["r2_full_profiles"],
        "r2_matrix": r2["r2_matrix"],
    }


# ============================================================
# Minimal main test:
# - synthesize a(s), φ(s) from known Legendre coefficients
# - generate ψ(s,t) = a(s) cos(2π f t + φ(s))
# - recover a(s), φ(s), coefficients, and λ
# ============================================================

def main():
    # Synthetic settings
    f_true = 1.0
    dt = 0.01
    T = 20.0
    t = np.arange(0.0, T, dt)
    

    # Equidistant positions in [0,1]
    Ns = 80
    s = np.linspace(0, 1, Ns)

    # Choose Legendre coefficients like the paper discussion (up to order 2)
    # a(x) = a0 ℓ0 + a1 ℓ1 + a2 ℓ2
    a0_true, a1_true, a2_true = 0.8, -0.10, 0.01

    # φ(x) = φ1 ℓ1 + φ2 ℓ2 , and φ0=0
    # For relative wavelength, set lam_true = 0.5 (in units of length)
    lam_true = 0.5
    phi1_true = -math.pi / lam_true
    phi2_true = 0.6

    a_true = (
        a0_true * shifted_legendre(0, s)
        + a1_true * shifted_legendre(1, s)
        + a2_true * shifted_legendre(2, s)
    )

    phi_true = (
        0.0 * shifted_legendre(0, s)
        + phi1_true * shifted_legendre(1, s)
        + phi2_true * shifted_legendre(2, s)
    )

    # Generate ideal traveling-wave-like tangent angle
    psi = a_true[None, :] * np.cos(2 * np.pi * f_true * t[:, None] + phi_true[None, :])

    # Decompose
    out = geyer_fit_from_tangent_angle_series(
        tangent_angles=psi,
        power_threshold_fraction=0.01,
        nmax=2,
    )

    print("=== TRUE ===")
    print(f"lambda_true= {lam_true:.6f}")
    print(f"a coeffs   = (a0,a1,a2)=({a0_true:.6f},{a1_true:.6f},{a2_true:.6f})")
    print(f"phi coeffs = (phi1,phi2)=({phi1_true:.6f},{phi2_true:.6f})")

    print("\n=== ESTIMATED ===")
    print(f"lambda_hat = {out['lambda']:.6f}")
    print("a_coeffs_hat:")
    for n in ("0", "1", "2"):
        print(f"  a{n} = {out['a_coeffs'][n]: .6f}")
    print("phi_coeffs_hat:")
    for n in ("0", "1", "2"):
        print(f"  phi{n} = {out['phi_coeffs'][n]: .6f}")

    # Quick profile-level sanity (optional): relative errors
    a_coeffs_int = {int(k): v for k, v in out["a_coeffs"].items()}
    phi_coeffs_int = {int(k): v for k, v in out["phi_coeffs"].items()}
    a_rec = reconstruct_from_legendre(a_coeffs_int, s)
    phi_rec = reconstruct_from_legendre(phi_coeffs_int, s)

    a_rms_err = np.sqrt(np.mean((a_rec - a_true) ** 2))
    phi_rms_err = np.sqrt(np.mean((phi_rec - (phi_true - np.mean(phi_true))) ** 2))

    print("\n=== PROFILE RMS ERRORS ===")
    print(f"RMS a(s) error   = {a_rms_err:.3e}")
    print(f"RMS phi(s) error = {phi_rms_err:.3e} (after removing constant offset)")

    # --- cycle-based test ---
    Nphi = 600
    phi = 2 * np.pi * np.arange(Nphi) / Nphi
    psi_avg = a_true[None, :] * np.cos(phi[:, None] + phi_true[None, :])

    out_cycle = geyer_fit_from_cycle(
        psi_avg=psi_avg,
        uncertainty=0.1,
        n_sigma=3,
        nmax=4,
    )

    print("\n=== GEYER FROM CYCLE ===")
    print(f"lambda_hat = {out_cycle['lambda']:.6f}")

    for n in ("0", "1", "2"):
        print(f"a{n} = {out_cycle['a_coeffs'][n]: .6f}, "
              f"phi{n} = {out_cycle['phi_coeffs'][n]: .6f}")


if __name__ == "__main__":
    main()
