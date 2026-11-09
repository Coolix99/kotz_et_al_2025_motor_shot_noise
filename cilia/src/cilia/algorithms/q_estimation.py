import numpy as np
from numba import njit
from scipy.signal import welch
from scipy.stats import norm
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def estimate_phase_diffusion_mle(phi, debug_plot=True):
    phi = np.unwrap(phi)

    # increments
    dphi = np.diff(phi)
    N = len(dphi)

    # MLE for omega
    omega_hat = (phi[-1] - phi[0]) / N

    residuals = dphi - omega_hat

    # MLE for D
    D_hat = np.mean(residuals**2) / 2.0
    Q_hat = omega_hat / (2 * D_hat)

    if debug_plot:
        mu_fit = np.mean(dphi)
        sigma_fit = np.std(dphi)

        x = np.linspace(mu_fit - 5*sigma_fit, mu_fit + 5*sigma_fit, 400)
        pdf = norm.pdf(x, mu_fit, sigma_fit)

        plt.figure(figsize=(6, 4))
        plt.hist(dphi, bins=60, density=True, alpha=0.6, label="increments")
        plt.plot(x, pdf, 'r', lw=2, label="Gaussian fit")
        plt.axvline(omega_hat, color='k', ls='--', label=r"$\hat{\omega}$")
        plt.xlabel(r"$\Delta\phi$")
        plt.ylabel("PDF")
        plt.title("Increment distribution (MLE diagnostics)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return Q_hat, D_hat, omega_hat

def estimate_omega_endpoint(phi, dt=1.0):
    """omega_app from endpoints (MLE for constant drift in phase diffusion model)."""
    return (phi[-1] - phi[0]) / ((len(phi) - 1) * dt)

def msd_phase_increments(phi_unwrapped, omega0, taus):
    """
    Compute MSD(τ) = <(phi(t+τ)-phi(t)-omega0*τ)^2>_{t} for a list of taus (in same units as dt).
    Returns taus (float), msd (float), n_terms used per tau.
    """
    N = len(phi_unwrapped)
    taus = np.asarray(taus, dtype=float)
    msd = np.zeros_like(taus, dtype=float)
    n_terms = np.zeros_like(taus, dtype=int)

    for i, tau in enumerate(taus):
        m = int(round(tau))
        if m <= 0:
            msd[i] = 0.0
            n_terms[i] = 0
            continue
        if m >= N:
            msd[i] = np.nan
            n_terms[i] = 0
            continue

        dphi = phi_unwrapped[m:] - phi_unwrapped[:-m]
        resid = dphi - omega0 * (m)
        msd[i] = np.mean(resid**2)
        n_terms[i] = len(resid)

    return taus, msd, n_terms

def weighted_fit_through_origin(x, y, w):
    """
    Fit y ≈ a x through origin using weights w (w ~ 1/Var(y)).
    Returns a, a_std (approx), and fitted yhat.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(y) & (w > 0) & (x > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.full_like(x, np.nan)

    # a = sum(w x y)/sum(w x^2)
    num = np.sum(w * x * y)
    den = np.sum(w * x * x)
    a = num / den

    # residual variance estimate (weighted)
    yhat = a * x
    dof = max(len(x) - 1, 1)
    s2 = np.sum(w * (y - yhat) ** 2) / dof

    # Var(a) ≈ s2 / sum(w x^2)
    a_var = s2 / den
    a_std = np.sqrt(a_var)

    return a, a_std, yhat

def weighted_fit_with_intercept(x, y, w):
    """
    Fit y ≈ a x + b using weights w (w ~ 1/Var(y)).

    Returns
    -------
    a : float
        slope
    b : float
        intercept
    a_std : float
        standard deviation of slope (approx)
    b_std : float
        standard deviation of intercept (approx)
    yhat : ndarray
        fitted values at x
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]

    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.full_like(x, np.nan)

    # weighted means
    W = np.sum(w)
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W

    # weighted covariance and variance
    Sxx = np.sum(w * (x - xbar) ** 2)
    Sxy = np.sum(w * (x - xbar) * (y - ybar))

    if Sxx == 0:
        return np.nan, np.nan, np.nan, np.nan, np.full_like(x, np.nan)

    # slope and intercept
    a = Sxy / Sxx
    b = ybar - a * xbar

    # fitted values
    yhat = a * x + b

    # weighted residual variance
    dof = max(len(x) - 2, 1)
    s2 = np.sum(w * (y - yhat) ** 2) / dof

    # parameter variances
    a_var = s2 / Sxx
    b_var = s2 * (1.0 / W + xbar ** 2 / Sxx)

    a_std = np.sqrt(a_var)
    b_std = np.sqrt(b_var)

    return a, b, a_std, b_std, yhat

def default_tau_grid(t_end, tau_min, tau_max, n=30, log=True):
    tau_min = max(tau_min, 1)
    tau_max = min(tau_max, t_end - 1)
    if tau_max <= tau_min:
        return np.array([tau_min])

    if log:
        taus = np.geomspace(tau_min, tau_max, n)
    else:
        taus = np.linspace(tau_min, tau_max, n)

    # snap to integer multiples of dt (important for indexing)
    taus = np.unique(np.maximum(1, np.round(taus).astype(int)))
    taus = taus[(taus >= tau_min) & (taus <= tau_max)]
    return taus

def error_weight_tau3(taus, t_end, eps=1e-12):
    """
    Heuristic weight ~ 1/Var(MSD(τ)).
    For phase diffusion with overlapping windows, Var(MSD) scales ~ τ^3 / t_end (for small τ),
    hence weights ~ t_end / τ^3.
    """
    taus = np.asarray(taus, float)
    return t_end / np.maximum(taus**3, eps)

def estimate_Q_msd_fixed_tau(
    phase_wrapped,
    tau_factor=0.01,
    debug_plot=True,
):
    # unwrap
    phi = np.unwrap(phase_wrapped)
    N = len(phi)
    t_end = N - 1

    # omega MLE
    omega_hat = (phi[-1] - phi[0]) / t_end

    # main tau (integer lag!)
    tau0 = int(round(tau_factor * N))
    tau0 = max(1, min(tau0, N - 1))

    # MSD at tau0
    dphi = phi[tau0:] - phi[:-tau0]
    resid = dphi - omega_hat * tau0
    msd0 = np.mean(resid**2)

    # diffusion estimate
    D_hat = msd0 / (2 * tau0)
    Q_hat = omega_hat / (2 * D_hat)

    # ---------------- debug plots ----------------
    if debug_plot:
        taus_plot = np.array([tau0])
        msd_plot = np.array([msd0])

        # automatic log-spaced diagnostic taus
        tau_min_dbg = max(1, int(0.01 * tau0))
        tau_max_dbg = min(int(10 * t_end), N - 1)

        if tau_max_dbg > tau_min_dbg:
            taus_extra = np.unique(
                np.round(
                    np.geomspace(tau_min_dbg, tau_max_dbg, 8)
                ).astype(int)
            )
        else:
            taus_extra = np.array([], dtype=int)

        taus_extra = taus_extra[(taus_extra > 0) & (taus_extra < N)]

        msd_extra = []
        for m in taus_extra:
            dphi_m = phi[m:] - phi[:-m]
            resid_m = dphi_m - omega_hat * m
            msd_extra.append(np.mean(resid_m**2))

        taus_plot = np.concatenate([taus_plot, taus_extra])
        msd_plot = np.concatenate([msd_plot, msd_extra])

        idx = np.argsort(taus_plot)
        taus_plot = taus_plot[idx]
        msd_plot = msd_plot[idx]

        # MSD vs tau
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(taus_plot, msd_plot, "o-", label="MSD(τ)")
        plt.plot(taus_plot, 2 * D_hat * taus_plot, "--", label="2Dτ (fit)")
        plt.axvline(tau0, color="k", ls=":", label=r"$\tau=%.2fT$" % tau_factor)
        plt.xlabel("τ")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r"$\langle(\Delta\phi-\omega\tau)^2\rangle$")
        plt.title("Fixed-τ MSD estimator (MLE-style)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # residual histogram (this mirrors increment MLE diagnostics)
        plt.figure(figsize=(6, 4))
        plt.hist(resid, bins=40, density=True, alpha=0.6, label="residuals")
        x = np.linspace(resid.min(), resid.max(), 300)
        plt.plot(
            x,
            norm.pdf(x, 0, np.sqrt(2 * D_hat * tau0)),
            "r-",
            label="Gaussian prediction",
        )
        plt.xlabel("residual")
        plt.ylabel("PDF")
        plt.title(f"Residuals at τ = {tau0}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    info = dict(
        omega_hat=omega_hat,
        tau0=tau0,
        msd_tau0=msd0,
        t_end=t_end,
    )

    return Q_hat, D_hat, info

def estimate_Q_msd_auto_interval(
    phase_wrapped,
    tau_min_factor=1.0,     # τ > 1 period
    tau_max_factor=0.1,     # τ < 0.1 T
    n_taus=30,
    p_gauss_min=0.05,
    debug_plot=True,
):
    """
    Automatic MSD-based estimator (robust)

    Selection criteria:
    - τ > period
    - τ < tau_max_factor * T
    - residuals approximately Gaussian
    - MSD linear in τ (robust fit with outlier rejection)
    """

    phi = np.unwrap(phase_wrapped)
    N = len(phi)
    T = N - 1

    # omega from endpoints
    omega_hat = (phi[-1] - phi[0]) / T
    period = 2 * np.pi / omega_hat

    tau_min = max(int(tau_min_factor * period), 1)
    tau_max = int(tau_max_factor * T)
    tau_max = max(tau_min + 1, tau_max)

    taus = np.unique(
        np.round(np.geomspace(tau_min, tau_max, n_taus)).astype(int)
    )
    taus = taus[taus < N]

    msd = []
    gauss_p = []

    for m in taus:
        r = (phi[m:] - phi[:-m]) - omega_hat * m
        msd.append(np.mean(r**2))
        if len(r) > 20:
            _, p = normaltest(r)
        else:
            p = 0.0
        gauss_p.append(p)

    msd = np.asarray(msd)
    gauss_p = np.asarray(gauss_p)

    # ------------------------------------------------------------
    # Gaussianity selection with fallback
    # ------------------------------------------------------------
    gauss_good = gauss_p > p_gauss_min

    if np.sum(gauss_good) < 3:
        # force-keep top 3 by p-value
        idx = np.argsort(gauss_p)[-3:]
        gauss_good = np.zeros_like(gauss_good, dtype=bool)
        gauss_good[idx] = True

    taus_sel = taus[gauss_good]
    msd_sel = msd[gauss_good]

    # ------------------------------------------------------------
    # Robust linear fit through origin with iterative outlier removal
    # ------------------------------------------------------------
    keep = np.ones(len(taus_sel), dtype=bool)

    while True:
        if np.sum(keep) <= 3:
            break

        t = taus_sel[keep]
        y = msd_sel[keep]

        # unweighted fit through origin for outlier detection
        slope = np.sum(t * y) / np.sum(t**2)
        resid = y - slope * t
        sigma = np.std(resid)

        if sigma == 0:
            break

        bad = np.abs(resid) > 2 * sigma

        if not np.any(bad):
            break

        # remove worst outlier only
        worst = np.argmax(np.abs(resid))
        keep[np.where(keep)[0][worst]] = False

    taus_fit = taus_sel[keep]
    msd_fit = msd_sel[keep]

    if len(taus_fit) < 3:
        return np.nan, np.nan, dict(reason="insufficient τ after outlier rejection")

    # ------------------------------------------------------------
    # Final weighted fit (as before)
    # ------------------------------------------------------------
    w = T / np.maximum(taus_fit**3, 1e-12)
    # slope2D, _, _ = weighted_fit_through_origin(taus_fit, msd_fit, w)
    slope2D,b, _, _,_ = weighted_fit_with_intercept(taus_fit, msd_fit, w)
    D_hat = slope2D / 2
    Q_hat = omega_hat / (2 * D_hat)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    if debug_plot:
        plt.figure(figsize=(6.8, 4.2))
        plt.loglog(taus, msd, "o-", label="MSD")
        plt.loglog(taus_fit, msd_fit, "o", label="accepted τ")
        plt.loglog(taus_fit, slope2D * taus_fit+b, "--", label="fit")
        plt.xlabel("τ")
        plt.ylabel("MSD")
        plt.title("Auto-selected MSD interval (robust)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6.8, 4.2))
        plt.semilogx(taus, gauss_p, "o-")
        plt.axhline(p_gauss_min, color="r", ls="--", label="Gaussian cutoff")
        plt.xlabel("τ")
        plt.ylabel("normaltest p-value")
        plt.title("Gaussianity of residuals")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return Q_hat, D_hat, dict(
        omega_hat=omega_hat,
        taus=taus,
        msd=msd,
        gauss_p=gauss_p,
        taus_fit=taus_fit,
        T=T,
    )

def estimate_Q_msd_bridge_corrected_full_range(
    phi,
    tau_min=None,
    tau_max=None,
    n_taus=60,
    debug_plot=True,
):
    """
    MSD-based estimator using full τ range with proper Brownian-bridge correction.

    Assumes ω is estimated from endpoints → MSD(T) = 0 constraint.
    """
    phi = np.unwrap(phi)
    N = len(phi)
    T = N - 1

    # ω from endpoints → induces bridge
    omega_used = estimate_omega_endpoint(phi)

    if tau_min is None:
        tau_min = 1
    if tau_max is None:
        tau_max = int(0.95 * T)

    taus = default_tau_grid(T, tau_min, tau_max, n=n_taus, log=True)
    taus, msd_raw, n_terms = msd_phase_increments(phi, omega_used, taus)

    # ------------------------------------------------------------
    # Bridge correction: divide out (1 - τ/T)
    # ------------------------------------------------------------
    bridge_factor = 1.0 - taus / T
    valid = (bridge_factor > 0) & np.isfinite(msd_raw)

    taus_fit = taus[valid]
    y_fit = msd_raw[valid] / bridge_factor[valid]

    # ------------------------------------------------------------
    # Correct bridge-aware weights
    # Var(MSD) ~ τ^3 (T-τ)^3 / T^2
    # ------------------------------------------------------------
    eps = 1e-12
    #w = (T**2) / np.maximum(taus_fit**3 * (T - taus_fit)**3, eps)
    w = (T**2) / np.maximum(taus_fit**3 , eps) # we have to rethink how error scales, big tau have less statistics

    # ------------------------------------------------------------
    # Fit through origin (intercept MUST be zero)
    # ------------------------------------------------------------
    slope2D, slope_std, yhat = weighted_fit_through_origin(
        taus_fit, y_fit, w
    )

    D_hat = slope2D / 2.0
    Q_hat = (
        omega_used / (2.0 * D_hat)
        if np.isfinite(D_hat) and D_hat > 0
        else np.nan
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    if debug_plot:
        plt.figure(figsize=(6.8, 4.2))
        plt.loglog(taus, msd_raw, "o", alpha=0.4, label="raw MSD")
        plt.loglog(
            taus_fit,
            2 * D_hat * taus_fit * (1 - taus_fit / T),
            "-",
            label="bridge fit",
        )
        plt.xlabel("τ")
        plt.ylabel("MSD")
        plt.title("Bridge-corrected MSD fit")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6.8, 4.2))
        plt.loglog(taus_fit, y_fit, "o", label="MSD / (1-τ/T)")
        plt.loglog(
            taus_fit,
            slope2D * taus_fit,
            "-",
            label="linearized fit",
        )
        plt.xlabel("τ")
        plt.ylabel("corrected MSD")
        plt.title("Linearized bridge MSD")
        plt.legend()
        plt.tight_layout()
        plt.show()

    info = dict(
        omega_used=omega_used,
        taus=taus,
        msd_raw=msd_raw,
        taus_fit=taus_fit,
        msd_linearized=y_fit,
        slope_2D=slope2D,
        slope_std=slope_std,
        T=T,
    )

    return Q_hat, D_hat, omega_used, info

def estimate_Q_from_exp_phase_correlation(
    phase_wrapped,
    tau_min=None,
    tau_max=None,
    n_taus=60,
    debug_plot=True,
):
    """
    Fits C(τ) = A exp(-D τ) + C_inf
    """

    phi = np.unwrap(phase_wrapped)
    N = len(phi)
    T = N - 1

    
    omega_used = (phi[-1] - phi[0]) / T
    if omega_used<0:
        return np.nan, np.nan, np.nan, np.nan
    
    if tau_min is None:
        tau_min = 1
    if tau_max is None:
        tau_max = max(3,int(0.1 * T))
    
    taus = np.unique(
        np.round(np.geomspace(tau_min, tau_max, n_taus)).astype(int)
    )
    taus = taus[(taus > 0) & (taus < N)]

    C = []
    for m in taus:
        dphi = phi[m:] - phi[:-m]
        dphi -= omega_used * m
        C.append(np.abs(np.mean(np.exp(1j * dphi))))
    C = np.asarray(C)

    # model with noise floor
    def model(t, A, D, Cinf):
        return A * np.exp(-D * t) + Cinf

    # initial guesses 
    p0 = (1.0, omega_used, np.min(C))
    bounds = ([0, 0, 0], [1.5, np.inf, 1.0])
   
    popt, pcov = curve_fit(
        model, taus, C, p0=p0, bounds=bounds, maxfev=10000
    )

    A_hat, D_hat, Cinf_hat = popt
    Q_hat = omega_used / (2 * D_hat)

    # ---------------- diagnostics ----------------
    if debug_plot:
        plt.figure(figsize=(6.8, 4.2))
        plt.loglog(taus, C, "o", label="data")
        tfit = np.linspace(taus.min(), taus.max(), 400)
        plt.loglog(tfit, model(tfit, *popt), "-", label="fit")
        plt.axhline(Cinf_hat, color="k", ls=":", label="noise floor")
        plt.xlabel("τ")
        plt.ylabel("C(τ)")
        plt.title("Phase correlation with noise floor")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6.8, 4.2))
        plt.plot(taus, C - Cinf_hat, "o", label="C - C∞")
        plt.plot(tfit, A_hat * np.exp(-D_hat * tfit), "-", label="exp(-Dτ)")
        plt.yscale("log")
        plt.xlabel("τ")
        plt.ylabel("C(τ) - C∞")
        plt.title("Noise-floor corrected correlation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    info = dict(
        omega_used=omega_used,
        A=A_hat,
        Cinf=Cinf_hat,
        taus=taus,
        C=C,
        cov=pcov,
    )

    return Q_hat, D_hat, omega_used, info

def _tangent_angles_to_1d_signal(tangent_angles, center=True):
    arr = np.asarray(tangent_angles)
    if arr.ndim == 1:
        x = arr
    elif arr.ndim == 2:
        x = np.mean(arr, axis=1)
    else:
        raise ValueError("tangent_angles must be 1D or 2D array")

    x = x.astype(float, copy=False)
    if center:
        x = x - np.mean(x)
    return x

def estimate_Q_from_tangent_angle_lorentz_width(
    tangent_angles,
    window="hann",
    debug_plot=True,
):
    """
    Estimate Q from Lorentzian linewidth of the tangent-angle power spectrum.
    Assumes dt = 1, so frequencies are in cycles per sample.
    """
    x = _tangent_angles_to_1d_signal(tangent_angles, center=True)
    if x.size < 2:
        return np.nan, np.nan, np.nan, dict(reason="too_few_samples")

    nperseg = max(8, x.size // 4)
    nperseg = min(nperseg, x.size)

    freqs, Pxx = welch(
        x,
        fs=1.0,
        window=window,
        nperseg=nperseg,
        scaling="density",
    )

    if freqs.size < 2:
        return np.nan, np.nan, np.nan, dict(reason="no_spectrum")

    omega = 2 * np.pi * freqs

    # locate peak (ignore DC)
    idx0 = np.argmax(Pxx[1:]) + 1
    omega0_hat = omega[idx0]

    # fit Lorentzian near peak
    fit_mask = (omega > 0.3 * omega0_hat) & (omega < 1.5 * omega0_hat)
    omega_fit = omega[fit_mask]
    P_fit = Pxx[fit_mask]

    if omega_fit.size < 3:
        return np.nan, np.nan, omega0_hat, dict(reason="insufficient_fit_window")

    lower_C = np.min(P_fit)
    upper_C = np.max(P_fit)

    p0 = (
        np.max(P_fit),
        omega0_hat,
        0.1 * omega0_hat if omega0_hat > 0 else 0.1,
        lower_C,
    )

    bounds = (
        [0.0, 0.0, 0.0, 0.0],        # lower bounds
        [np.inf, np.inf, np.inf, upper_C]  # upper bounds
    )

    def lorentzian_log(w, A, w0, D, C):
        model = A * D / ((w - w0)**2 + D**2) + C
        return np.log(model)

    eps = 1e-12  # avoid log(0)
    ydata = np.log(np.maximum(P_fit, eps))

    popt, pcov = curve_fit(
        lorentzian_log,
        omega_fit,
        ydata,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    A_hat, w0_hat, D_hat, C_hat = popt

    Q_hat = w0_hat / (2 * D_hat) if D_hat != 0 else np.nan

    if debug_plot:
        def lorentzian(w, A, w0, D, C):
            return A * D / ((w - w0)**2 + D**2) + C
        plt.figure(figsize=(6.8, 4.2))
        plt.semilogy(omega, Pxx, label="PSD")
        plt.semilogy(
            omega_fit,
            lorentzian(omega_fit, *popt),
            "-",
            label="Lorentzian fit",
        )
        plt.axvline(w0_hat, color="k", ls="--", label="Ï‰â‚€")
        plt.xlabel("Ï‰")
        plt.ylabel("PSD")
        plt.title("Tangent-angle Lorentzian linewidth")
        plt.legend()
        plt.tight_layout()
        plt.show()

    info = dict(
        omega0_hat=w0_hat,
        D_hat=D_hat,
        A=A_hat,
        C=C_hat,
        cov=pcov,
    )

    return Q_hat, D_hat, w0_hat, info

def estimate_Q_from_fft_peak(
    phase_wrapped,
    window="hann",
    debug_plot=True,
):
    """
    Estimate Q from Lorentzian linewidth of cos(phi(t)).
    Uses log-Lorentzian fit for statistical robustness.
    """

    phi = np.unwrap(phase_wrapped)
    x = np.cos(phi)

    if len(x) < 8:
        return np.nan, np.nan, np.nan, dict(reason="too_few_samples")

    nperseg = max(8, len(x) // 4)

    freqs, Pxx = welch(
        x,
        fs=1.0,
        window=window,
        nperseg=nperseg,
        scaling="density",
    )

    if len(freqs) < 5:
        return np.nan, np.nan, np.nan, dict(reason="no_spectrum")

    omega = 2 * np.pi * freqs

    # locate peak (ignore DC)
    idx0 = np.argmax(Pxx[1:]) + 1
    omega0_hat = omega[idx0]

    # fit window around peak
    fit_mask = (omega > 0.5 * omega0_hat) & (omega < 1.5 * omega0_hat)
    omega_fit = omega[fit_mask]
    P_fit = Pxx[fit_mask]

    if len(P_fit) < 5:
        return np.nan, np.nan, omega0_hat, dict(reason="insufficient_fit_window")

    lower_C = np.min(P_fit)
    upper_C = np.max(P_fit)

    # initial guesses
    p0 = (
        np.max(P_fit),            # A
        omega0_hat,               # w0
        0.1 * omega0_hat,         # D
        lower_C,                  # C
    )

    bounds = (
        [0.0, 0.0, 0.0, 0.0],
        [np.inf, np.inf, np.inf, upper_C],
    )

    # log-Lorentzian
    def lorentzian_log(w, A, w0, D, C):
        model = A * D / ((w - w0) ** 2 + D ** 2) + C
        return np.log(model)

    eps = 1e-12
    ydata = np.log(np.maximum(P_fit, eps))

    try:
        popt, pcov = curve_fit(
            lorentzian_log,
            omega_fit,
            ydata,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
    except Exception:
        return np.nan, np.nan, omega0_hat, dict(reason="fit_failed")

    A_hat, w0_hat, D_hat, C_hat = popt

    Q_hat = w0_hat / (2 * D_hat) if D_hat > 0 else np.nan

    # --------- diagnostics ----------
    if debug_plot:
        def lorentzian(w, A, w0, D, C):
            return A * D / ((w - w0) ** 2 + D ** 2) + C

        plt.figure(figsize=(6.8, 4.2))
        plt.semilogy(omega, Pxx, label="PSD")
        plt.semilogy(
            omega_fit,
            lorentzian(omega_fit, *popt),
            "-",
            label="Lorentzian fit",
        )
        plt.axvline(w0_hat, color="k", ls="--", label="ω₀")
        plt.xlabel("ω")
        plt.ylabel("PSD")
        plt.title("FFT-based linewidth estimator (log-fit)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    info = dict(
        omega0_hat=w0_hat,
        D_hat=D_hat,
        A=A_hat,
        C=C_hat,
        cov=pcov,
    )

    return Q_hat, D_hat, w0_hat, info

import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize_scalar

def estimate_ou_phase_ml(
    phi_meas,
    debug=True,
    tau_bounds=(1e-2, 1e2),
    max_iter=50,
):
    """
    ML estimation for OU-driven phase model.
    
    Parameters
    ----------
    phi_meas : array (N,)
        Unwrapped measured phase
    debug : bool
        Print debugging info
    tau_bounds : tuple
        Bounds for tau (in linear space)
    
    Returns
    -------
    dict with estimates
    """
    phi_meas=np.unwrap(phi_meas)
    y = np.asarray(phi_meas)
    N = len(y)

    H = np.array([[1.0, 0.0]])

    # Initial weak prior
    m0 = np.array([y[0], 0.0])
    P0 = np.diag([10.0, 10.0])

    best = {}

    def kalman_filter_smoother(A, Q, R, omega0, tau, a):
        """Run KF + RTS smoother and return moments + loglik"""

        m_pred = np.zeros((N, 2))
        P_pred = np.zeros((N, 2, 2))
        m_filt = np.zeros((N, 2))
        P_filt = np.zeros((N, 2, 2))

        m_filt[0] = m0
        P_filt[0] = P0

        loglik = 0.0

        for k in range(1, N):
            c = np.array([
                (1 - tau * (1 - a)) * omega0,
                (1 - a) * omega0,
            ])

            m_pred[k] = A @ m_filt[k-1] + c
            P_pred[k] = A @ P_filt[k-1] @ A.T + Q

            S = (H @ P_pred[k] @ H.T + R)[0, 0]   # <-- scalar
            v = y[k] - (H @ m_pred[k])[0]        # <-- scalar

            K = (P_pred[k] @ H.T)[:, 0] / S      # shape (2,)
            m_filt[k] = m_pred[k] + K * v
            P_filt[k] = P_pred[k] - np.outer(K, H @ P_pred[k])

            loglik += -0.5 * (np.log(2*np.pi*S) + v**2 / S)


        # RTS smoother
        m_smooth = m_filt.copy()
        P_smooth = P_filt.copy()
        P_cross = np.zeros((N, 2, 2))

        for k in reversed(range(1, N)):
            C = P_filt[k-1] @ A.T @ inv(P_pred[k])
            m_smooth[k-1] += C @ (m_smooth[k] - m_pred[k])
            P_smooth[k-1] += C @ (P_smooth[k] - P_pred[k]) @ C.T
            P_cross[k] = P_smooth[k] @ C.T

        return m_smooth, P_smooth, P_cross, loglik

    def profile_loglik(log_tau):
    
        tau = np.exp(log_tau)
        a = np.exp(-1 / tau)

        # === analytic OU covariance pieces ===
        q_omega = tau * (1 - a**2)
        q_I = 2 * tau**2 * (
            1 - 2 * tau * (1 - a) + 0.5 * tau * (1 - a**2)
        )
        q_Iomega = tau**2 * (1 - a)**2

        # === transition matrix ===
        A = np.array([[1.0, tau * (1 - a)],
                      [0.0, a]])

        # === EM outer loop ===
        omega0 = np.mean(np.diff(y)) 
        D_omega = 1e-3
        Qval = 100.0
        sigma2 = 1e-3

        for _ in range(3):  # inner EM refinement
            Q = np.array([
                [D_omega * q_I + omega0  / Qval, D_omega * q_Iomega],
                [D_omega * q_Iomega, D_omega * q_omega],
            ])
            R = np.array([[sigma2]])

            m, P, Pc, ll = kalman_filter_smoother(A, Q, R, omega0, tau, a)

            # === sufficient statistics ===
            Ex = m
            # Exx = P + np.einsum("ni,nj->nij", m, m)
            # Exx1 = Pc + np.einsum("ni,nj->nij", m[1:], m[:-1])

            # === update omega0 ===
            num = 0.0
            den = 0.0
            for k in range(1, N):
                num += Ex[k,1] - a * Ex[k-1,1]
                den += (1 - a)
            omega0 = num / den

            # === update D_omega ===
            s = 0.0
            for k in range(1, N):
                diff = Ex[k,1] - a*Ex[k-1,1] - (1-a)*omega0
                s += diff**2 + P[k,1,1]
            D_omega = s / ((N-1) * tau * (1 - a**2))

            # === update Q ===
            phase_var = 0.0
            for k in range(1, N):
                diff = Ex[k,0] - Ex[k-1,0] - tau*(1-a)*Ex[k-1,1] - (1-tau*(1-a))*omega0
                phase_var += diff**2 + P[k,0,0]
            Qval = omega0  / (phase_var / (N-1))

            # === update sigma ===
            resid = y - Ex[:,0]
            sigma2 = np.mean(resid**2 + P[:,0,0])

        if debug:
            print(f"log_tau={log_tau:.3e}, tau={tau:.3e}, ll={ll:.2e}, "
                  f"omega0={omega0:.3e}, Dω={D_omega:.3e}, Q={Qval:.1e}, σ={np.sqrt(sigma2):.3e}")

        # store last estimates (will correspond to current tau)
        best.clear()
        best.update(
            omega0=omega0,
            D_omega=D_omega,
            Q=Qval,
            sigma_phi=np.sqrt(sigma2),
            loglik=ll,
        )

        return -ll # type: ignore

    res = minimize_scalar(
        profile_loglik,
        bounds=(np.log(tau_bounds[0]), np.log(tau_bounds[1])),
        method="bounded",
        options=dict(maxiter=max_iter),
    )

    log_tau_hat = res.x # type: ignore
    tau_hat = np.exp(log_tau_hat)

    profile_loglik(log_tau_hat)  # final debug print

    return dict(
        tau=tau_hat,
        log_tau=log_tau_hat,
        omega0=best["omega0"],
        D_omega=best["D_omega"],
        Q=best["Q"],
        sigma_phi=best["sigma_phi"],
        loglik=best["loglik"],
        success=res.success, # type: ignore
    )



# ------------------------------------------------------------
# Simple simulator (matches assumptions used in estimator)
# ------------------------------------------------------------

def simulate_ou_phase(
    N=2000,
    dt=0.1,
    omega0=2.0,
    tau=3.0,
    D_omega=0.05,
    Q=100.0,
    sigma_phi=0.05,
    seed=0,
):
    """
    Simulate OU-driven phase with constant phase diffusion
    """
    rng = np.random.default_rng(seed)

    omega = np.zeros(N)
    phi = np.zeros(N)

    omega[0] = omega0
    phi[0] = 0.0

    D_phi = omega0 / (2 * Q)

    for k in range(1, N):
        omega[k] = (
            omega[k-1]
            + (omega0 - omega[k-1]) * dt / tau
            + np.sqrt(2 * D_omega * dt) * rng.standard_normal()
        )

        phi[k] = (
            phi[k-1]
            + omega[k] * dt
            + np.sqrt(2 * D_phi * dt) * rng.standard_normal()
        )

    # measurement noise
    phi_meas = phi + sigma_phi * rng.standard_normal(N)

    return phi, phi_meas, omega


# ------------------------------------------------------------
# Minimal main
# ------------------------------------------------------------

def main2():
    # --- true parameters ---
    true_params = dict(
        omega0=2.0,
        tau=3.0,
        D_omega=0.05,
        Q=120.0,
        sigma_phi=0.05,
    )

    dt = 0.1
    N = 3000

    # --- simulate ---
    phi_true, phi_meas, omega = simulate_ou_phase(
        N=N,
        dt=dt,
        **true_params, # type: ignore
    )

    # unwrap for safety (even though sim is continuous)
    phi_meas = np.unwrap(phi_meas)

    # --- estimate ---
    print("\nRunning ML estimation...\n")

    estimates = estimate_ou_phase_ml(
        phi_meas=phi_meas,
        debug=True,
        tau_bounds=(0.2, 20.0),
    )

    # --- report ---
    print("\n=== RESULTS ===")

    print(f"tau:")
    print(f"  true      = {true_params['tau']:.3f}")
    print(f"  estimated = {estimates['tau']:.3f}")

    print(f"\nomega0:")
    print(f"  true      = {true_params['omega0']:.3f}")
    print(f"  estimated = {estimates['omega0']:.3f}")

    print(f"\nD_omega:")
    print(f"  true      = {true_params['D_omega']:.4f}")
    print(f"  estimated = {estimates['D_omega']:.4f}")

    print(f"\nQ:")
    print(f"  true      = {true_params['Q']:.1f}")
    print(f"  estimated = {estimates['Q']:.1f}")

    print(f"\nsigma_phi:")
    print(f"  true      = {true_params['sigma_phi']:.4f}")
    print(f"  estimated = {estimates['sigma_phi']:.4f}")

    print(f"\nlog-likelihood = {estimates['loglik']:.2f}")
    print(f"success        = {estimates['success']}")






def main():
    f = 10              # samples per 2π
    Q_true = 10.0
    T = 5000

    omega0 = 2 * np.pi / f
    D_true = omega0 / (2 * Q_true)

    rng = np.random.default_rng(0)
    noise = np.sqrt(2 * D_true) * rng.normal(size=T)

    @njit
    def generate_phase_process(T, omega, noise):
        phase = np.zeros(T)
        for i in range(1, T):
            phase[i] = phase[i - 1] + omega + noise[i]
        return phase

    phi = generate_phase_process(T, omega0, noise)

    print(f"True: omega0={omega0:.6f}, D={D_true:.6f}, Q={Q_true:.3f}")

    Q1, D1, w1 = estimate_phase_diffusion_mle(phi, debug_plot=True)
    print("Increment MLE:", Q1, "D=", D1, "omega_hat=", w1)

    Q1, D1, info1 = estimate_Q_msd_fixed_tau(phi, debug_plot=True)
    print("MSD fixed :", "Q=", Q1, "D=", D1)

    Q2, D2, info2 = estimate_Q_msd_auto_interval(phi, debug_plot=True)
    print("MSD auto interval:", "Q=", Q2, "D=", D2)

    Q3, D3, w3, info3 = estimate_Q_msd_bridge_corrected_full_range(phi, debug_plot=True)
    print("MSD bridge-corrected full range:", "Q=", Q3, "D=", D3, "omega_used=", w3)

    Q4, D4, w4, info4 = estimate_Q_from_exp_phase_correlation(phi, debug_plot=True)
    print("Correlation exp-fit:", "Q=", Q4, "D=", D4, "omega_used=", w4)

    Q_fft, D_fft, omega_fft, info_fft = estimate_Q_from_fft_peak(phi,debug_plot=True)
    print(f"FFT peak-width: Q={Q_fft:.3f}, D={D_fft:.6f}, omega_hat={omega_fft:.6f}")


if __name__ == "__main__":
    # main()
    main2()
