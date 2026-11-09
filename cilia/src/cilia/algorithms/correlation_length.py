import numpy as np


def get_correlation_length(
    phase: np.ndarray,
    *,
    ds_norm_lo: float = 0.1,
    ds_norm_hi: float = 0.5,
    eps: float = 1e-12,
):
    """
    Estimate spatial correlation length ξ from phase field.

    Parameters
    ----------
    phase : (Nt, Ns)
    ds_norm_lo, ds_norm_hi : float
        Fit window in normalized arc length Δs / Ns

    Returns
    -------
    xi : float
        Correlation length (in units of s, NOT normalized)
    aux : dict
        Debug info
    """

    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 2:
        raise ValueError("phase must be (Nt, Ns)")
    # import matplotlib.pyplot as plt
    # plt.imshow(phase.T, aspect="auto")
    # plt.show()

    Nt, Ns = phase.shape

    if Nt < 5 or Ns < 10:
        return np.nan, {"too_small": True}

    max_ds = Ns // 2
    corr = np.zeros(max_ds)

    for ds in range(max_ds):
        if ds == 0:
            corr[ds] = 1.0
            continue
        # phase difference
        dphi = phase[:, :-ds] - phase[:, ds:]
        # complex exponential
        exp_term = np.exp(1j * dphi)
        # average over time 
        time_avg = np.mean(exp_term, axis=0)   # shape: (Ns - ds,)
        # magnitude
        mag = np.abs(time_avg)
        # average over space
        corr[ds] = np.mean(mag)

    ds_axis = np.arange(max_ds)
    ds_norm = ds_axis / float(Ns)

    #  fit exponential decay
    mask = (
        (ds_norm >= ds_norm_lo)
        & (ds_norm <= ds_norm_hi)
        & (corr > 0)
        & np.isfinite(corr)
    )

    x = ds_norm[mask]
    y = corr[mask]

    if x.size < 2:
        return np.nan, {"too_small": True}

    ly = np.log(y + eps)

    try:
        b, a = np.polyfit(x, ly, 1)
    except Exception:
        return np.nan, {"fit_failed": True}

    if b >= 0:
        return np.nan, {"non_decaying": True}


    # correlation length
    xi_norm = -1.0 / b        # in units of L
    xi = xi_norm * Ns         # convert to absolute s

    aux = {
        "xi_norm": xi_norm,
        "slope": b,
        "intercept": a,
        "ds_axis": ds_axis,
        "corr": corr,
    }

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # # --- left: correlation ---
    # ax = axes[0]
    # ax.plot(ds_norm, corr, label="C(ds)")
    # ax.axvspan(ds_norm_lo, ds_norm_hi, alpha=0.2, label="fit window")

    # ax.set_xlabel(r"$\Delta s / L$")
    # ax.set_ylabel("Correlation")
    # ax.set_title("Correlation function")
    # ax.grid(alpha=0.3)
    # ax.legend()

    # # --- right: log fit ---
    # ax = axes[1]

    # ax.plot(ds_norm, np.log(corr + eps), ".", alpha=0.3, label="log data")

    # # highlight fit region
    # ax.plot(x, ly, "o", label="fit points")

    # # fitted line
    # x_fit = np.linspace(x.min(), x.max(), 100)
    # y_fit = a + b * x_fit
    # ax.plot(x_fit, y_fit, "-", label=f"fit: slope={b:.3f}")

    # ax.axvspan(ds_norm_lo, ds_norm_hi, alpha=0.2)

    # ax.set_xlabel(r"$\Delta s / L$")
    # ax.set_ylabel(r"$\log C$")
    # ax.set_title(f"Log-fit (xi ≈ {xi:.2f})")
    # ax.grid(alpha=0.3)
    # ax.legend()

    # plt.tight_layout()
    # plt.show()


    return float(xi), aux


import numpy as np
import matplotlib.pyplot as plt


def get_correlation_fluctuation(
    phase: np.ndarray,
    amplitude: np.ndarray,
    f: float,
    dt: float,
    *,
    s_lo: float = 0.1,
    s_hi: float = 0.9,
    eps: float = 1e-12,
    debug_plot: bool = True,
):
    """
    Analyze fluctuations of instantaneous wavelength and amplitude.

    Parameters
    ----------
    phase : ndarray, shape (Nt, Ns)
        Local phase field.
    amplitude : ndarray, shape (Nt, Ns)
        Local amplitude field.
    s_lo, s_hi : float
        Relative spatial interval in [0, 1] used for the wavelength fit
        and amplitude analysis.
    eps : float
        Small regularization for divisions.
    debug_plot : bool
        If True, make diagnostic plots.

    Returns
    -------
    result : dict
        Contains:
            - "lambda_t" : instantaneous normalized wavelength lambda/L
            - "mean_a_t" : mean amplitude in [s_lo, s_hi]
            - "lambda_autocorr"
            - "mean_a_autocorr"
            - "lambda_mean_a_crosscorr"
            - "amp_corr_dt_ds"
            - "lags_t"
            - "lags_s"
            - "fit_mask"
            - "s"
    """
    phase = np.asarray(phase, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)

    if phase.ndim != 2:
        raise ValueError("phase must be (Nt, Ns)")
    if amplitude.ndim != 2:
        raise ValueError("amplitude must be (Nt, Ns)")
    if phase.shape != amplitude.shape:
        raise ValueError("phase and amplitude must have same shape")

    Nt, Ns = phase.shape
    if Nt < 5 or Ns < 10:
        return {
            # signals
            "lambda_t": np.full(Nt, np.nan),
            "mean_a_t": np.full(Nt, np.nan),

            # correlations (match new sparse lag structure)
            "lambda_autocorr": np.full(1, np.nan),
            "mean_a_autocorr": np.full(1, np.nan),
            "lambda_mean_a_crosscorr": np.full(1, np.nan),

            # 2D correlation
            "amp_corr_dt_ds": np.full((1, 1), np.nan),

            # lag definitions (minimal consistent placeholders)
            "lags_dt_idx": np.array([0]),
            "lags_t_phys": np.array([0.0]),
            "lags_t_rel": np.array([0.0]),

            "lags_s_idx": np.array([0]),
            "lags_s_rel": np.array([0.0]),

            # meta
            "fit_mask": np.zeros(Ns, dtype=bool),
            "s": np.linspace(0.0, 1.0, Ns),

            "too_small": True,
        }
    s = np.linspace(0.0, 1.0, Ns)
    fit_mask = (s >= s_lo) & (s <= s_hi)

    if np.count_nonzero(fit_mask) < 4:
        raise ValueError("Too few spatial points inside [s_lo, s_hi]")

    s_fit = s[fit_mask]
    phase_fit = phase[:, fit_mask]
    amp_fit = amplitude[:, fit_mask]


    # -------------------------------------------------
    # instantaneous wavelength lambda(t)
    # -------------------------------------------------
    # unwrap along s for each time
    phase_unwrap_fit = np.unwrap(phase_fit, axis=1)

    lambda_t = np.full(Nt, np.nan)
    slope_t = np.full(Nt, np.nan)

    for t in range(Nt):
        y = phase_unwrap_fit[t]
        good = np.isfinite(y)
        if np.count_nonzero(good) < 4:
            continue

        p = np.polyfit(s_fit[good], y[good], deg=1)
        slope = p[0]
        slope_t[t] = slope

        # phase ~ k*s + b, with s in [0,1]
        # normalized wavelength lambda/L = 2π / |k|
        if np.isfinite(slope) and abs(slope) > eps:
            lambda_t[t] = 2.0 * np.pi / abs(slope)

    # -------------------------------------------------
    # mean amplitude a(t)
    # -------------------------------------------------
    mean_a_t = np.full(Nt, np.nan)
    for t in range(Nt):
        vals = amp_fit[t]
        good = np.isfinite(vals)
        if np.any(good):
            mean_a_t[t] = np.mean(vals[good])

    # -------------------------------------------------
    # 🔥 remove outliers (joint mask)
    # -------------------------------------------------
    good = np.isfinite(lambda_t) & np.isfinite(mean_a_t)

    if np.count_nonzero(good) > 10:
        lam_mean = np.nanmean(lambda_t[good])
        lam_std = np.nanstd(lambda_t[good])

        a_mean = np.nanmean(mean_a_t[good])
        a_std = np.nanstd(mean_a_t[good])

        mask_good = (
            good
            & (np.abs(lambda_t - lam_mean) < 4 * lam_std)
            & (np.abs(mean_a_t - a_mean) < 4 * a_std)
        )

        lambda_t[~mask_good] = np.nan
        mean_a_t[~mask_good] = np.nan

    # -------------------------------------------------
    # 🔥 TIME LAGS (include zero!)
    # -------------------------------------------------
    if not np.isfinite(f) or f <= 0:
        target_dt_idx = np.arange(min(Nt, 100))
    else:
        T = 1.0 / f
        target_dt_phys = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3]) * T
        target_dt_idx = np.round(target_dt_phys / dt).astype(int)

        target_dt_idx = target_dt_idx[target_dt_idx < Nt]
        target_dt_idx = np.unique(target_dt_idx)

    lags_t_phys = target_dt_idx * dt
    lags_t_rel = lags_t_phys * f   # 🔥 THIS is Δt/T

    # -------------------------------------------------
    # 🔥 correlations ONLY at physical lag times
    # -------------------------------------------------
    def _corr_at_lags(x, y=None, lag_idx=None):
        x = np.asarray(x, float)
        if y is None:
            y = x

        out = np.full(len(lag_idx), np.nan)

        for i, dt_idx in enumerate(lag_idx):

            # 🔥 zero lag special case
            if dt_idx == 0:
                good = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(good) < 5:
                    continue

                a = x[good] - np.mean(x[good])
                b = y[good] - np.mean(y[good])

                denom = np.std(a) * np.std(b)
                if denom > eps:
                    out[i] = np.mean(a * b) / denom
                continue

            x0 = x[:-dt_idx]
            x1 = y[dt_idx:]

            good = np.isfinite(x0) & np.isfinite(x1)
            if np.count_nonzero(good) < 5:
                continue

            a = x0[good] - np.mean(x0[good])
            b = x1[good] - np.mean(x1[good])

            denom = np.std(a) * np.std(b)
            if denom > eps:
                out[i] = np.mean(a * b) / denom

        return out


    lambda_autocorr = _corr_at_lags(lambda_t, lag_idx=target_dt_idx)
    mean_a_autocorr = _corr_at_lags(mean_a_t, lag_idx=target_dt_idx)
    lambda_mean_a_crosscorr = _corr_at_lags(lambda_t, mean_a_t, lag_idx=target_dt_idx)

    # -------------------------------------------------
    # amplitude correlation C_a(dt, ds)
    # -------------------------------------------------
    amp_sub = amp_fit.copy()
    amp_sub = amp_sub - np.nanmean(amp_sub)

    var_a = np.nanmean(amp_sub**2)

    Ns_fit = len(s_fit)

    # ----------------------------------------
    # 🔥 SPATIAL LAGS (relative)
    # ----------------------------------------
    target_ds_rel = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target_ds_idx = np.round(target_ds_rel * Ns_fit).astype(int)

    lags_s = np.unique(
        np.concatenate([[-d for d in target_ds_idx], [0], target_ds_idx])
    )

    # ----------------------------------------
    # allocate
    # ----------------------------------------
    amp_corr_dt_ds = np.full((len(target_dt_idx), len(lags_s)), np.nan)

    # ----------------------------------------
    # compute correlations
    # ----------------------------------------
    if np.isfinite(var_a) and var_a > eps:

        for i_dt, dt_idx in enumerate(target_dt_idx):

            a0 = amp_sub[:Nt - dt_idx]
            a1 = amp_sub[dt_idx:Nt]

            for k, ds_idx in enumerate(lags_s):

                if ds_idx >= 0:
                    x0 = a0[:, :Ns_fit - ds_idx]
                    x1 = a1[:, ds_idx:]
                else:
                    dd = -ds_idx
                    x0 = a0[:, dd:]
                    x1 = a1[:, :Ns_fit - dd]

                prod = x0 * x1
                mask = np.isfinite(prod)

                if np.any(mask):
                    amp_corr_dt_ds[i_dt, k] = (
                        np.nanmean(prod[mask]) / (var_a + eps)
                    )

    # store physical lags for plotting
    lags_t_phys = target_dt_idx * dt
    lags_s_rel = lags_s / Ns_fit

    # -------------------------------------------------
    # plotting
    # -------------------------------------------------
    if debug_plot:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))

        # phase kymograph
        im0 = axes[0, 0].imshow(
            phase.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="twilight",
        )
        axes[0, 0].set_title("Phase kymograph")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("s index")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # amplitude kymograph
        im1 = axes[0, 1].imshow(
            amplitude.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        axes[0, 1].set_title("Amplitude kymograph")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("s index")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    

        # lambda(t)
        axes[1, 0].plot(lambda_t, color="black")
        axes[1, 0].set_title(r"Instantaneous wavelength $\lambda(t)/L$")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel(r"$\lambda/L$")

        # mean amplitude
        axes[1, 1].plot(mean_a_t, color="black")
        axes[1, 1].set_title(r"Mean amplitude $\bar a(t)$")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel(r"$\bar a$")

        # histogram lambda
        axes[0, 2].hist(lambda_t[np.isfinite(lambda_t)], bins=40)
        axes[0, 2].set_title(
            f"λ histogram\nmean={np.nanmean(lambda_t):.3f}, std={np.nanstd(lambda_t):.3f}"
        )
        axes[0, 2].set_xlabel("λ/L")

        # histogram amplitude
        axes[1, 2].hist(mean_a_t[np.isfinite(mean_a_t)], bins=40)
        axes[1, 2].set_title(
            f"a histogram\nmean={np.nanmean(mean_a_t):.3f}, std={np.nanstd(mean_a_t):.3f}"
        )
        axes[1, 2].set_xlabel("a")

        # autocorr lambda + mean_a
        axes[2, 0].plot(lags_t_rel, lambda_autocorr, "o-", label=r"$C_\lambda$")
        axes[2, 0].plot(lags_t_rel, mean_a_autocorr, "o-", label=r"$C_a$")
        axes[2, 0].plot(lags_t_rel, lambda_mean_a_crosscorr, "o-", label="cross")

        axes[2, 0].set_xlabel(r"$\Delta t / T$")
        axes[2, 0].axhline(0, color="black", lw=1)
        axes[2, 0].set_ylabel("corr")
        axes[2, 0].set_title("Correlations (selected lags)")
        axes[2, 0].legend()

        
        from scipy.interpolate import griddata

        # ----------------------------------------
        # build coordinates
        # ----------------------------------------
        dt_vals = lags_t_rel   # Δt/T
        ds_vals = lags_s_rel   # Δs/L

        DT, DS = np.meshgrid(dt_vals, ds_vals, indexing="ij")

        # flatten
        points = np.column_stack([DS.ravel(), DT.ravel()])
        values = amp_corr_dt_ds.ravel()

        mask = np.isfinite(values)
        points = points[mask]
        values = values[mask]

        # ----------------------------------------
        # create regular grid
        # ----------------------------------------
        ds_i = np.linspace(ds_vals.min(), ds_vals.max(), 200)
        dt_i = np.linspace(dt_vals.min(), dt_vals.max(), 200)

        DS_i, DT_i = np.meshgrid(ds_i, dt_i)

        Zi = griddata(points, values, (DS_i, DT_i), method="linear")

        # ----------------------------------------
        # plot
        # ----------------------------------------
        pcm = axes[2, 1].pcolormesh(
            DS_i,
            DT_i,
            Zi,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            shading="auto",
        )

        axes[2, 1].set_title(r"$C_a(\Delta t,\Delta s)$")
        axes[2, 1].set_xlabel(r"$\Delta s / L$")
        axes[2, 1].set_ylabel(r"$\Delta t / T$")
        plt.colorbar(pcm, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # ----------------------------------------
        # overlay propagation line
        # ----------------------------------------
        lam_mean = np.nanmean(lambda_t)

        if np.isfinite(lam_mean) and lam_mean > 0:
            slope = 1.0 / lam_mean  # in (Δs/L) per (Δt/T)

            # current axis limits (IMPORTANT: don't change them)
            x_min, x_max = axes[2, 1].get_xlim()
            y_min, y_max = axes[2, 1].get_ylim()

            # construct line within bounds
            y_line = np.linspace(y_min, y_max, 200)
            x_line = slope * y_line

            # keep only visible part
            mask = (x_line >= x_min) & (x_line <= x_max)

            axes[2, 1].plot(
                x_line[mask],
                y_line[mask],
                color="black",
                linestyle="--",
                linewidth=2,
                label=r"$v_{\mathrm{phase}}$"
            )

            axes[2, 1].legend()


        zero_ds_idx = np.argmin(np.abs(lags_s))
        # use physical time lags instead of Nt
        x = lags_t_rel
        axes[2, 2].plot(x, amp_corr_dt_ds[:, zero_ds_idx], label=r"$\Delta s=0$")
        if len(lags_s) > 2:
            # use your defined ds values instead of index=3 hack
            ds_vals = [0.1, 0.2, 0.5]
            for ds_val in ds_vals:
                ds_idx = int(round(ds_val * Ns_fit))

                idxp = np.argmin(np.abs(lags_s - ds_idx))
                idxm = np.argmin(np.abs(lags_s + ds_idx))

                axes[2, 2].plot(
                    x,
                    amp_corr_dt_ds[:, idxp],
                    label=rf"$\Delta s={ds_val:.1f}$"
                )
                axes[2, 2].plot(
                    x,
                    amp_corr_dt_ds[:, idxm],
                    linestyle="--",
                    label=rf"$\Delta s=-{ds_val:.1f}$"
                )
        axes[2, 2].axhline(0.0, color="black", linewidth=1)
        axes[2, 2].set_title("Cuts of amplitude correlation")
        axes[2, 2].set_xlabel(r"$\Delta t / T$")
        axes[2, 2].set_ylabel("corr")
        axes[2, 2].legend()

        plt.tight_layout()
        plt.show()

    return {
        # core signals
        "lambda_t": lambda_t,
        "mean_a_t": mean_a_t,

        # correlations (computed at selected lags)
        "lambda_autocorr": lambda_autocorr,
        "mean_a_autocorr": mean_a_autocorr,
        "lambda_mean_a_crosscorr": lambda_mean_a_crosscorr,

        # amplitude 2D correlation
        "amp_corr_dt_ds": amp_corr_dt_ds,

        # 🔥 correct lag definitions
        "lags_dt_idx": target_dt_idx,     # integer frame lags
        "lags_t_phys": lags_t_phys,       # in seconds
        "lags_t_rel": lags_t_rel,         # Δt / T

        "lags_s_idx": lags_s,             # integer spatial shifts
        "lags_s_rel": lags_s_rel,         # Δs / L

        # meta
        "fit_mask": fit_mask,
        "s": s,

        "too_small": False,
    }

