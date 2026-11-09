import numpy as np

from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import (
    TANGENT_ANGLE,
    MEAN_SHAPE,
    GLOBAL_PHASE,
    PERIODIC_AVG,
    GEYER,
    R2_OBSERVABLES,
)


def _safe_r2(y, y_hat, center=True, weights=None):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    mask = np.isfinite(y) & np.isfinite(y_hat)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        mask = mask & np.isfinite(weights) & (weights > 0)

    if np.count_nonzero(mask) < 2:
        return np.nan

    yy = y[mask]
    hh = y_hat[mask]

    if weights is None:
        ww = np.ones_like(yy)
    else:
        ww = weights[mask]

    if center:
        y0 = np.average(yy, weights=ww)
    else:
        y0 = 0.0

    ss_res = np.sum(ww * (yy - hh) ** 2)
    ss_tot = np.sum(ww * (yy - y0) ** 2)

    if not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def _as_mean_shape(value):
    if isinstance(value, dict):
        for key in ("mean_shape", "theta", "value", "data"):
            if key in value:
                value = value[key]
                break

    theta = np.asarray(value, dtype=float).squeeze()

    if theta.ndim != 1:
        raise ValueError(f"Expected MEAN_SHAPE with shape (n_s,), got {theta.shape}")

    return theta


def _as_segments(value):
    if isinstance(value, dict):
        for key in ("segments", "value", "data"):
            if key in value:
                value = value[key]
                break

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list of segments, got {type(value)}")

    return [np.asarray(seg, dtype=float) for seg in value]


def _as_phase_segments(value):
    if isinstance(value, dict):
        for key in ("segments", "value", "data"):
            if key in value:
                value = value[key]
                break

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list of phase segments, got {type(value)}")

    out = []
    for seg in value:
        if isinstance(seg, dict):
            out.append(np.asarray(seg["phase"], dtype=float).squeeze())
        else:
            out.append(np.asarray(seg, dtype=float).squeeze())

    return out


def _as_periodic_mean(value):
    if not isinstance(value, dict) or "mean" not in value:
        raise ValueError("PERIODIC_AVG must be a dict with field 'mean'")

    return np.asarray(value["mean"], dtype=float).squeeze()


def _orient_matrix_to_ns(arr, n_s):
    arr = np.asarray(arr, dtype=float).squeeze()

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {arr.shape}")

    if arr.shape[1] == n_s:
        return arr

    if arr.shape[0] == n_s:
        return arr.T

    raise ValueError(f"Cannot orient matrix with shape {arr.shape} to n_s={n_s}")


def _r2_mean_shape_linear(theta):
    theta = np.asarray(theta, dtype=float).squeeze()
    s = np.linspace(0.0, 1.0, theta.size)

    mask = np.isfinite(theta)
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan

    s_fit = s[mask]
    theta_fit = np.unwrap(theta[mask])

    # linear shape with intercept: theta = kappa_L * s + b
    p = np.polyfit(s_fit, theta_fit, deg=1)
    theta_hat_linear = np.polyval(p, s)

    # curvature-only, fixed base gauge: theta = kappa_L * s
    denom = np.sum(s_fit**2)
    if denom > 0:
        kappa_only = np.sum(s_fit * theta_fit) / denom
        theta_hat_curv_only = kappa_only * s
    else:
        theta_hat_curv_only = np.full_like(theta, np.nan)

    r2_linear = _safe_r2(theta, theta_hat_linear, center=True)
    r2_curv_only = _safe_r2(theta, theta_hat_curv_only, center=True)

    return r2_linear, r2_curv_only


def _orient_segment(seg, n_s):
    seg = np.asarray(seg, dtype=float)

    if seg.ndim != 2:
        raise ValueError(f"Expected segment with shape (n_t, n_s), got {seg.shape}")

    if seg.shape[1] == n_s:
        return seg

    if seg.shape[0] == n_s:
        return seg.T

    raise ValueError(f"Cannot orient segment with shape {seg.shape} to n_s={n_s}")


def _r2_gamma_pca2(gamma_segments, n_s):
    total_ss = 0.0
    total_res = 0.0
    total_frames = 0

    for seg in gamma_segments:
        try:
            Y = _orient_segment(seg, n_s)
        except Exception:
            continue

        if Y.shape[0] < 3 or Y.shape[1] < 2:
            continue

        Yc = Y - np.nanmean(Y, axis=0, keepdims=True)
        Yc = np.nan_to_num(Yc, nan=0.0, posinf=0.0, neginf=0.0)

        ss = float(np.sum(Yc**2))
        if ss <= 0.0:
            continue

        singular_values = np.linalg.svd(Yc, full_matrices=False, compute_uv=False)
        captured = float(np.sum(singular_values[:2] ** 2))

        total_ss += ss
        total_res += max(ss - captured, 0.0)
        total_frames += Y.shape[0]

    if total_ss <= 0.0:
        return np.nan, total_frames

    return float(1.0 - total_res / total_ss), float(total_frames)


def _interp_periodic_gamma(gamma_phi, phase):
    gamma_phi = np.asarray(gamma_phi, dtype=float)
    phase = np.mod(np.asarray(phase, dtype=float), 2.0 * np.pi)

    n_phi, n_s = gamma_phi.shape
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    xp = np.concatenate([phi_grid, [2.0 * np.pi]])
    out = np.empty((phase.size, n_s), dtype=float)

    for j in range(n_s):
        fp = np.concatenate([gamma_phi[:, j], [gamma_phi[0, j]]])
        out[:, j] = np.interp(phase, xp, fp)

    return out


def _r2_gamma_limit_cycle(gamma_segments, phase_segments, gamma_phi, n_s):
    total_ss = 0.0
    total_res = 0.0

    n = min(len(gamma_segments), len(phase_segments))

    for i in range(n):
        try:
            Y = _orient_segment(gamma_segments[i], n_s)
            phase = np.asarray(phase_segments[i], dtype=float).squeeze()
        except Exception:
            continue

        if phase.size != Y.shape[0]:
            continue

        Y_hat = _interp_periodic_gamma(gamma_phi, phase)

        mask = np.isfinite(Y) & np.isfinite(Y_hat)
        if np.count_nonzero(mask) < 2:
            continue

        yy = Y[mask]
        hh = Y_hat[mask]

        ss = float(np.sum(yy**2))
        res = float(np.sum((yy - hh) ** 2))

        if ss > 0.0:
            total_ss += ss
            total_res += res

    if total_ss <= 0.0:
        return np.nan

    return float(1.0 - total_res / total_ss)
A_MODELS = ("const", "linear", "quadratic", "full")
PHI_MODELS = ("const", "linear", "quadratic", "full")


def _shifted_legendre_x(n_s):
    return np.linspace(0.0, 1.0, n_s)


def _shifted_legendre(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = 2.0 * x - 1.0

    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t

    p0 = np.ones_like(t)
    p1 = t

    for k in range(2, n + 1):
        p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k
        p0, p1 = p1, p2

    return p1


def _reconstruct_shifted_legendre(coeffs, n_s, degree):
    x = _shifted_legendre_x(n_s)
    out = np.zeros(n_s, dtype=float)

    for n in range(degree + 1):
        key = str(n)
        if key not in coeffs:
            return None
        out += float(coeffs[key]) * _shifted_legendre(n, x)

    return out


def _fit_shifted_legendre(profile, degree, keep=None, weights=None):
    profile = np.asarray(profile, dtype=float).squeeze()
    n_s = profile.size
    x = _shifted_legendre_x(n_s)

    if keep is None:
        keep = np.ones(n_s, dtype=bool)
    else:
        keep = np.asarray(keep, dtype=bool).squeeze()
        if keep.size != n_s:
            keep = np.ones(n_s, dtype=bool)

    keep &= np.isfinite(profile)

    if weights is not None:
        weights = np.asarray(weights, dtype=float).squeeze()
        if weights.size == n_s:
            keep &= np.isfinite(weights) & (weights > 0)
            w = np.sqrt(weights[keep])
        else:
            w = None
    else:
        w = None

    if np.count_nonzero(keep) < degree + 1:
        return np.full(n_s, np.nan)

    X = np.stack(
        [_shifted_legendre(n, x[keep]) for n in range(degree + 1)],
        axis=1,
    )

    coeffs, *_ = np.linalg.lstsq(
        X * (w[:, None] if w is not None else 1.0),
        profile[keep] * (w if w is not None else 1.0),
        rcond=None,
    )

    X_full = np.stack(
        [_shifted_legendre(n, x) for n in range(degree + 1)],
        axis=1,
    )

    return X_full @ coeffs


def _resample_profile(profile, n_s):
    profile = np.asarray(profile, dtype=float).squeeze()

    if profile.size == n_s:
        return profile

    return np.interp(
        np.linspace(0.0, 1.0, n_s),
        np.linspace(0.0, 1.0, profile.size),
        profile,
    )


def _get_geyer_profile(geyer, key, n_s, default=None):
    if key not in geyer:
        if default is None:
            raise KeyError(f"GEYER output has no key '{key}'")
        return np.full(n_s, default, dtype=float)

    return _resample_profile(geyer[key], n_s=n_s)


def _geyer_keep_mask(geyer, n_s):
    keep = np.ones(n_s, dtype=bool)

    if "keep_mask" in geyer:
        raw = np.asarray(geyer["keep_mask"], dtype=bool).squeeze()
        if raw.size == n_s:
            keep &= raw

    return keep


def _safe_r2_masked(y, y_hat, keep=None, center=True, weights=None):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    mask = np.isfinite(y) & np.isfinite(y_hat)

    if keep is not None:
        keep = np.asarray(keep, dtype=bool)

        if y.ndim == 1 and keep.size == y.size:
            mask &= keep

        if y.ndim == 2 and keep.size == y.shape[1]:
            mask &= keep[None, :]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)

        if y.ndim == 1 and weights.size == y.size:
            w_full = weights
        elif y.ndim == 2 and weights.size == y.shape[1]:
            w_full = np.broadcast_to(weights[None, :], y.shape)
        else:
            w_full = np.ones_like(y, dtype=float)

        mask &= np.isfinite(w_full) & (w_full > 0)
        ww = w_full[mask]
    else:
        ww = np.ones(np.count_nonzero(mask), dtype=float)

    if np.count_nonzero(mask) < 2:
        return np.nan

    yy = y[mask]
    hh = y_hat[mask]

    if center:
        y0 = np.average(yy, weights=ww)
    else:
        y0 = 0.0

    ss_res = np.sum(ww * (yy - hh) ** 2)
    ss_tot = np.sum(ww * (yy - y0) ** 2)

    if not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)

def _a_model_profile(geyer, model, n_s, keep=None):
    a_full = _get_geyer_profile(geyer, "a_s", n_s=n_s)

    if model == "full":
        return a_full

    if model == "const":
        local_keep = np.isfinite(a_full)
        if keep is not None and keep.size == n_s:
            local_keep &= keep

        value = np.nanmean(a_full[local_keep])
        return np.full(n_s, value, dtype=float)

    if model == "linear":
        return _fit_shifted_legendre(
            a_full,
            degree=1,
            keep=keep,
        )

    if model == "quadratic":
        return _fit_shifted_legendre(
            a_full,
            degree=2,
            keep=keep,
        )

    raise ValueError(f"Unknown amplitude model '{model}'")


def _phi_model_profile(geyer, model, n_s, keep=None):
    phi_full = np.unwrap(_get_geyer_profile(geyer, "phi_s", n_s=n_s))

    weights = None
    if "a_s" in geyer:
        try:
            a_full = _get_geyer_profile(geyer, "a_s", n_s=n_s)
            weights = a_full**2
        except Exception:
            weights = None

    if model == "full":
        return phi_full

    if model == "const":
        local_keep = np.isfinite(phi_full)

        if keep is not None and keep.size == n_s:
            local_keep &= keep

        if weights is not None and weights.size == n_s:
            local_keep &= np.isfinite(weights) & (weights > 0)
            value = np.average(phi_full[local_keep], weights=weights[local_keep])
        else:
            value = np.nanmean(phi_full[local_keep])

        return np.full(n_s, value, dtype=float)

    if model == "linear":
        return _fit_shifted_legendre(
            phi_full,
            degree=1,
            keep=keep,
            weights=weights,
        )

    if model == "quadratic":
        return _fit_shifted_legendre(
            phi_full,
            degree=2,
            keep=keep,
            weights=weights,
        )

    raise ValueError(f"Unknown phase model '{model}'")

def _geyer_dynamic_reference(gamma_phi, geyer):
    n_phi, n_s = gamma_phi.shape

    if "psi0_s" in geyer:
        psi0 = _get_geyer_profile(geyer, "psi0_s", n_s=n_s, default=0.0)
    else:
        psi0 = np.mean(gamma_phi, axis=0)

    return gamma_phi - psi0[None, :]


def _geyer_reconstruction_from_profiles(gamma_phi, geyer, a_model, phi_model):
    n_phi, n_s = gamma_phi.shape
    varphi = 2.0 * np.pi * np.arange(n_phi) / n_phi

    keep = _geyer_keep_mask(geyer, n_s=n_s)

    a_s = _a_model_profile(geyer, a_model, n_s=n_s, keep=keep)
    phi_s = _phi_model_profile(geyer, phi_model, n_s=n_s, keep=keep)

    # IMPORTANT:
    # This matches your geyer_fit.py convention:
    # psi_hat = a(s) * cos(varphi + Phi(s))
    return a_s[None, :] * np.cos(varphi[:, None] + phi_s[None, :])

def _eval_geyer_model_at_phase(phase, n_s, geyer, a_model, phi_model):
    phase = np.asarray(phase, dtype=float).squeeze()
    phase_mod = np.mod(phase, 2.0 * np.pi)

    keep = _geyer_keep_mask(geyer, n_s=n_s)

    a_s = _a_model_profile(
        geyer,
        a_model,
        n_s=n_s,
        keep=keep,
    )

    phi_s = _phi_model_profile(
        geyer,
        phi_model,
        n_s=n_s,
        keep=keep,
    )

    # Matches geyer_fit_from_cycle convention:
    # psi_hat = a(s) cos(varphi + Phi(s))
    return a_s[None, :] * np.cos(
        phase_mod[:, None] + phi_s[None, :]
    )


def _r2_gamma_geyer_matrix_against_time_series(
    gamma_segments,
    phase_segments,
    geyer,
    n_s,
):
    if not isinstance(geyer, dict):
        return np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    if "a_s" not in geyer or "phi_s" not in geyer:
        return np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    keep = _geyer_keep_mask(geyer, n_s=n_s)

    matrix = np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    for i, a_model in enumerate(A_MODELS):
        for j, phi_model in enumerate(PHI_MODELS):
            total_ss = 0.0
            total_res = 0.0

            n = min(len(gamma_segments), len(phase_segments))

            for k in range(n):
                try:
                    Y = _orient_segment(gamma_segments[k], n_s)
                    phase = np.asarray(phase_segments[k], dtype=float).squeeze()
                except Exception:
                    continue

                if phase.size != Y.shape[0]:
                    continue

                try:
                    Y_hat = _eval_geyer_model_at_phase(
                        phase,
                        n_s=n_s,
                        geyer=geyer,
                        a_model=a_model,
                        phi_model=phi_model,
                    )
                except Exception:
                    continue

                mask = np.isfinite(Y) & np.isfinite(Y_hat)

                if keep.size == n_s:
                    mask &= keep[None, :]

                if np.count_nonzero(mask) < 2:
                    continue

                yy = Y[mask]
                hh = Y_hat[mask]

                ss = float(np.sum(yy**2))
                res = float(np.sum((yy - hh) ** 2))

                if ss > 0.0:
                    total_ss += ss
                    total_res += res

            if total_ss > 0.0:
                matrix[i, j] = float(1.0 - total_res / total_ss)

    return matrix

def _r2_geyer_a_phi_matrix(gamma_phi, geyer):
    if not isinstance(geyer, dict):
        return np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    if "a_s" not in geyer or "phi_s" not in geyer:
        return np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    n_phi, n_s = gamma_phi.shape
    keep = _geyer_keep_mask(geyer, n_s=n_s)

    psi_c = _geyer_dynamic_reference(gamma_phi, geyer)

    matrix = np.full((len(A_MODELS), len(PHI_MODELS)), np.nan)

    for i, a_model in enumerate(A_MODELS):
        for j, phi_model in enumerate(PHI_MODELS):
            try:
                psi_hat = _geyer_reconstruction_from_profiles(
                    gamma_phi,
                    geyer,
                    a_model=a_model,
                    phi_model=phi_model,
                )

                matrix[i, j] = _safe_r2_masked(
                    psi_c,
                    psi_hat,
                    keep=keep,
                    center=False,
                )
            except Exception:
                matrix[i, j] = np.nan

    return matrix


def _r2_profile_models(geyer, profile_key, coeff_key, models=("const", "linear", "quadratic")):
    if not isinstance(geyer, dict) or profile_key not in geyer:
        return {model: np.nan for model in models}

    profile = np.asarray(geyer[profile_key], dtype=float).squeeze()

    if profile.ndim != 1 or profile.size < 3:
        return {model: np.nan for model in models}

    n_s = profile.size
    keep = _geyer_keep_mask(geyer, n_s=n_s)

    if profile_key == "phi_s":
        y = np.unwrap(profile)
        weights = None

        if "a_s" in geyer:
            a_s = np.asarray(geyer["a_s"], dtype=float).squeeze()
            if a_s.size == n_s:
                weights = a_s**2
    else:
        y = profile
        weights = None

    out = {}

    for model in models:
        try:
            if model == "const":
                local_keep = keep & np.isfinite(y)

                if weights is not None:
                    local_keep &= np.isfinite(weights) & (weights > 0)
                    y0 = np.average(y[local_keep], weights=weights[local_keep])
                else:
                    y0 = np.nanmean(y[local_keep])

                y_hat = np.full(n_s, y0)

            elif model == "linear":
                if coeff_key in geyer:
                    y_hat = _reconstruct_shifted_legendre(
                        geyer[coeff_key],
                        n_s=n_s,
                        degree=1,
                    )
                    if y_hat is None:
                        y_hat = _fit_shifted_legendre(
                            y,
                            degree=1,
                            keep=keep,
                            weights=weights,
                        )
                else:
                    y_hat = _fit_shifted_legendre(
                        y,
                        degree=1,
                        keep=keep,
                        weights=weights,
                    )

            elif model == "quadratic":
                if coeff_key in geyer:
                    y_hat = _reconstruct_shifted_legendre(
                        geyer[coeff_key],
                        n_s=n_s,
                        degree=2,
                    )
                    if y_hat is None:
                        y_hat = _fit_shifted_legendre(
                            y,
                            degree=2,
                            keep=keep,
                            weights=weights,
                        )
                else:
                    y_hat = _fit_shifted_legendre(
                        y,
                        degree=2,
                        keep=keep,
                        weights=weights,
                    )

            else:
                y_hat = np.full(n_s, np.nan)

            out[model] = _safe_r2_masked(
                y,
                y_hat,
                keep=keep,
                center=True,
                weights=weights,
            )

        except Exception:
            out[model] = np.nan

    return out

@data_transformer(
    inputs=(MEAN_SHAPE, TANGENT_ANGLE, GLOBAL_PHASE, PERIODIC_AVG, GEYER),
    outputs=(R2_OBSERVABLES,),
)
def observable_r2_t(mean_shape, tangent_angle_segments, phase_segments, periodic_avg, geyer):
    theta_mean = _as_mean_shape(mean_shape)
    n_s = theta_mean.size

    gamma_segments = _as_segments(tangent_angle_segments)
    phase_segments = _as_phase_segments(phase_segments)

    gamma_phi = _as_periodic_mean(periodic_avg)
    gamma_phi = _orient_matrix_to_ns(gamma_phi, n_s=n_s)
    n_phi = gamma_phi.shape[0]

    r2_mean_linear, r2_mean_curv_only = _r2_mean_shape_linear(theta_mean)

    r2_gamma_pca2, n_gamma_frames = _r2_gamma_pca2(
        gamma_segments,
        n_s=n_s,
    )

    r2_gamma_limit_cycle = _r2_gamma_limit_cycle(
        gamma_segments,
        phase_segments,
        gamma_phi,
        n_s=n_s,
    )

    geyer_matrix = _r2_gamma_geyer_matrix_against_time_series(
        gamma_segments=gamma_segments,
        phase_segments=phase_segments,
        geyer=geyer,
        n_s=n_s,
    )

    a_profile_r2 = _r2_profile_models(
        geyer,
        profile_key="a_s",
        coeff_key="a_coeffs",
    )

    phi_profile_r2 = _r2_profile_models(
        geyer,
        profile_key="phi_s",
        coeff_key="phi_coeffs",
    )

    return {
        R2_OBSERVABLES.key: {
            "mean_shape_linear": float(r2_mean_linear),
            "mean_shape_curvature_only": float(r2_mean_curv_only),
            "gamma_pca2": float(r2_gamma_pca2),
            "gamma_limit_cycle": float(r2_gamma_limit_cycle),

            "geyer_r2_a_phi_matrix": geyer_matrix.astype(float),

            # Matrix convenience aliases
            "gamma_geyer_full_profiles": float(geyer_matrix[3, 3]),
            "gamma_geyer_a_const_phi_linear": float(geyer_matrix[0, 1]),
            "gamma_geyer_a_const_phi_quadratic": float(geyer_matrix[0, 2]),
            "gamma_geyer_a_linear_phi_linear": float(geyer_matrix[1, 1]),
            "gamma_geyer_a_quadratic_phi_quadratic": float(geyer_matrix[2, 2]),

            # Profile-only R2 values
            "a_profile_const": float(a_profile_r2["const"]),
            "a_profile_linear": float(a_profile_r2["linear"]),
            "a_profile_quadratic": float(a_profile_r2["quadratic"]),
            "phi_profile_const": float(phi_profile_r2["const"]),
            "phi_profile_linear": float(phi_profile_r2["linear"]),
            "phi_profile_quadratic": float(phi_profile_r2["quadratic"]),

            "n_gamma_frames": float(n_gamma_frames),
            "n_s": float(n_s),
            "n_phi": float(n_phi),
        }
    }
   
   
   