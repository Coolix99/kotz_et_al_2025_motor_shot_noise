import numpy as np
import matplotlib.pyplot as plt


def _shifted_legendre_basis(u: np.ndarray, order: int) -> np.ndarray:
    if order <= 0:
        return np.empty((u.size, 0), dtype=float)

    x = 2.0 * u - 1.0
    V = np.polynomial.legendre.legvander(x, order)
    return V[:, 1:]  # drop constant


def _fourier_basis(phi: np.ndarray, n_fourier: int) -> np.ndarray:
    if n_fourier <= 0:
        return np.empty((phi.size, 0), dtype=float)

    cols = []
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(k * phi))
        cols.append(np.cos(k * phi))
    return np.column_stack(cols)


def _interaction_basis(Bs: np.ndarray, Bp: np.ndarray) -> np.ndarray:
    """
    All pairwise products of columns of Bs and Bp.
    """
    if Bs.shape[1] == 0 or Bp.shape[1] == 0:
        return np.empty((Bs.shape[0], 0), dtype=float)

    cols = []
    for i in range(Bs.shape[1]):
        for j in range(Bp.shape[1]):
            cols.append(Bs[:, i] * Bp[:, j])
    return np.column_stack(cols)

def get_non_iso_local(
    state: np.ndarray,
    s: np.ndarray,
    *,
    poly_order: int = 2,
    n_fourier: int = 2,
    include_phase_terms: bool = True,
    include_mixed_terms: bool = True,
    debug: bool = False,
    eps: float = 1e-12,
):
    """
    Fit amplitude-frequency coupling from local state field.

    Parameters
    ----------
    state : ndarray, shape (Nt, Ns, 3)
        state[...,0] = phase
        state[...,1] = omega
        state[...,2] = amplitude
    s : ndarray, shape (Ns,)
        spatial positions of the retained good s-points
    """

    state = np.asarray(state, dtype=float)
    s = np.asarray(s, dtype=float)

    print("=== DEBUG SHAPES ===")
    print(f"state: shape={state.shape}, dtype={state.dtype}")
    print(f"s:     shape={s.shape}, dtype={s.dtype}")
    print("====================")

    if state.ndim != 3 or state.shape[2] != 3:
        raise ValueError("state must have shape (Nt, Ns, 3)")

    Nt, Ns, _ = state.shape

    if s.shape != (Ns,):
        raise ValueError("s must have shape (Ns,)")

    phase = state[..., 0]
    omega = state[..., 1]
    amplitude = state[..., 2]

    good = (
        np.isfinite(phase)
        & np.isfinite(omega)
        & np.isfinite(amplitude)
    )

    if np.count_nonzero(good) < 20:
        return np.nan, {"reason": "too_few_points", "n_used": int(np.count_nonzero(good))}

    phase_v = phase[good]
    omega_v = omega[good]
    amp_v = amplitude[good]
    s_v = np.broadcast_to(s[None, :], (Nt, Ns))[good]

    mean_omega = np.mean(omega_v)
    mean_amp = np.mean(amp_v)

    if abs(mean_omega) < eps or abs(mean_amp) < eps:
        return np.nan, {
            "reason": "mean_too_small",
            "mean_omega": float(mean_omega),
            "mean_amp": float(mean_amp),
        }

    y_rel = (omega_v - mean_omega) / (mean_omega + eps)
    x_rel = (amp_v - mean_amp) / (mean_amp + eps)

    # normalize s to [0,1]
    s_min = np.min(s_v)
    s_max = np.max(s_v)
    if s_max - s_min < eps:
        u = np.zeros_like(s_v)
    else:
        u = (s_v - s_min) / (s_max - s_min)

    # wrap phase to [0, 2pi)
    phi_mod = np.mod(phase_v, 2.0 * np.pi)

    # -------------------------------------------------
    # bases
    # -------------------------------------------------
    Bs = _shifted_legendre_basis(u, poly_order)
    Bp = _fourier_basis(phi_mod, n_fourier) if include_phase_terms else np.empty((u.size, 0))
    Bsp = _interaction_basis(Bs, Bp) if include_mixed_terms else np.empty((u.size, 0))

    # -------------------------------------------------
    # model:
    # y = baseline + x * coupling
    #
    # baseline = c0 + Bs + Bp + Bsp
    # coupling = beta0 + Bs + Bp + Bsp
    # -------------------------------------------------
    baseline_cols = [np.ones((u.size, 1))]
    coupling_cols = [x_rel[:, None]]

    if Bs.shape[1] > 0:
        baseline_cols.append(Bs)
        coupling_cols.append(x_rel[:, None] * Bs)

    if Bp.shape[1] > 0:
        baseline_cols.append(Bp)
        coupling_cols.append(x_rel[:, None] * Bp)

    if Bsp.shape[1] > 0:
        baseline_cols.append(Bsp)
        coupling_cols.append(x_rel[:, None] * Bsp)

    X = np.column_stack(baseline_cols + coupling_cols)

    coef, *_ = np.linalg.lstsq(X, y_rel, rcond=None)
    y_pred = X @ coef

    n_base = sum(c.shape[1] for c in baseline_cols)
    coupling_coef = coef[n_base:]

    beta0 = float(coupling_coef[0])

    beta_v = np.full_like(y_rel, beta0)

    idx = 1
    if Bs.shape[1] > 0:
        beta_v += Bs @ coupling_coef[idx:idx + Bs.shape[1]]
        idx += Bs.shape[1]
    if Bp.shape[1] > 0:
        beta_v += Bp @ coupling_coef[idx:idx + Bp.shape[1]]
        idx += Bp.shape[1]
    if Bsp.shape[1] > 0:
        beta_v += Bsp @ coupling_coef[idx:idx + Bsp.shape[1]]

    ss_res = float(np.sum((y_rel - y_pred) ** 2))
    ss_tot = float(np.sum((y_rel - np.mean(y_rel)) ** 2))
    r2 = np.nan if ss_tot < eps else 1.0 - ss_res / ss_tot

    aux = {
        "n_used": int(y_rel.size),
        "mean_omega": float(mean_omega),
        "mean_amp": float(mean_amp),
        "beta_global": beta0,
        "beta_mean": float(np.mean(beta_v)),
        "beta_std": float(np.std(beta_v)),
        "r2": float(r2),
    }

    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].scatter(x_rel, y_rel, alpha=0.15, s=10)
        axes[0, 0].set_xlabel("x_rel = relative amplitude")
        axes[0, 0].set_ylabel("y_rel = relative omega")
        axes[0, 0].set_title("Raw scatter")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].scatter(y_rel, y_pred, alpha=0.15, s=10)
        axes[0, 1].set_xlabel("observed y_rel")
        axes[0, 1].set_ylabel("predicted y_rel")
        axes[0, 1].set_title(f"Fit prediction (R²={r2:.3f})")
        axes[0, 1].grid(alpha=0.3)

        sc = axes[1, 0].scatter(u, beta_v, c=phi_mod, alpha=0.3, s=10, cmap="twilight")
        axes[1, 0].set_xlabel("s")
        axes[1, 0].set_ylabel("local coupling beta")
        axes[1, 0].set_title("β(s, φ) on used points")
        axes[1, 0].grid(alpha=0.3)
        plt.colorbar(sc, ax=axes[1, 0], label="phase")

        sc2 = axes[1, 1].scatter(u, y_rel - y_pred, c=x_rel, alpha=0.3, s=10, cmap="coolwarm")
        axes[1, 1].set_xlabel("s")
        axes[1, 1].set_ylabel("residual")
        axes[1, 1].set_title("Residuals vs s")
        axes[1, 1].grid(alpha=0.3)
        plt.colorbar(sc2, ax=axes[1, 1], label="x_rel")

        plt.tight_layout()
        plt.show()

    return beta0, aux
