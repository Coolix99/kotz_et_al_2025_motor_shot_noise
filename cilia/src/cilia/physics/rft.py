from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================
# Parameters
# ============================================================

@dataclass
class RFTParams:
    xi_perp: float
    xi_par: float

def compute_geometry(theta, s):
    """
    Extract gamma and geometry from given theta(s).
    We define theta0 = theta(s=0).
    """
    theta0 = theta[0]
    gamma = theta - theta0

    ds = s[1] - s[0]
    C = build_cumulative_matrix(len(s), ds)

    _, t, n, r_rel = geometry_from_gamma(gamma, theta0, C)

    return gamma, theta0, t, n, r_rel, C

def build_cumulative_matrix(N: int, ds: float) -> np.ndarray:
    """
    Lower-triangular integration matrix:
        (C @ q)[i] ≈ ∫_0^{s_i} q(s') ds'
    """
    return ds * np.tri(N, N, k=-1, dtype=float)

def geometry_from_gamma(gamma: np.ndarray, theta0: float, C: np.ndarray):
    """
    Given gamma(s), reconstruct geometry.
    """
    theta = gamma + theta0

    ct = np.cos(theta)
    st = np.sin(theta)

    t = np.column_stack((ct, st))
    n = np.column_stack((-st, ct))

    rx = C @ t[:, 0]
    ry = C @ t[:, 1]
    r_rel = np.column_stack((rx, ry))

    return theta, t, n, r_rel

def w_times(Rblocks: np.ndarray, X: np.ndarray, ds: float) -> np.ndarray:
    """
    Apply W = ds * diag(R_i)
    """

    X2 = np.atleast_2d(X)
    if X2.shape[0] != 2 * Rblocks.shape[0]:
        X2 = X2.T

    N = Rblocks.shape[0]
    m = X2.shape[1]

    Xr = X2.reshape(2, N, m).transpose(1, 0, 2)   # (N,2,m)
    Y = ds * np.einsum("nij,njk->nik", Rblocks, Xr)
    Y2 = Y.transpose(1, 0, 2).reshape(2 * N, m)

    return Y2[:, 0] if X.ndim == 1 else Y2

def rft_force_density(Rblocks: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Local line force density f(s) = -R(s) v(s), without ds.
    Input:
        Rblocks: (N,2,2)
        v:       (2N,) or (2N,m)
    Output:
        same shape as v
    """
    V = np.atleast_2d(v)
    if V.shape[0] != 2 * Rblocks.shape[0]:
        V = V.T

    N = Rblocks.shape[0]
    m = V.shape[1]

    Vr = V.reshape(2, N, m).transpose(1, 0, 2)      # (N,2,m)
    Fr = np.einsum("nij,njk->nik", Rblocks, Vr)      # (N,2,m)
    F = Fr.transpose(1, 0, 2).reshape(2 * N, m)

    return F[:, 0] if v.ndim == 1 else F

def cumtrapz_from_tip(f, s):
    ds = s[1] - s[0]

    f_flip = np.flip(f)
    T_flip = np.cumsum(f_flip) * ds
    T_flip -= 0.5 * ds * f_flip

    return np.flip(T_flip)

def build_operators_from_theta(theta, s, params):
    gamma, theta0, t, n, r_rel, C = compute_geometry(theta, s)

    N = len(s)

    # --- RFT operator ---
    xi_ratio = params.xi_perp / params.xi_par

    nn = n[:, :, None] * n[:, None, :]
    tt = t[:, :, None] * t[:, None, :]

    Rblocks = params.xi_par * (tt + xi_ratio * nn)

    # --- A operator ---
    Ax = C * n[:, 0][None, :]
    Ay = C * n[:, 1][None, :]
    A = np.vstack((Ax, Ay))  # (2N, N)

    # --- B operator ---
    ones = np.ones(N)
    zeros = np.zeros(N)

    Btop = np.column_stack((ones, zeros, -r_rel[:, 1]))
    Bbot = np.column_stack((zeros, ones,  r_rel[:, 0]))
    B = np.vstack((Btop, Bbot))

    return A, B, Rblocks, gamma, theta0, r_rel, n, C

def compute_hydrodynamics(
    *,
    s,
    gamma,
    gamma_dot,
    params,
    base_mode="free",   # "free", "fixed", "chlamy"
    body_drag=0.08/2,      # only for chlamy, factor of two because of two flagella pN s um^-1 
    base_angle=0.0,     # symmetry axis angle
):
    """
    Returns hydrodynamic torque along filament.

    Output:
        torque_s_to_L : integrated torque from s to tip
    """

    N = len(s)
    ds = s[1] - s[0]

    # reconstruct theta
    theta = gamma

    # geometry
    theta_full = theta
    _, t, n, r_rel = geometry_from_gamma(gamma, 0.0, build_cumulative_matrix(N, ds))

    # operators
    A, B, Rblocks, *_ = build_operators_from_theta(theta_full, s, params)

    # --- velocities ---
    v_shape = A @ gamma_dot   # (2N,)

    # reshape helper
    def reshape_vec(x):
        return x.reshape(2, N).T  # (N,2)

    # ============================
    # DETERMINE q
    # ============================

    if base_mode == "fixed":
        q = np.zeros(3)

    else:
        WA = w_times(Rblocks, A, ds)
        WB = w_times(Rblocks, B, ds)

        M = B.T @ WB
        K = B.T @ WA

        if base_mode == "free":
            q = -np.linalg.solve(M, K @ gamma_dot)

        elif base_mode == "chlamy":
            # --------------------------------------------------------
            # Symmetry axis direction
            # --------------------------------------------------------
            e_axis = np.array([np.cos(base_angle), np.sin(base_angle)])

            # Project rigid-body motion onto axis
            # Only allow translation along symmetry axis
            # q = (Ux, Uy, Omega) with:
            #   [Ux, Uy] = U * e_axis
            #   Omega = 0

            # Build reduced B operator
            # v_rb = U * e_axis + Omega * J r
            # but Omega = 0

            B_axis = np.zeros((2*N, 1))
            B_axis[:N, 0] = e_axis[0]
            B_axis[N:, 0] = e_axis[1]

            WA = w_times(Rblocks, A, ds)
            WB_axis = w_times(Rblocks, B_axis, ds)

            M = B_axis.T @ WB_axis
            K = B_axis.T @ WA

            # add body drag along axis
            M += body_drag

            U = -np.linalg.solve(M, K @ gamma_dot)

            q = np.array([
                U[0] * e_axis[0],
                U[0] * e_axis[1],
                0.0
            ])

        else:
            raise ValueError(base_mode)

    # --- total velocity ---
    v = v_shape + B @ q

    # --- hydrodynamic force ---
    f = -rft_force_density(Rblocks, v)  # (2N,)

    f = reshape_vec(f)

    # ============================================================
    # TORQUE via A* (adjoint)
    # ============================================================

    # cumulative integral from tip → base
    # flip, integrate, flip back
    f_flip = np.flip(f, axis=0)
    cum_int = np.cumsum(f_flip, axis=0) * ds
    cum_int = np.flip(cum_int, axis=0)
    # trapezoidal correction (optional but better)
    cum_int -= 0.5 * ds * f
    # now dot with n
    torque_density = np.einsum("ij,ij->i", n, cum_int)


    v2 = v.reshape(2, N).T  # (N,2)
    power_density = np.einsum("ni,nij,nj->n", v2, Rblocks, v2)
    total_power = np.sum(power_density) * ds

    return {
        "torque_density": torque_density,
        "q": q,
        "power": total_power,
        "power_density": power_density,
    }

# ============================================================
# Elastic torque
# ============================================================

def compute_elastic_torque(gamma, s, B, d2K):
    dgamma_ds = np.gradient(gamma, s, axis=1)
    d2gamma_ds2 = np.gradient(dgamma_ds, s, axis=1)

    tau = B * d2gamma_ds2 - d2K * gamma

    return tau


# ============================================================
# Hydro-elastic balance (final observable)
# ============================================================

def compute_hydro_elastic_balance(
    theta,
    s,
    dt,
    *,
    B,
    d2K,
    params,
):


    modes = ["free", "fixed", "chlamy"]
    T_h = {m: np.zeros_like(theta) for m in modes}
    P_h = {m: np.zeros(theta.shape[0]) for m in modes}
    gamma = theta - theta[:, [0]]
    dtgamma_dt = np.gradient(gamma, dt, axis=0)
    for i in range(theta.shape[0]):
        for m in modes:
            out = compute_hydrodynamics(
                s=s,
                gamma=gamma[i],
                gamma_dot=dtgamma_dt[i],
                params=params,
                base_mode=m,
            )
            T_h[m][i] = out["torque_density"]
            P_h[m][i] = out["power"]

    tau_e = compute_elastic_torque(gamma, s, B, d2K)

    eps = 1e-12

    results = {}

    for m in modes:
        R = np.abs(T_h[m]) / (np.abs(T_h[m]) + np.abs(tau_e) + eps)
        # mean over s and t
        results[f"mean_R_{m}"] = float(np.mean(R))

        mean_power = np.mean(P_h[m])
        results[f"mean_power_{m}"] = float(mean_power)
    return results
