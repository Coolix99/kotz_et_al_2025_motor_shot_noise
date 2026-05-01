#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass
class Params:
    # dimensionless Cass parameters
    mu_a: float = 3000.0
    mu: float = 50.0
    zeta: float = 0.1
    eta: float = 0.3
    fstar: float = 2.0
    beta: float = 0.05

    # hydrodynamic strength
    lambda_h: float = 0.2

    # geometry
    N: int = 101

    # numerics
    dt: float = 1e-3
    T: float = 200.0
    save_every: int = 10

    gmres_rtol: float = 1e-8
    gmres_maxiter: int = 300


def build_second_derivative_matrix(N: int, ds: float) -> sp.csr_matrix:
    """
    Second derivative with rows to be overwritten later by BCs.
    Interior rows are standard centered differences.
    """
    e = np.ones(N)
    D2 = sp.diags([e, -2.0 * e, e], offsets=(-1, 0, 1), shape=(N, N), format="lil")
    D2 /= ds ** 2
    return D2.tocsr()


def build_cumulative_matrix(N: int, ds: float) -> np.ndarray:
    r"""
    Lower-triangular integration matrix:
      (C @ q)[i] ~ \int_0^{s_i} q(s') ds'
    using a left Riemann rule with zero at the base.
    """
    C = ds * np.tri(N, N, k=-1, dtype=float)
    return C


def geometry_from_gamma(gamma: np.ndarray, theta0: float, C: np.ndarray):
    """
    Given gamma(s), theta = gamma + theta0.
    Returns theta, t, n, r_rel where r_rel = r - r0.
    """
    theta = gamma + theta0
    ct = np.cos(theta)
    st = np.sin(theta)

    t = np.column_stack((ct, st))
    n = np.column_stack((-st, ct))

    # r_rel(s) = \int_0^s t(s') ds'
    rx = C @ t[:, 0]
    ry = C @ t[:, 1]
    r_rel = np.column_stack((rx, ry))
    return theta, t, n, r_rel


def w_times(Rblocks: np.ndarray, X: np.ndarray, ds: float) -> np.ndarray:
    """
    Apply the block-diagonal operator W = ds * diag(R_i) to X.

    Rblocks: (N, 2, 2)
    X:       (2N, m) or (2N,)
    return:  same shape as X
    """
    X2 = np.atleast_2d(X)
    if X2.shape[0] != 2 * Rblocks.shape[0]:
        X2 = X2.T

    N = Rblocks.shape[0]
    m = X2.shape[1]

    Xr = X2.reshape(2, N, m).transpose(1, 0, 2)      # (N, 2, m)
    Y = ds * np.einsum("nij,njk->nik", Rblocks, Xr)  # (N, 2, m)
    Y2 = Y.transpose(1, 0, 2).reshape(2 * N, m)

    if X.ndim == 1:
        return Y2[:, 0]
    return Y2


def build_hydro_operators(
    gamma: np.ndarray,
    theta0: float,
    params: Params,
    C: np.ndarray,
):
    """
    Build the nonlocal hydrodynamic friction operator H(gamma, theta0) and
    the rigid-body map P such that

        q = [Ux, Uy, Omega]^T = P @ gdot

    is obtained from total force / torque balance.

    Also returns the current geometry.
    """
    _, t, n, r_rel = geometry_from_gamma(gamma, theta0, C)
    N = params.N
    ds = 1.0 / (N - 1)
    s = np.linspace(0.0, 1.0, N)

    # RFT 2x2 blocks
    xi_ratio = 1.81   # xi_perp / xi_parallel
    nn = n[:, :, None] * n[:, None, :]
    tt = t[:, :, None] * t[:, None, :]
    Rblocks = params.lambda_h * (xi_ratio * nn + tt)

    # Shape velocity operator A: gdot -> v_shape
    # v_shape(s_i) = \int_0^{s_i} gdot(s') n(s') ds'
    Ax = C * n[:, 0][None, :]
    Ay = C * n[:, 1][None, :]
    A = np.vstack((Ax, Ay))  # (2N, N)

    # Rigid body velocity operator B:
    # v_rb = [Ux, Uy] + Omega * J (r-r0),  J(x,y)=(-y,x)
    ones = np.ones(N)
    zeros = np.zeros(N)
    Btop = np.column_stack((ones, zeros, -r_rel[:, 1]))
    Bbot = np.column_stack((zeros, ones,  r_rel[:, 0]))
    B = np.vstack((Btop, Bbot))  # (2N, 3)

    WA = w_times(Rblocks, A, ds)   # (2N, N)
    WB = w_times(Rblocks, B, ds)   # (2N, 3)

    # Force/torque balance:
    # B^T W (A gdot + B q) = 0
    M_rb = B.T @ WB                # (3,3)
    K_rb = B.T @ WA                # (3,N)

    # q = - M_rb^{-1} K_rb gdot
    P = -np.linalg.solve(M_rb, K_rb)   # (3,N)

    # Eliminated hydrodynamic generalized friction:
    # G_h = H gdot = A^T W (A gdot + B q)
    #      = A^T (WA + WB P) gdot
    H = A.T @ (WA + WB @ P)            # (N,N)

    return H, P, t, n, r_rel


def build_bc_replaced_sparse_part(
    gamma,
    n_plus,
    n_minus,
    params,
    D2,
    H,
):

    N = params.N
    dt = params.dt
    ds = 1/(N-1)

    nsum = n_plus + n_minus
    ndiff = n_minus - n_plus

    # friction coefficient
    c = params.beta + params.zeta * params.mu_a * nsum

    Cfric = sp.diags(c, 0)

    I = sp.eye(N)

    A_sparse = D2 - params.mu * I - Cfric/dt

    rhs = -params.mu_a * ndiff - (c/dt) * gamma
    rhs -= (H @ gamma) / dt

    A = A_sparse.tolil()

    # base sliding fixed
    A[0,:] = 0
    A[0,0] = 1
    rhs[0] = 0

    # free tip
    A[N-1,:] = 0
    A[N-1,N-2] = -1/ds
    A[N-1,N-1] = 1/ds
    rhs[N-1] = 0

    return A.tocsr(), rhs

def solve_linear_step(
    A_sparse: sp.csr_matrix,
    H: np.ndarray,
    rhs: np.ndarray,
    params: Params,
):
    """
    Solve
        (A_sparse - H/dt) gamma_new = rhs
    with GMRES and a sparse-LU preconditioner for A_sparse.
    """
    dt = params.dt
    N = params.N

    A_lu = spla.splu(A_sparse.tocsc())

    def matvec(x):
        return A_sparse @ x - (H @ x) / dt

    # explicitly name the shape parameter to appease type checkers
    Aop = LinearOperator(shape=(N, N), matvec=matvec, dtype=float)
    Mop = LinearOperator(shape=(N, N), matvec=A_lu.solve, dtype=float)

    try:
        sol, info = gmres(
            Aop, rhs, M=Mop,
            restart=80,
            maxiter=params.gmres_maxiter,
            rtol=params.gmres_rtol,
            atol=0.0,
        )
    except TypeError:
        sol, info = gmres(
            Aop, rhs, M=Mop,
            restart=80,
            maxiter=params.gmres_maxiter,
            tol=params.gmres_rtol,
        )
    if info != 0:
        raise RuntimeError(f"GMRES failed, info={info}")
    return sol


def update_motors(n_plus,n_minus,gdot,params):

    dt=params.dt

    exp_p = np.exp(params.fstar*(1 + params.zeta*gdot))
    exp_m = np.exp(params.fstar*(1 - params.zeta*gdot))

    n_plus += dt*( params.eta*(1-n_plus)
        - (1-params.eta)*n_plus*exp_p )

    n_minus += dt*( params.eta*(1-n_minus)
        - (1-params.eta)*n_minus*exp_m )

    return np.clip(n_plus,0,1),np.clip(n_minus,0,1)

def run_simulation(params: Params):
    N = params.N
    ds = 1.0 / (N - 1)
    s = np.linspace(0.0, 1.0, N)
    n_steps = int(np.round(params.T / params.dt))

    D2 = build_second_derivative_matrix(N, ds)
    C = build_cumulative_matrix(N, ds)

    # initial condition: small bias to leave the symmetric state
    gamma = 1e-4 * np.sin(np.pi * s)
    n_ss = params.eta
    n_plus = np.full(N, n_ss)
    n_minus = np.full(N, n_ss)

    r0 = np.zeros(2)
    theta0 = 0.0

    saved_gamma = []
    saved_n_plus = []
    saved_n_minus = []
    saved_time = []
    saved_theta0 = []
    saved_r0 = []

    for step in range(n_steps):
        # build hydrodynamic operator from current geometry
        H, P, _, _, _ = build_hydro_operators(gamma, theta0, params, C)

        # linear semi-implicit solve for gamma^{n+1}
        A_sparse, rhs = build_bc_replaced_sparse_part(
            gamma, n_plus, n_minus, params, D2, H
        )
        gamma_new = solve_linear_step(A_sparse, H, rhs, params)

        # base rigid-body velocities from current step's gammadot
        gdot = (gamma_new - gamma) / params.dt
        q = P @ gdot
        U = q[:2]
        Omega = q[2]

        # update base motion
        r0 = r0 + params.dt * U
        theta0 = theta0 + params.dt * Omega

        # update motor populations
        n_plus, n_minus = update_motors(n_plus, n_minus, gdot, params)

        gamma = gamma_new

        if step % params.save_every == 0:
            saved_gamma.append(gamma.copy())
            saved_time.append(step * params.dt)
            saved_theta0.append(theta0)
            saved_r0.append(r0.copy())
            saved_n_plus.append(n_plus.copy())
            saved_n_minus.append(n_minus.copy())

    return {
        "s": s,
        "time": np.array(saved_time),
        "gamma": np.array(saved_gamma),
        "n_plus": np.array(saved_n_plus),
        "n_minus": np.array(saved_n_minus),
        "theta0": np.array(saved_theta0),
        "r0": np.array(saved_r0),
    }

def run_simulation_jump(params: Params, Nmotor=300, seed=0):
    """
    Same as run_simulation, but motor dynamics are treated as a Poisson jump process.
    Hydrodynamics + gamma solver are unchanged.
    """

    rng = np.random.default_rng(seed)

    N = params.N
    ds = 1.0 / (N - 1)
    s = np.linspace(0.0, 1.0, N)
    n_steps = int(np.round(params.T / params.dt))

    D2 = build_second_derivative_matrix(N, ds)
    C = build_cumulative_matrix(N, ds)

    # initial condition
    gamma = 1e-4 * np.sin(np.pi * s)

    # steady-state motor fraction
    b = params.eta + (1 - params.eta) * np.exp(params.fstar)
    n0 = params.eta / b
    n0 = np.round(n0 * Nmotor) / Nmotor

    n_plus = np.full(N, n0)
    n_minus = np.full(N, n0)

    r0 = np.zeros(2)
    theta0 = 0.0

    saved_gamma = []
    saved_n_plus = []
    saved_n_minus = []
    saved_time = []
    saved_theta0 = []
    saved_r0 = []

    Nm_dt = Nmotor * params.dt
    one_minus_eta = 1 - params.eta

    # helper: fast poisson
    def sample_poisson(lam):
        if lam <= 0:
            return 0
        if lam < 1e-3:
            return 1 if rng.random() < lam else 0
        return rng.poisson(lam)

    for step in range(n_steps):

        # --- hydrodynamics ---
        H, P, _, _, _ = build_hydro_operators(gamma, theta0, params, C)

        # --- gamma solve ---
        A_sparse, rhs = build_bc_replaced_sparse_part(
            gamma, n_plus, n_minus, params, D2, H
        )
        gamma_new = solve_linear_step(A_sparse, H, rhs, params)

        gdot = (gamma_new - gamma) / params.dt

        # --- rigid body ---
        q = P @ gdot
        U = q[:2]
        Omega = q[2]

        r0 += params.dt * U
        theta0 += params.dt * Omega

        # ============================================================
        # STOCHASTIC MOTOR UPDATE (Poisson jump process)
        # ============================================================
        for i in range(N):

            zg = params.zeta * gdot[i]

            exp_p = np.exp(params.fstar * (1 + zg))
            exp_m = np.exp(params.fstar * (1 - zg))

            bind_p = params.eta * (1 - n_plus[i]) * Nm_dt
            bind_m = params.eta * (1 - n_minus[i]) * Nm_dt

            unbind_p = one_minus_eta * n_plus[i] * exp_p * Nm_dt
            unbind_m = one_minus_eta * n_minus[i] * exp_m * Nm_dt

            dNp = sample_poisson(bind_p) - sample_poisson(unbind_p)
            dNm = sample_poisson(bind_m) - sample_poisson(unbind_m)

            n_plus[i] = np.clip(n_plus[i] + dNp / Nmotor, 0.0, 1.0)
            n_minus[i] = np.clip(n_minus[i] + dNm / Nmotor, 0.0, 1.0)

        gamma = gamma_new

        # --- save ---
        if step % params.save_every == 0:
            saved_gamma.append(gamma.copy())
            saved_time.append(step * params.dt)
            saved_theta0.append(theta0)
            saved_r0.append(r0.copy())
            saved_n_plus.append(n_plus.copy())
            saved_n_minus.append(n_minus.copy())

    return {
        "s": s,
        "time": np.array(saved_time),
        "gamma": np.array(saved_gamma),
        "n_plus": np.array(saved_n_plus),
        "n_minus": np.array(saved_n_minus),
        "theta0": np.array(saved_theta0),
        "r0": np.array(saved_r0),
    }

def plot_kymograph(result):
    gamma = result["gamma"].T       # (s, t)
    n_plus = result["n_plus"].T
    n_minus = result["n_minus"].T

    s = result["s"]
    time = result["time"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    extent = (time[0], time[-1], s[0], s[-1])

    # --- gamma ---
    im0 = axes[0].imshow(
        gamma,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[0].set_ylabel(r"$s$")
    axes[0].set_title(r"$\gamma(s,t)$")
    plt.colorbar(im0, ax=axes[0])

    # --- n_plus ---
    im1 = axes[1].imshow(
        n_plus,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[1].set_ylabel(r"$s$")
    axes[1].set_title(r"$n_+(s,t)$")
    plt.colorbar(im1, ax=axes[1])

    # --- n_minus ---
    im2 = axes[2].imshow(
        n_minus,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    axes[2].set_ylabel(r"$s$")
    axes[2].set_xlabel(r"$t$")
    axes[2].set_title(r"$n_-(s,t)$")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

def plot_base_motion(result):
    r0 = result["r0"]
    theta0 = result["theta0"]
    time = result["time"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    axes[0].plot(time, r0[:, 0], label=r"$r_{0x}$")
    axes[0].plot(time, r0[:, 1], label=r"$r_{0y}$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_title("base translation")
    axes[0].legend()

    axes[1].plot(time, theta0)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_title(r"base angle $\theta_0(t)$")

    plt.tight_layout()
    plt.show()

import time
import csv
from cilia.algorithms.a_estimation import estimate_amplitude_psd
from cilia.algorithms.f_estimation import estimate_f_from_tangent_angle_power


def main():
    lambda_values = [0.01, 0.1, 0.5, 1, 1.5,  2, 2.5, 5, 10, 15]
    beta_values = [1.0, 2.0, 3.0]

    results = []

    for beta in beta_values:
        for lam in lambda_values:
            print(f"\nRunning lambda_h = {lam}, beta = {beta}")

            params = Params(
                mu_a=1570,
                mu=10,
                zeta=0.96,
                eta=0.096,
                fstar=2.0,
                beta=beta,
                lambda_h=lam,
                N=75,
                dt=5e-3,
                T=1000,
            )

            t0 = time.time()
            result = run_simulation(params)
            t1 = time.time()

            gamma = result["gamma"]      # (time, s)
            time_arr = result["time"]
            s = result["s"]

            # -----------------------------
            # REMOVE TRANSIENT
            # -----------------------------
            mask = time_arr >= 10.0
            gamma = gamma[mask]
            time_arr = time_arr[mask]

            # effective dt (important!)
            dt_eff = params.dt * params.save_every
            print(dt_eff, time_arr[1]-time_arr[0])
            # -----------------------------
            # AMPLITUDE (cilia version)
            # -----------------------------
            A = estimate_amplitude_psd(
                tangent_angles=gamma,
                arclength=s,
            )

            # -----------------------------
            # FREQUENCY (cilia version)
            # -----------------------------
            freq = estimate_f_from_tangent_angle_power(
                tangent_angles=gamma,
                arclength=s,
                dt_frame=dt_eff,
            )

            results.append([lam, beta, A, freq])

            print(
                f"lambda_h={lam:>6}, beta={beta:>4}, "
                f"A={A:.4e}, f={freq:.4f}, runtime={t1-t0:.2f}s"
            )

    # -----------------------------
    # SAVE CSV
    # -----------------------------
    with open("results_hydro_fluc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda_h", "beta", "amplitude", "frequency"])
        writer.writerows(results)

    print("\nSaved results to results_hydro_fluc.csv")

    # -----------------------------
    # OPTIONAL: quick plot
    # -----------------------------
    results = np.array(results)

    plt.figure(figsize=(6,4))
    for beta in beta_values:
        mask = results[:,1] == beta
        plt.plot(results[mask,0], results[mask,2], "o-", label=f"beta={beta}")

    plt.xscale("log")
    plt.xlabel(r"$\lambda_h$")
    plt.ylabel(r"Amplitude")
    plt.title("Amplitude vs hydrodynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_results_from_csv(filename="results_hydro_fluc.csv"):

    data = np.genfromtxt(filename, delimiter=",", names=True)

    lam = data["lambda_h"]
    beta = data["beta"]
    amp = data["amplitude"]
    freq = data["frequency"]

    beta_unique = np.unique(beta)

    # fixed colors
    colors = ["#b8b8b8", "#5b5c5b", "#000000"]

    # font sizes
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 13

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for i, b in enumerate(beta_unique):
        mask = beta == b
        color = colors[i % len(colors)]

        idx = np.argsort(lam[mask])

        label = rf"$\beta={b}$"

        axes[0].plot(
            lam[mask][idx],
            amp[mask][idx],
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label
        )

        axes[1].plot(
            lam[mask][idx],
            freq[mask][idx] * 250,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label
        )

    # formatting
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Lambda$", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

        # vertical dashed line at lambda = 2
        ax.axvline(2, linestyle="--", color="black", linewidth=1.5)

    axes[0].set_ylabel(r"Amplitude $A$ [rad]", fontsize=LABEL_SIZE)
    axes[0].set_ylim(0, 1)

    axes[1].set_ylabel(r"Frequency $f_0$", fontsize=LABEL_SIZE)
    axes[1].set_ylim(0, 100)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        #loc="center right",
        #title=r"$\beta$",
        fontsize=LEGEND_SIZE,
        #title_fontsize=LEGEND_SIZE
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # type: ignore
    plt.show()

if __name__ == "__main__":
    #main()
    plot_results_from_csv()
