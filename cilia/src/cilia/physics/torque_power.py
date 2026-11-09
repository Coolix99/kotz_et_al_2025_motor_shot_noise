from __future__ import annotations
import numpy as np

# Elastic torque
def compute_elastic_torque(gamma, s, B, d2K):
    dgamma_ds = np.gradient(gamma, s, axis=1)
    d2gamma_ds2 = np.gradient(dgamma_ds, s, axis=1)
    # print(s)
    # print(d2gamma_ds2)
    # print(np.max(d2gamma_ds2), np.min(d2gamma_ds2))
    # print(np.sqrt(np.mean(d2gamma_ds2*d2gamma_ds2)))
    
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    # ax=axes[0]
    # im = ax.imshow(
    #             d2gamma_ds2,
    #             aspect="auto",
    #             origin="lower",
    #             #extent=[s.min(), s.max(), t.min(), t.max()],
    #             cmap="RdBu_r",
    #             # vmin=vmin,
    #             # vmax=vmax,
    #         )
    # ax.set_title('d2gamma_ds2')
    # ax.set_xlabel("s")
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('d2gamma_ds2')
    # ax.set_ylabel("time t")
    # plt.tight_layout()
    # plt.show()
    # raise

    tau = B * d2gamma_ds2 - d2K * gamma

    return tau

def periodic_time_derivative(gamma, dt):
    g_m2 = np.roll(gamma,  2, axis=0)
    g_m1 = np.roll(gamma,  1, axis=0)
    g_p1 = np.roll(gamma, -1, axis=0)
    g_p2 = np.roll(gamma, -2, axis=0)

    dgamma_dt = (-2.0 * g_m2 - 1.0 * g_m1 + 1.0 * g_p1 + 2.0 * g_p2) / (10.0 * dt)
    return dgamma_dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def _clean_xy(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    return x[mask], y[mask]


# minimum-area triangle containing q fraction
# boundary: x/a + y/b <= 1
# equivalently: m x + y <= c
def fit_min_area_triangle(x, y, q=0.9, m_grid=None):
    x, y = _clean_xy(x, y)

    if len(x) == 0:
        raise ValueError("No valid points")

    if m_grid is None:
        # broad log-grid for slope
        m_grid = np.logspace(-3, 3, 2000)

    best = None

    for m in m_grid:
        z = m * x + y
        c = np.quantile(z, q)

        # line: m x + y = c
        # intercepts: a = c/m, b = c
        a = c / m
        b = c
        area = 0.5 * a * b

        if best is None or area < best["area"]:
            best = {
                "m": m,
                "c": c,
                "a": a,
                "b": b,
                "area": area,
                "coverage": np.mean(z <= c),
            }

    return best


def plot_triangle(ax, fit, **kwargs):
    a = fit["a"]
    b = fit["b"]

    xx = np.array([0.0, a, 0.0])
    yy = np.array([0.0, 0.0, b])

    ax.plot([0, a], [b, 0], **kwargs)
    ax.plot([0, 0], [0, b], alpha=0.15, color=kwargs.get("color", "black"))
    ax.plot([0, a], [0, 0], alpha=0.15, color=kwargs.get("color", "black"))


# nonlinear generalization: superellipse
# (x/a)^p + (y/b)^p <= 1
def quarter_superellipse_area(a, b, p):
    # area in first quadrant
    # A = a*b * Gamma(1+1/p)^2 / Gamma(1+2/p)
    return a * b * (gamma(1 + 1/p) ** 2) / gamma(1 + 2/p)


def fit_min_area_superellipse(
    x,
    y,
    q=0.9,
    p_grid=None,
    r_grid=None,
):
    """
    Fit smallest-area quarter-superellipse containing fraction q.

    We parameterize a = r * b.
    Then score:
        u = (x/a)^p + (y/b)^p
          = (x^p / r^p + y^p) / b^p
    For fixed (p, r), minimal b for q coverage is:
        b = quantile(x^p / r^p + y^p, q)^(1/p)
    """
    x, y = _clean_xy(x, y)

    if len(x) == 0:
        raise ValueError("No valid points")

    if p_grid is None:
        p_grid = np.concatenate([
            np.linspace(0.4, 1.0, 30),
            np.linspace(1.05, 4.0, 60),
        ])

    if r_grid is None:
        r_grid = np.logspace(-2, 2, 300)

    best = None

    for p in p_grid:
        xp = x ** p
        yp = y ** p

        for r in r_grid:
            u = xp / (r ** p) + yp
            bp = np.quantile(u, q)
            b = bp ** (1.0 / p)
            a = r * b

            area = quarter_superellipse_area(a, b, p)
            coverage = np.mean((x / a) ** p + (y / b) ** p <= 1.0 + 1e-12)

            if best is None or area < best["area"]:
                best = {
                    "p": p,
                    "r": r,
                    "a": a,
                    "b": b,
                    "area": area,
                    "coverage": coverage,
                }

    return best


def plot_superellipse(ax, fit, n=775, **kwargs):
    a = fit["a"]
    b = fit["b"]
    p = fit["p"]

    xx = np.linspace(0, a, n)
    yy = b * np.maximum(1.0 - (xx / a) ** p, 0.0) ** (1.0 / p)
    ax.plot(xx, yy, **kwargs)


# full plotting function
def work_region_envelopes(
    dtgamma_work,
    tau_work,
    q=0.9,
    debug_plot=False
):
    x, y = _clean_xy(dtgamma_work, tau_work)

    tri = fit_min_area_triangle(x, y, q=q)
    sup = fit_min_area_superellipse(x, y, q=q)

    if debug_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=5, alpha=0.25, color="black", label="work points")

        plot_triangle(
            ax, tri,
            color="tab:red",
            linewidth=2,
            label=f"triangle ({100*q:.0f}%)"
        )

        plot_superellipse(
            ax, sup,
            color="tab:blue",
            linewidth=2,
            label=f"superellipse p={sup['p']:.2f}" # type: ignore
        )

        
        ax.set_xlabel(r"$|\partial_t \gamma|$")
        ax.set_ylabel(r"$|\tau_e|$")
        ax.set_title("Minimal envelopes of work-producing points")
        ax.grid(True)
        ax.legend()

        print("---- TRIANGLE ----")
        print(f"coverage = {tri['coverage']:.3f}") # type: ignore
        print(f"a = {tri['a']:.4g}") # type: ignore
        print(f"b = {tri['b']:.4g}") # type: ignore
        print(f"area = {tri['area']:.4g}") # type: ignore

        print("\n---- SUPERELLIPSE ----") 
        print(f"coverage = {sup['coverage']:.3f}")# type: ignore
        print(f"p = {sup['p']:.4g}") # type: ignore
        print(f"a = {sup['a']:.4g}") # type: ignore
        print(f"b = {sup['b']:.4g}") # type: ignore
        print(f"area = {sup['area']:.4g}") # type: ignore


        plt.tight_layout()
        plt.show()

    return tri, sup

import matplotlib.pyplot as plt

def v0_F0_from_fits(
    sup_fit,
    di,
    d,
    rho,
    N_curve=75,
    N_v0=400,
    v0_min=0.1,
    v0_max=30.0,
    eps=1e-9,
    debug_plot=True,
    title=None,
):
    """
    For each projected gap i:
        F_i = Fstall * (1 - d * di_i * gamma_t / v0)
        tau_i = rho * d * di_i * F_i

    Ideal detached:
        F_i = Fstall if d * di_i * gamma_t < v0, else 0

    Linear detached:
        F_i = Fstall * max(1 - d * di_i * gamma_t / v0, 0)

    Linear with drag:
        F_i = Fstall * (1 - d * di_i * gamma_t / v0)
    """
    di = np.asarray(di, dtype=float)
    di = di[np.isfinite(di) & (di > 0)]

    if len(di) == 0:
        raise ValueError("di must contain positive finite entries.")
    if rho <= 0 or d <= 0:
        raise ValueError("rho and d must be positive.")

    Sd = float(np.sum(di))
    S2 = float(np.sum(di**2))

    a = float(sup_fit["a"])
    b = float(sup_fit["b"])
    p = float(sup_fit["p"])

    Fstall_min = b / (rho * d * Sd)
    # print('b', b)
    # print('rho', rho)
    # print('d', d)
    # print('Sd',Sd)
    v0_min_detach = d * a * np.min(di)
    v0_min_linear_drag = d * a * S2 / Sd

    x_vals = np.linspace(0.0, a, N_curve)
    tau_vals = b * np.maximum(1.0 - (x_vals / a) ** p, 0.0) ** (1.0 / p)

    def envelope(v0_vals, denom_func):
        all_limits = []
        for dtg, tau in zip(x_vals, tau_vals):
            if tau <= 0:
                continue
            denom = denom_func(dtg, v0_vals)
            F0 = np.full_like(v0_vals, np.nan)
            valid = denom > eps
            F0[valid] = tau / (rho * denom[valid])
            all_limits.append(F0)

        if not all_limits:
            return np.full_like(v0_vals, np.nan)

        stack = np.vstack(all_limits)
        out = np.full_like(v0_vals, np.nan)
        valid_cols = np.any(np.isfinite(stack), axis=0)
        out[valid_cols] = np.nanmax(stack[:, valid_cols], axis=0)
        return out

    v0_start_detach = max(v0_min, v0_min_detach * (1.0 + 1e-6))
    v0_start_drag = max(v0_min, v0_min_linear_drag * (1.0 + 1e-6))

    v0_grid_detach = np.linspace(v0_start_detach, v0_max, N_v0)
    v0_grid_linear_drag = np.linspace(v0_start_drag, v0_max, N_v0)

    thresholds = (d * di[:, None] * x_vals[None, :]).ravel()
    thresholds = thresholds[
        np.isfinite(thresholds)
        & (thresholds >= v0_start_detach)
        & (thresholds <= v0_max)
    ]
    ideal_step_v0 = np.unique(np.r_[v0_start_detach, np.sort(thresholds), v0_max])

    def denom_ideal(dtg, v0_vals):
        active = d * di[:, None] * dtg < v0_vals[None, :]
        return d * np.sum(di[:, None] * active, axis=0)

    def denom_linear_detach(dtg, v0_vals):
        force = 1.0 - d * di[:, None] * dtg / v0_vals[None, :]
        return d * np.sum(di[:, None] * np.maximum(force, 0.0), axis=0)

    def denom_linear_drag(dtg, v0_vals):
        return d * (Sd - d * dtg * S2 / v0_vals)

    ideal_step_F0 = envelope(
        ideal_step_v0 * (1.0 + 1e-10),
        denom_ideal,
    )

    F0_linear_detach = envelope(
        v0_grid_detach,
        denom_linear_detach,
    )

    F0_linear_drag = envelope(
        v0_grid_linear_drag,
        denom_linear_drag,
    )

    if debug_plot:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            ideal_step_v0,
            ideal_step_F0,
            #where="post",
            color="tab:green",
            linewidth=2.2,
            linestyle="-",
            label="ideal, detached",
        )

        ax.plot(
            v0_grid_detach,
            F0_linear_detach,
            color="tab:blue",
            linewidth=2.2,
            linestyle="--",
            label="linear, detached",
        )

        ax.plot(
            v0_grid_linear_drag,
            F0_linear_drag,
            color="tab:red",
            linewidth=2.2,
            linestyle="-.",
            label="linear, with drag",
        )

        ax.axhline(
            Fstall_min,
            color="black",
            linestyle=":",
            linewidth=2,
            label=rf"$\check F_{{stall}}={Fstall_min:.2g}$ pN",
        )

        ax.axvline(
            v0_min_detach,
            color="tab:blue",
            linestyle=":",
            linewidth=1.8,
            label=rf"$\check v_0={v0_min_detach:.2g}$ $\mu$m/s",
        )

        ax.axvline(
            v0_min_linear_drag,
            color="tab:red",
            linestyle=":",
            linewidth=1.8,
            label=rf"$\tilde v_0={v0_min_linear_drag:.2g}$ $\mu$m/s",
        )

        ax.set_xlabel(r"$v_0$ [$\mu$m/s]")
        ax.set_ylabel(r"$F_\mathrm{stall}$ [pN]")
        ax.set_title("Motor force constraints" if title is None else title)
        ax.set_xlim(v0_min, v0_max)
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return {
        "Fstall_min": Fstall_min,
        "v0_min_detach": v0_min_detach,
        "v0_min_linear_drag": v0_min_linear_drag,
        "ideal_step_v0": ideal_step_v0,
        "ideal_step_F0": ideal_step_F0,
        "v0_grid_detach": v0_grid_detach,
        "F0_linear_detach": F0_linear_detach,
        "v0_grid_linear_drag": v0_grid_linear_drag,
        "F0_linear_drag": F0_linear_drag,
        "x_vals": x_vals,
        "tau_vals": tau_vals,
    }


def compute_torque_power_bounds(
    theta,
    s,
    dt,
    *,
    B,
    d2K,
    d,
    di,
    rho_gap,
    debug_plot=False,
):
    gamma = theta - theta[:, [0]]
    Nt, Ns = gamma.shape
    t = np.arange(Nt) * dt

    tau_e = compute_elastic_torque(gamma, s, B, d2K)
    dtgamma_dt = periodic_time_derivative(gamma, dt)

    if debug_plot:
        def sym_limits(arr):
            vmax = np.nanmax(np.abs(arr))
            return -vmax, vmax

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        data = [
            (gamma, *sym_limits(gamma), r"$\gamma$"),
            (tau_e, *sym_limits(tau_e), r"$m_e$"),
            (dtgamma_dt, *sym_limits(dtgamma_dt), r"$\partial_t \gamma$"),
        ]

        for ax, (Z, vmin, vmax, label) in zip(axes, data):
            im = ax.imshow(
                Z,
                aspect="auto",
                origin="lower",
                extent=[s.min(), s.max(), t.min(), t.max()],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(label)
            ax.set_xlabel("s")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(label)

        axes[0].set_ylabel("time t")
        plt.tight_layout()
        plt.show()

    # Motor work uses tau_m = -m_e.
    motor_power_density = -tau_e * dtgamma_dt

    diss_power = np.minimum(motor_power_density, 0.0)
    work_power = np.maximum(motor_power_density, 0.0)

    E_diss = -np.sum(np.trapezoid(diss_power, s, axis=1)) * dt
    E_work = np.sum(np.trapezoid(work_power, s, axis=1)) * dt

    W = 0.5 * (E_diss + E_work)
    N_ATP = W / 0.08

    work_mask = motor_power_density > 0

    tau_work = np.abs(tau_e[work_mask])
    dtgamma_work = np.abs(dtgamma_dt[work_mask])

    mask = np.isfinite(tau_work) & np.isfinite(dtgamma_work)
    tau_work = tau_work[mask]
    dtgamma_work = dtgamma_work[mask]

    if debug_plot:
        plt.figure(figsize=(4, 4))
        plt.scatter(dtgamma_work, tau_work, s=5, alpha=0.3)
        plt.xlabel(r"$|\partial_t \gamma|$")
        plt.ylabel(r"$|m_e|$")
        plt.title("Motor-work-producing regions")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    _, sup_fit_85 = work_region_envelopes(
        dtgamma_work,
        tau_work,
        q=0.85,
    )

    _, sup_fit_75 = work_region_envelopes(
        dtgamma_work,
        tau_work,
        q=0.75,
        debug_plot=debug_plot
    )
    
    res_85 = v0_F0_from_fits(
        sup_fit_85,
        di=di,
        d=d,
        rho=rho_gap,
        debug_plot=debug_plot,
        title="Motor force constraints, q85",
    )
    
    res_75 = v0_F0_from_fits(
        sup_fit_75,
        di=di,
        d=d,
        rho=rho_gap,
        debug_plot=debug_plot,
        title="Motor force constraints, q75",
    )

    return {
        "q85": res_85,
        "q75": res_75,
        "work": W,
        "N_ATP": N_ATP,
    }
