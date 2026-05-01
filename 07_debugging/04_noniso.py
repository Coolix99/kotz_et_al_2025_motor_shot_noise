import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from scipy.ndimage import gaussian_filter
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
)
from open_res import read_spde
from cilia.datasets.data_loaders import register_data_loader

class SimulationDataset(DataSet):

    def __init__(self, path: str):
        super().__init__(path)

    def _initialize_realizations(self):
        if not os.path.isdir(self.path):
            self.realizations = []
            return

        exp_dirs = [
            d for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]

        self.realizations = [
            Realization(exp_id)
            for exp_id in sorted(exp_dirs)
        ]

@register_data_loader(
    dataset_cls=SimulationDataset,
    name=TANGENT_ANGLE_SERIES.key,
    algorithm=None,
)
def load_spde_tangent_angle_series(exp_path, data_meta):
    fname = data_meta["metadata"]["filename"]

    fpath = os.path.join(exp_path, fname)
    data = read_spde(fpath)
    gamma = data["gamma_mat"]          # (Nt, Ns)

    return gamma

class SharmaDataset(DataSet):
    def _initialize_realizations(self):
        if not os.path.isdir(self.path):
            self.realizations = []
            return

        self.realizations = [
            Realization(d)
            for d in sorted(os.listdir(self.path))
            if os.path.isdir(os.path.join(self.path, d))
        ]


def get_realization(dataset, exp_id):
    for r in dataset:
        if r.experiment_id == exp_id:
            return r
    raise ValueError(f"Experiment {exp_id} not found")

def get_exp_path(dataset, realization):
    return os.path.join(dataset.path, realization.experiment_id)

def inspect_experiment(dataset, exp_ids):
    all_amp = []
    all_omega = []
    records = []

    for exp_id in exp_ids:
        realization = get_realization(dataset, exp_id)
        exp_path = get_exp_path(dataset, realization)

        phase_item = None
        dt_item = None

        for item in realization.data_items.values():
            if item.data_name.key == "phase_segments" and "protophase_from_spatial_modes_segment_t" in item.algorithm:
                phase_item = item
            if item.data_name.key == "dt_frame":
                dt_item = item
        
        if not (phase_item and dt_item):
            print(f"⚠️ Missing data for {exp_id}")
            continue

        data_list = phase_item.resolve(dataset, exp_path)
        dt_frame = dt_item.resolve(dataset, exp_path)

        for data in data_list:

            phi = np.asarray(data["phase"])
            amp = np.asarray(data["rel_amplitude"])
            if phi.shape[0]<1000:
                continue
            if not np.isfinite(phi).any():
                print(f"[SKIP] {exp_id}: phase NaN")
                continue

            # unwrap + orientation
            phi = np.unwrap(phi)
            if phi[-1] - phi[0] < 0:
                phi = -phi
            
            omega = np.gradient(phi, dt_frame)

            # remove bad values
            mask = np.isfinite(omega) & np.isfinite(amp)
            amp=amp[mask]
            omega=omega[mask]
            amp_mean = np.mean(amp)
            omega_mean = np.mean(omega)
            amp_rel = (amp - amp_mean) / amp_mean
            omega_rel = (omega - omega_mean) / omega_mean

            all_amp.append(amp_rel)
            all_omega.append(omega_rel)
            records.append({
                "phi": phi[mask],
                "amp_rel": amp_rel,
                "omega_rel": omega_rel,
                "exp_id": exp_id,
            })

    if len(all_amp) == 0:
        print("No valid data collected")
        return

    all_amp = np.concatenate(all_amp)
    all_omega = np.concatenate(all_omega)


    mask = (
        np.abs(all_amp) <= 0.5
    ) & (
        np.abs(all_omega) <= 0.5
    )

    all_amp = all_amp[mask]
    all_omega = all_omega[mask]

    return all_omega, all_amp, records

def inspect_simulation(dataset, exp_ids_N20):
    all_amp = []
    all_omega = []
    records = []

    for exp_id in exp_ids_N20:
        realization = get_realization(dataset, exp_id)
        exp_path = get_exp_path(dataset, realization)

        phase_item = None
        dt_item = None

        for item in realization.data_items.values():
            if item.data_name.key == "phase_series" and "estimate_phase_from_protophase_t" in item.algorithm:
                phase_item = item
            if item.data_name.key == "dt_frame":
                dt_item = item

        if not (phase_item and dt_item):
            print(f"⚠️ Missing data for {exp_id}")
            continue

        data = phase_item.resolve(dataset, exp_path)
        dt_frame = dt_item.resolve(dataset, exp_path)

        phi = np.asarray(data["phase"])
        amp = np.asarray(data["rel_amplitude"])

        if not np.isfinite(phi).any():
            print(f"[SKIP] {exp_id}: phase NaN")
            continue

        # unwrap + orientation
        phi = np.unwrap(phi)
        if phi[-1] - phi[0] < 0:
            phi = -phi

        omega = np.gradient(phi, dt_frame)

        # remove bad values
        mask = np.isfinite(omega) & np.isfinite(amp)
        amp=amp[mask]
        omega=omega[mask]
        amp_mean = np.mean(amp)
        omega_mean = np.mean(omega)
        amp_rel = (amp - amp_mean) / amp_mean
        omega_rel = (omega - omega_mean) / omega_mean

        all_amp.append(amp_rel)
        all_omega.append(omega_rel)
        records.append({
            "phi": phi[mask],
            "amp_rel": amp_rel,
            "omega_rel": omega_rel,
            "exp_id": exp_id,
        })


    if len(all_amp) == 0:
        print("No valid data collected")
        return

    all_amp = np.concatenate(all_amp)
    all_omega = np.concatenate(all_omega)



    mask = (
        np.abs(all_amp) <= 0.5
    ) & (
        np.abs(all_omega) <= 0.5
    )

    all_amp = all_amp[mask]
    all_omega = all_omega[mask]

    return all_omega, all_amp, records

def fit_phase_dependent_nonisochrony(phi, amp_rel, omega_rel, nharm=3, ridge=1e-6):
    phi = np.asarray(phi)
    amp_rel = np.asarray(amp_rel)
    omega_rel = np.asarray(omega_rel)

    mask = np.isfinite(phi) & np.isfinite(amp_rel) & np.isfinite(omega_rel)
    phi = phi[mask]
    amp_rel = amp_rel[mask]
    omega_rel = omega_rel[mask]

    phi = np.mod(phi, 2*np.pi)

    cols = [amp_rel]
    for n in range(1, nharm + 1):
        cols.append(amp_rel * np.cos(n * phi))
        cols.append(amp_rel * np.sin(n * phi))

    X = np.column_stack(cols)
    y = omega_rel

    A = X.T @ X + ridge * np.eye(X.shape[1])
    b = X.T @ y
    beta = np.linalg.solve(A, b)

    return beta


def evaluate_nonisochrony_fourier(beta, phi_grid, nharm):
    c = beta[0] * np.ones_like(phi_grid)

    j = 1
    for n in range(1, nharm + 1):
        a = beta[j]
        b = beta[j + 1]
        c += a * np.cos(n * phi_grid) + b * np.sin(n * phi_grid)
        j += 2

    return c


def dominant_harmonic_shift(beta, nharm):
    """
    Gauge fixing:
    shift phase so the dominant harmonic of c(phi) has maximum at phi=0.
    """
    amps = []
    phases = []

    j = 1
    for n in range(1, nharm + 1):
        a = beta[j]
        b = beta[j + 1]

        amp = np.hypot(a, b)
        phase = np.arctan2(b, a) / n

        amps.append(amp)
        phases.append(phase)

        j += 2

    amps = np.asarray(amps)
    phases = np.asarray(phases)

    if np.max(amps) < 1e-12:
        return 0.0

    n_dom = np.argmax(amps)
    return phases[n_dom]


def average_phase_dependent_nonisochrony(records, nharm=3, ngrid=200):
    phi_grid = np.linspace(0, 2*np.pi, ngrid, endpoint=False)

    curves = []

    for rec in records:
        beta = fit_phase_dependent_nonisochrony(
            rec["phi"],
            rec["amp_rel"],
            rec["omega_rel"],
            nharm=nharm,
        )

        shift = dominant_harmonic_shift(beta, nharm)

        # evaluate in aligned gauge
        c = evaluate_nonisochrony_fourier(
            beta,
            phi_grid + shift,
            nharm=nharm,
        )

        curves.append(c)

    curves = np.asarray(curves)

    mean = np.mean(curves, axis=0)
    sem = np.std(curves, axis=0) / np.sqrt(curves.shape[0])

    return phi_grid, mean, sem

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FuncFormatter
def plot_amp_omega_phase_space(
    amp_rel_2d, omega_rel_2d, records_2d,
    amp_rel_3d, omega_rel_3d, records_3d,
    amp_rel_exp, omega_rel_exp, records_exp,
    bins=100,
    sigma_smooth=2.0,
    fit_mode="linear",
    nharm_phase=3,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, (ax_main, ax_phase) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    def plot_density(ax, x, y, color):
        x = (1 + x) * 100
        y = (1 + y) * 100

        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=bins,
            range=[[80, 120], [80, 120]],
            density=True,
        )

        H = gaussian_filter(H, sigma=sigma_smooth)
        H /= np.max(H)

        Xc = 0.5 * (xedges[:-1] + xedges[1:])
        Yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(Xc, Yc, indexing="ij")

        sigma_levels = [2, 1, 0.5, 0.25]
        levels = [np.exp(-0.5 * s**2) for s in sigma_levels]

        ax.contour(X, Y, H, levels=levels, colors=color, linewidths=2)

    def fit_and_plot(ax, x, y, color, mode):
        x = np.asarray(x)
        y = np.asarray(y)

        if mode == "pca":
            x_p = (1 + x) * 100
            y_p = (1 + y) * 100

            X = np.vstack([x_p, y_p]).T
            mean = np.mean(X, axis=0)
            Xc = X - mean

            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            dx, dy = Vt[0]

            slope = dy / dx if np.abs(dx) > 1e-12 else np.inf

            t = np.linspace(-20, 20, 200)
            line = mean + t[:, None] * np.array([dx, dy])
            ax.plot(line[:, 0], line[:, 1], color=color, lw=2)

        elif mode == "linear":
            slope = np.sum(x * y) / np.sum(x**2)

            x_line = np.linspace(80, 120, 200)
            y_line = 100 + slope * (x_line - 100)
            ax.plot(x_line, y_line, color=color, lw=2)

        else:
            raise ValueError(f"Unknown fit_mode: {mode}")

        ax_phase.axhline(
            slope,
            color=color,
            linestyle="--",
            lw=2,
            alpha=0.8,
        )

        return slope

    datasets = [
        ("Two filament model", amp_rel_2d, omega_rel_2d, records_2d, "#1f77b4"),
        ("Three dimensional model", amp_rel_3d, omega_rel_3d, records_3d, "#2ca02c"),
        ("Experiment", amp_rel_exp, omega_rel_exp, records_exp, "red"),
    ]

    for name, amp, omega, records, color in datasets:
        plot_density(ax_main, amp, omega, color)
        slope = fit_and_plot(ax_main, amp, omega, color, fit_mode)
        ax_main.plot([], [], color=color, label=f"{name} " +r"$\chi_\text{global}$"+f"={slope:1.2f}")

    ax_main.axhline(100, color="black", lw=1)
    ax_main.axvline(100, color="black", lw=1)

    ax_main.set_xlabel(r"Relative instantaneous amplitude $\alpha$")
    ax_main.set_ylabel(r"Relative instantaneous frequency $\omega/\omega_0$")

    ax_main.set_xlim(80, 120)
    ax_main.set_ylim(80, 120)

    ticks = [80, 90, 100, 110, 120]
    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)

    percent_formatter = FuncFormatter(lambda x, pos: f"{int(x)}%")
    ax_main.xaxis.set_major_formatter(percent_formatter)
    ax_main.yaxis.set_major_formatter(percent_formatter)

    ax_main.legend()

    # -------------------------------------------------
    # phase-dependent non-isochrony
    # -------------------------------------------------
    for name, amp, omega, records, color in datasets:
        phi_grid, mean_c, sem_c = average_phase_dependent_nonisochrony(
            records,
            nharm=nharm_phase,
            ngrid=300,
        )

        x = phi_grid 

        ax_phase.plot(x, mean_c, color=color, lw=2, label=name)
        ax_phase.fill_between(
            x,
            mean_c - sem_c,
            mean_c + sem_c,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax_phase.axhline(0, color="black", lw=1)

    ax_phase.set_xlabel(r"Phase $\phi$")
    ax_phase.set_ylabel(r"Phase-dependent non-isochrony $\chi(\phi)$")

    ax_phase.set_xlim(0, 2*np.pi)
    ax_phase.set_xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
    ax_phase.set_xticklabels([
        r"$0$",
        r"$\frac{\pi}{2}$",
        r"$\pi$",
        r"$\frac{3\pi}{2}$",
        r"$2\pi$",
    ])

    plt.tight_layout()
    plt.show()

def debug_simulation(dataset, exp_ids_N20):
    for exp_id in exp_ids_N20:
        realization = get_realization(dataset, exp_id)
        exp_path = get_exp_path(dataset, realization)

        tangent_item = None
        dt_item = None

        for item in realization.data_items.values():
            if item.data_name.key == "tangent_angle_series" :
                tangent_item = item
            if item.data_name.key == "dt_frame":
                dt_item = item
        
        if not (tangent_item and dt_item):
            print(f"⚠️ Missing data for {exp_id}")
            continue

        data = tangent_item.resolve(dataset, exp_path)
        dt_frame = dt_item.resolve(dataset, exp_path)

        from cilia.algorithms.phase_estimation import protophase_from_spatial_modes, estimate_phase_from_protophase
        data=data[2000:,:]
        res=protophase_from_spatial_modes(data)
        phi=res['protophase']
        phi=estimate_phase_from_protophase(phi)
        amp=res['rel_amplitude']
        # phi=phi[1000:]
        # amp=amp[1000:]
        phi = np.unwrap(phi)
        if phi[-1] - phi[0] < 0:
            phi = -phi
        
        omega = np.gradient(phi, dt_frame)

        # remove bad values
        mask = np.isfinite(omega) & np.isfinite(amp)
        amp=amp[mask]
        omega=omega[mask]
        amp_mean = np.mean(amp)
        omega_mean = np.mean(omega)
        amp_rel = (amp - amp_mean) / amp_mean
        omega_rel = (omega - omega_mean) / omega_mean

        # --- 2D histogram + contour ---
        H, xedges, yedges = np.histogram2d(
            amp_rel,
            omega_rel,
            bins=100,
            range=[[-0.5, 0.5], [-0.5, 0.5]],
            density=True
        )

        # smooth the histogram
        H = gaussian_filter(H, sigma=1.0)
        H /= np.max(H) # type: ignore

        # grid centers
        Xc = 0.5 * (xedges[:-1] + xedges[1:])
        Yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(Xc, Yc, indexing="ij")

        # contour levels (Gaussian-style)
        sigma_levels = [2, 1, 0.5, 0.25]
        levels = [np.exp(-0.5 * s**2) for s in sigma_levels]

        # plot
        plt.figure(figsize=(5, 5))

        # optional: background density
        plt.imshow(
            H.T,
            origin="lower",
            extent=[-0.5, 0.5, -0.5, 0.5], # type: ignore
            aspect="auto"
        )

        # contour lines
        plt.contour(
            X, Y, H,
            levels=levels,
            colors="black",
            linewidths=1.5
        )

        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)

        plt.xlabel(r"$A_\mathrm{rel}$")
        plt.ylabel(r"$\omega_\mathrm{rel}$")

        plt.grid(alpha=0.3)
        plt.show()


# ============================================================
# MAIN
# ============================================================
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from local_config import CILIA_FOLDER
except ModuleNotFoundError:
    cfg_path = PROJECT_ROOT / "local_config.py"
    if not cfg_path.exists():
        raise
    import importlib.util
    spec = importlib.util.spec_from_file_location("local_config", str(cfg_path))
    local_config = importlib.util.module_from_spec(spec) # type: ignore
    spec.loader.exec_module(local_config) # type: ignore
    CILIA_FOLDER = local_config.CILIA_FOLDER

def main():
    dataset_path = os.path.join(CILIA_FOLDER, "structured/sharma")
    dataset = SharmaDataset(dataset_path)
    out_csv = f"./scalar_observables.csv"
    df_exp=pd.read_csv(out_csv)
    exp_ids = df_exp.loc[
        (df_exp["KCl_mM"] == 0) & (df_exp["ATP_uM"] == 750),
        "experiment_id"
    ].tolist()
    omega_rel_exp, amp_rel_exp, records_exp = inspect_experiment(dataset, exp_ids) # type: ignore

    dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_3d_extraction')
    dataset = SimulationDataset(dataset_path)
    out_csv = f"./scalar_observables_cass_3d_extraction.csv"
    df_sim=pd.read_csv(out_csv)
    exp_ids_N20 = df_sim.loc[df_sim["Nmotor"] == 20, "experiment_id"].tolist()
    print(exp_ids_N20)
    omega_rel_3d, amp_rel_3d, records_3d = inspect_simulation(dataset, exp_ids_N20) # type: ignore

    dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_2d_extraction')
    dataset = SimulationDataset(dataset_path)
    out_csv = f"./scalar_observables_cass_2d_extraction.csv"
    df_sim=pd.read_csv(out_csv)
    exp_ids = df_sim.loc[df_sim["Nmotor"] == 85, "experiment_id"].tolist()
    print(exp_ids)
    # debug_simulation(dataset, exp_ids)
    omega_rel_2d, amp_rel_2d, records_2d = inspect_simulation(dataset, exp_ids) # type: ignore

    plot_amp_omega_phase_space(
        amp_rel_2d, omega_rel_2d, records_2d,
        amp_rel_3d, omega_rel_3d, records_3d,
        amp_rel_exp, omega_rel_exp, records_exp,
        nharm_phase=3,
    )
    

if __name__ == "__main__":
    main()