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
    all_defects = []
    Tges=0
    for exp_id in exp_ids:
        realization = get_realization(dataset, exp_id)
        exp_path = get_exp_path(dataset, realization)

        defect_item = None
        dt_item = None

        for item in realization.data_items.values():

            if item.data_name.key == "phase_defects_segments" :
                defect_item = item
            if item.data_name.key == "dt_frame":
                dt_item = item
        
        if not (defect_item and dt_item):
            print(f"⚠️ Missing data for {exp_id}")
            continue

        data_list = defect_item.resolve(dataset, exp_path)
        dt_frame = dt_item.resolve(dataset, exp_path)

        for data in data_list:
            clean_defects=data['clean']
            Teff=data['effective_T']
            if np.isfinite(Teff):
                all_defects.append(clean_defects)
                Tges += Teff


    
    all_defects=np.concatenate(all_defects)
    
    return all_defects, Tges

def inspect_simulation(dataset, exp_ids_N20):
    all_defects = []
    Tges=0

    for exp_id in exp_ids_N20:
        realization = get_realization(dataset, exp_id)
        exp_path = get_exp_path(dataset, realization)

        defect_item = None
        dt_item = None

        for item in realization.data_items.values():
            if item.data_name.key == "phase_defects_series":
                defect_item = item
            if item.data_name.key == "dt_frame":
                dt_item = item

        if not (defect_item and dt_item):
            print(f"⚠️ Missing data for {exp_id}")
            continue

        data = defect_item.resolve(dataset, exp_path)
        dt_frame = dt_item.resolve(dataset, exp_path)

        clean_defects=data['clean']
        Teff=data['effective_T']
        all_defects.append(clean_defects)
        Tges += Teff


    all_defects=np.concatenate(all_defects)
    
    return all_defects, Tges

def plot_amp_omega_phase_space(
    amp_rel_2d, omega_rel_2d,
    amp_rel_3d, omega_rel_3d,
    amp_rel_exp, omega_rel_exp,
    bins=100,
    sigma_smooth=1.0,
):
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.6, 0.6])

    ax_main = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_omega = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------
    # helper: density + contours
    # -------------------------------------------------
    def plot_density(ax, x, y, color, label):
        # histogram
        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=bins,
            range=[[-0.5, 0.5], [-0.5, 0.5]],
            density=True
        )

        # smooth
        H = gaussian_filter(H, sigma=sigma_smooth)
        H /= np.max(H) # type: ignore

        # grid centers
        Xc = 0.5 * (xedges[:-1] + xedges[1:])
        Yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(Xc, Yc, indexing="ij")

        # sigma levels (Gaussian interpretation)
        sigma_levels = [ 2, 1, 0.5, 0.25]
        levels = [np.exp(-0.5 * s**2) for s in sigma_levels]

        ax.contour(
            X, Y, H,
            levels=levels,
            colors=color,
            linewidths=2,
            alpha=0.9,
        )

        # fake line for legend
        ax.plot([], [], color=color, label=label)

    # -------------------------------------------------
    # main density plots
    # -------------------------------------------------
    plot_density(ax_main, amp_rel_2d, omega_rel_2d, "#1f77b4", "2D")
    plot_density(ax_main, amp_rel_3d, omega_rel_3d, "#2ca02c", "3D")
    plot_density(ax_main, amp_rel_exp, omega_rel_exp, "red", "experiment")
    # -------------------------------------------------
    # linear fit (no offset!)
    # ω_rel = c * A_rel
    # -------------------------------------------------
    def fit_and_plot(ax, x, y, color, label):
        x = np.asarray(x)
        y = np.asarray(y)

        # stack data
        X = np.vstack([x, y]).T

        # subtract mean (important!)
        mean = np.mean(X, axis=0)
        Xc = X - mean

        # PCA via SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        direction = Vt[0]   # principal direction (unit vector)

        dx, dy = direction

        # slope (handle vertical case)
        if np.abs(dx) > 1e-12:
            slope = dy / dx
        else:
            slope = np.inf

        # line for plotting
        t = np.linspace(-0.5, 0.5, 200)
        line = mean + t[:, None] * direction

        ax.plot(line[:, 0], line[:, 1], color=color, lw=2)

        print(f"{label}: slope (PCA) = {slope:.4f}")

    fit_and_plot(ax_main, amp_rel_2d, omega_rel_2d, "#1f77b4", "2D")
    fit_and_plot(ax_main, amp_rel_3d, omega_rel_3d, "#2ca02c", "3D")
    fit_and_plot(ax_main, amp_rel_exp, omega_rel_exp, "red", "experiment")

    # -------------------------------------------------
    # formatting main
    # -------------------------------------------------
    ax_main.axhline(0, color="black", lw=1)
    ax_main.axvline(0, color="black", lw=1)

    ax_main.set_xlabel(r"$(A - \bar A)/\bar A$")
    ax_main.set_ylabel(r"$(\omega - \bar \omega)/\bar \omega$")

    ax_main.set_xlim(-0.5, 0.5)
    ax_main.set_ylim(-0.5, 0.5)

    ax_main.grid(alpha=0.3)
    ax_main.legend()

    # -------------------------------------------------
    # marginal histograms (line only)
    # -------------------------------------------------
    def plot_hist(ax, data, color, label):
        hist, edges = np.histogram(data, bins=100, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist = gaussian_filter(hist, sigma=1.0)

        ax.plot(centers, hist, color=color, lw=2, label=label)

    # amplitude
    plot_hist(ax_amp, amp_rel_2d, "#1f77b4", "2D")
    plot_hist(ax_amp, amp_rel_3d, "#2ca02c", "3D")
    plot_hist(ax_amp, amp_rel_exp, "red", "exp")

    ax_amp.set_title("rel amplitude")
    ax_amp.set_xlabel(r"$A_\mathrm{rel}$")
    ax_amp.grid(alpha=0.3)
    ax_amp.legend()

    # omega
    plot_hist(ax_omega, omega_rel_2d, "#1f77b4", "2D")
    plot_hist(ax_omega, omega_rel_3d, "#2ca02c", "3D")
    plot_hist(ax_omega, omega_rel_exp, "red", "exp")

    ax_omega.set_title(r"rel frequency")
    ax_omega.set_xlabel(r"$\omega_\mathrm{rel}$")
    ax_omega.grid(alpha=0.3)
    ax_omega.legend()

    plt.tight_layout()
    plt.show()


def plot_defect_density_by_charge(defects, T_total, nbins=30):
    """
    defects: array of shape (N, 3) -> [time_idx, space_idx, charge]
    T_total: total observation time (for rate normalization)
    """

    LABEL_SIZE = 14
    TICK_SIZE = 12
    TITLE_SIZE = 14

    defects = np.asarray(defects)

    space = defects[:, 1]
    charge = defects[:, 2]

    # split charges
    space_pos = space[charge > 0]
    space_neg = space[charge < 0]

    # common bins
    smin, smax = np.min(space), np.max(space)
    bins = np.linspace(smin, smax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # convert to relative position
    centers_rel = centers / 50.0
    bin_width_rel = bin_width / 50.0

    # histogram counts
    hist_pos, _ = np.histogram(space_pos, bins=bins)
    hist_neg, _ = np.histogram(space_neg, bins=bins)

    # convert to density (rate per space per time)
    density_pos = hist_pos / (T_total * bin_width)
    density_neg = hist_neg / (T_total * bin_width)

    # avoid log(0)
    density_pos = np.where(density_pos > 0, density_pos, np.nan)
    density_neg = np.where(density_neg > 0, density_neg, np.nan)

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    axes[0].bar(centers_rel, density_pos, width=bin_width_rel, color="red", alpha=0.8)
    axes[0].set_title("+1 defects", fontsize=TITLE_SIZE)

    axes[1].bar(centers_rel, density_neg, width=bin_width_rel, color="red", alpha=0.8)
    axes[1].set_title("−1 defects", fontsize=TITLE_SIZE)

    for ax in axes:
        # vertical reference lines
        ax.axvline(0.15, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(0.75, color="black", linestyle="--", linewidth=1.5)

        ax.set_xlabel(r"relative position $s/L$", fontsize=LABEL_SIZE)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

        ax.tick_params(axis="both", labelsize=TICK_SIZE)

    axes[0].set_ylabel(r"defect rate density [1/s]", fontsize=LABEL_SIZE)

    plt.tight_layout()
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
    # print(df_exp.head())
    # print(df_exp.columns)
    exp_ids = df_exp.loc[
        (df_exp["KCl_mM"] == 0) & (df_exp["ATP_uM"] == 750),
        "experiment_id"
    ].tolist()
    
    defects_exp, Texp=inspect_experiment(dataset, exp_ids) # type: ignore

    print(defects_exp, Texp)

    plot_defect_density_by_charge(defects_exp, Texp)
    return

    dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_3d_extraction')
    dataset = SimulationDataset(dataset_path)
    out_csv = f"./scalar_observables_cass_3d_extraction.csv"
    df_sim=pd.read_csv(out_csv)
    exp_ids_N20 = df_sim.loc[df_sim["Nmotor"] == 20, "experiment_id"].tolist()
    print(exp_ids_N20)
    defects_3d, T3d=inspect_simulation(dataset, exp_ids_N20) # type: ignore

    dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_2d_extraction')
    dataset = SimulationDataset(dataset_path)
    out_csv = f"./scalar_observables_cass_2d_extraction.csv"
    df_sim=pd.read_csv(out_csv)
    exp_ids = df_sim.loc[df_sim["Nmotor"] == 85, "experiment_id"].tolist()
    print(exp_ids)
    defects_2d, T2d=inspect_simulation(dataset, exp_ids) # type: ignore

   

if __name__ == "__main__":
    main()