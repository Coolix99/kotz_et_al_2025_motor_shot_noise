import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

from utils_rdf.config import (
    OUTPUT_DATA_FLUCTUATIONS_EXTRACTION,
    OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER,OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT
)
from utils_rdf.config import DT_SIMULATION


def load_npz_dict(path):
    """Safely load a .npz file as a dict or return None if missing."""
    try:
        data = np.load(path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
        return None

def parse_folder_name(folder_name):
    

    parts = folder_name.split("_")
    if len(parts) >= 4 and parts[0] == "Nmotor" and parts[2] == "mu":
        Nmotor = int(parts[1])
        mu_a = float(parts[3])
    else:
        # fallback for nonstandard patterns
        Nmotor = int(next(p for i, p in enumerate(parts) if p.isdigit()))
        mu_a = float(parts[-1])


    return Nmotor, mu_a

def find_group_dirs(root_dir, kind="auto"):
    """
    Find all group result folders within root_dir.

    Supports both simulation and experimental naming conventions:
      - Simulation:   Nmotor_<N>_mu_<μ>
      - Experimental: genotype_<X>_ATP_<Y>_KCL_<Z>

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing group subfolders.
    kind : {"auto", "simulation", "experiment"}, default="auto"
        If "simulation", only return Nmotor_*_mu_* folders.
        If "experiment", only return genotype_*_ATP_*_KCL_* folders.
        If "auto", return both.

    Returns
    -------
    list[Path]
        Sorted list of matching group directories.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    sim_pattern = "Nmotor_*_mu_*"
    exp_pattern = "genotype_*_ATP_*_KCL_*"

    if kind == "simulation":
        paths = sorted(p for p in root_dir.glob(sim_pattern) if p.is_dir())
    elif kind == "experiment":
        paths = sorted(p for p in root_dir.glob(exp_pattern) if p.is_dir())
    else:  # "auto"
        # Collect both kinds (simulation + experiment)
        paths = sorted(
            p for pat in (sim_pattern, exp_pattern)
            for p in root_dir.glob(pat)
            if p.is_dir()
        )

    return paths

def load_all_results(root_dir):
    """Load all .npz results per (Nmotor, mu_a)."""
    all_results = []
    for folder in tqdm(find_group_dirs(root_dir), desc=f"Loading from {root_dir.name}"):
        # parse parameters
        try:
            parts = folder.name.split("_")
            Nmotor = int(parts[1])
            mu_a = float(parts[3])
        except Exception:
            Nmotor, mu_a = np.nan, np.nan

        # collect all files
        entry = {"Nmotor": Nmotor, "mu_a": mu_a, "path": folder}
        for npz_file in folder.glob("*.npz"):
            entry[npz_file.stem] = load_npz_dict(npz_file)
        all_results.append(entry)

    return pd.DataFrame(all_results)

def plot_phase_defects(root_dir, dt, figsize=(8, 6), cutoff=7, lambda_assumed=2):
    """
    Plot detected topological defects (phase slips/singularities) 
    overlaid on the phase field φ(s,t).

    The background shows the aligned phase field φ(s,t) in HSV colormap.
    +1 and −1 singularities are overlaid as black markers (X and O).
    
    Parameters
    ----------
    root_dir : Path or str
        Directory containing simulation folders.
    dt : float
        Time step size for scaling the time axis.
    figsize : tuple
        Figure size.
    cutoff : float
        Minimum distance below which opposite-charge pairs are removed (for clarity).
    """
    for folder in tqdm(find_group_dirs(root_dir), desc="Phase defect plots"):
        Nm, mu_a = parse_folder_name(folder.name)
        if mu_a<400:
            continue
        # if Nm!=85 or mu_a != 850:
        #     continue
        path_def = folder / "phase_defects.npz"
        if not path_def.exists():
            continue

        res = load_npz_dict(path_def)
        if not res or "all_results" not in res:
            continue

        all_results = res["all_results"]
        if isinstance(all_results, np.ndarray):
            all_results = list(all_results)
        if not all_results:
            continue

        # Load aligned phase fields (saved by phase_defects)
        phi_list = None
        if "phi_aligned_list" in res:
            try:
                phi_list = list(res["phi_aligned_list"])
            except Exception as e:
                print(f"⚠️ Could not parse phi_aligned_list in {folder.name}: {e}")
                phi_list = None
        else:
            continue

        for i, r in enumerate(all_results):
            # if i != 16:
            #     continue
            if not isinstance(r, dict):
                continue

            charges = np.asarray(r.get("charges", []))
            pos = np.asarray(r.get("pos_coords", []))
            neg = np.asarray(r.get("neg_coords", []))
            if charges.size == 0:
                continue

            phi_field = None
            if phi_list is not None and i < len(phi_list):
                phi_field = phi_list[i]
            if phi_field is None:
                continue

            # --- Remove close +/− pairs (same as in phase_defects)
            if len(pos) > 0 and len(neg) > 0:
                pos_keep = np.ones(len(pos), dtype=bool)
                neg_keep = np.ones(len(neg), dtype=bool)
                dists = cdist(pos, neg)
                while True:
                    i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
                    if dists[i_min, j_min] >= cutoff:
                        break
                    pos_keep[i_min] = False
                    neg_keep[j_min] = False
                    dists[i_min, :] = np.inf
                    dists[:, j_min] = np.inf
                pos = pos[pos_keep]
                neg = neg[neg_keep]


            # --- Prepare axes
            n_t, n_s = phi_field.shape
            s_rel = np.linspace(0, 1, n_s)
            t_axis = np.arange(n_t) * dt

            # --- Plot φ(s,t) with overlaid singularities ---
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                np.angle(np.exp(1j * (phi_field.T - (np.linspace(0,1,n_s)*2*np.pi/lambda_assumed)[:,np.newaxis]))) % (2 * np.pi),
                cmap="hsv", origin="lower", aspect="auto",
                extent=[t_axis[0], t_axis[-1], s_rel[0], s_rel[-1]],
                interpolation="none"
            )
            cbar = plt.colorbar(im, ax=ax, label=r"Phase $\phi$ [rad]")
            cbar.ax.tick_params(labelsize=12)

            # +1 singularities
            if pos.size > 0:
                ax.scatter(
                    t_axis[pos[:, 0]], s_rel[pos[:, 1]],
                    c="black", marker="x", label="+1 singularity", s=70, lw=2
                )
            # −1 singularities
            if neg.size > 0:
                ax.scatter(
                    t_axis[neg[:, 0]], s_rel[neg[:, 1]],
                    facecolors="none", edgecolors="black",
                    marker="o", label="−1 singularity", s=90, lw=2
                )

            # --- Styling ---
            #ax.set_xlim(1.375,1.55) #exp
            # ax.set_xlim(26.15,26.325) #exp
            ax.set_xlabel("Time t [s]", fontsize=14)
            ax.set_ylabel("s (relative position)", fontsize=14)
            ax.set_title(f"{folder.name} – set {i}", fontsize=15, pad=10)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.legend(loc="upper right", fontsize=12, frameon=True)
            plt.tight_layout()
            plt.show()

def run_all_plots(root_dir, dt):
    print(f"Scanning {root_dir}...")
    root_dir = Path(root_dir)

    plot_phase_defects(root_dir, dt=dt)


# ----------------------------------------------------------------------
# --- main
# ----------------------------------------------------------------------

def main():
    run_all_plots(OUTPUT_DATA_FLUCTUATIONS_EXTRACTION, DT_SIMULATION)
    run_all_plots(OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER, DT_SIMULATION/0.45)
    run_all_plots(OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT, 0.001)


if __name__ == "__main__":
    main()
