import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
from scipy.spatial.distance import cdist
import matplotlib.animation as animation

from utils_rdf.config import (
    OUTPUT_DATA_NEW_PARAMETER_DIR,
    OUTPUT_DATA_SHARME_DIR,DT_SIMULATION,  INITIAL_TRANSIENT_TIME_SIMULATION,
)


def _read_experiment_csv(csv_path):
    """Read all_sharma_results.csv and keep only the columns we need."""
    df = pd.read_csv(csv_path)
    needed = ["File", "ATP", "genotype", "KCL"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Experimental CSV missing columns: {missing}")
    return df[needed].copy()

def _list_experiment_folders(root_dir):
    """
    Return a DataFrame of experimental folders with:
      - path (Path)
      - folder (str)
      - file_tag (str)  # inferred as folder.name without the trailing "_{i_segment}"
    """
    rows = []
    for p in sorted(Path(root_dir).iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        # robust: split ONCE from right; last chunk is segment index
        if "_" not in name:
            # if no underscore, treat entire name as tag (rare)
            file_tag = name
        else:
            file_tag = name.rsplit("_", 1)[0]
        rows.append({"path": p, "folder": name, "file_tag": file_tag})
    return pd.DataFrame(rows)

def _build_experiment_groups(exp_root, csv_path):
    """
    Build groups { (genotype, ATP, KCL) : [Path, ...] } by joining folders with the CSV on 'File'.
    Every folder '{file_tag}_{i_segment}' inherits metadata of the CSV row where File==file_tag.
    """
    folders_df = _list_experiment_folders(exp_root)
    if folders_df.empty:
        print(f"⚠️ No experimental folders found in {exp_root}")
        return {}

    csv_df = _read_experiment_csv(csv_path)

    # join: many folders per single csv 'File'
    merged = folders_df.merge(csv_df, left_on="file_tag", right_on="File", how="left", validate="many_to_one")
    # drop missing metadata rows
    bad = merged["genotype"].isna() | merged["ATP"].isna() | merged["KCL"].isna()
    if bad.any():
        missing_rows = merged[bad]
        for _, r in missing_rows.iterrows():
            print(f"⚠️ Missing metadata in CSV for experimental folder '{r['folder']}' (file_tag='{r['file_tag']}'). Skipping.")
        merged = merged[~bad].copy()

    groups = {}
    for _, r in merged.iterrows():
        key = (r["genotype"], r["ATP"], r["KCL"])
        groups.setdefault(key, []).append(r["path"])
    return groups

def plot_singularity_interactive(
    t_sing, si, charge, t_axis, s_rel, phi_aligned, centered,
    Delta_t, lambda_assumed, n_times, fontsize_axlabel,
    tick_fontsize=14, colorbar_ticksize=14,
    return_fig=False, specific_t_ms=None
):

    # Convert t-axis to milliseconds, shift so t_sing → 0
    t_axis_ms = (t_axis - t_sing) * 1000.0
    t_sing_ms = 0.0
    Delta_t_ms = Delta_t * 1000.0

    t_min_ms = -Delta_t_ms
    t_max_ms = +Delta_t_ms

    t_mask = (t_axis_ms >= t_min_ms) & (t_axis_ms <= t_max_ms)
    phi_window = phi_aligned[t_mask]
    t_window_ms = t_axis_ms[t_mask]

    # idx_min, idx_max = np.searchsorted(t_axis_ms, [t_min_ms, t_max_ms])
    y_min = -1.7
    y_max = 1.7

    s_axis = np.linspace(0, 1, centered.shape[1])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(22, 5),
        gridspec_kw={'width_ratios': [1.5, 1.2, 1.2, 1.2]}
    )
    fig.suptitle(
        f"Singularity at s={s_rel[si]:.2f}, charge={charge:+d}",
        fontsize=14
    )

  
    phase_shift = (
        np.linspace(0, 1, phi_window.shape[1]) * 2 * np.pi / lambda_assumed
    )[:, None]

    im = ax1.imshow(
        np.angle(np.exp(1j * (phi_window.T - phase_shift))) % (2 * np.pi),
        cmap="hsv",
        origin="lower",
        aspect="auto",
        extent=[t_window_ms[0], t_window_ms[-1], s_rel[0], s_rel[-1]],
        interpolation="none"
    )

    ax1.axvline(x=t_sing_ms, color='w', linestyle='--', lw=1.5)
    ax1.set_xlabel("t [ms]", fontsize=fontsize_axlabel)
    ax1.set_ylabel("s", fontsize=fontsize_axlabel)

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label(r"Phase $\phi$", fontsize=fontsize_axlabel)
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])

    # mark defect
    if charge == +1:
        ax1.scatter(t_sing_ms, s_rel[si], c="black", marker="x", s=70, lw=2)
    else:
        ax1.scatter(t_sing_ms, s_rel[si], facecolors="none", edgecolors="black",
                    marker="o", s=90, lw=2)

   
    idx_nearest = np.argmin(np.abs(t_axis_ms - t_sing_ms))
    line_mid, = ax2.plot(s_axis, centered[idx_nearest], 'k-', lw=2)

    ax2.set_xlabel("s", fontsize=fontsize_axlabel)
    ax2.set_ylabel(r"$\gamma(s)$", fontsize=fontsize_axlabel)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("t = 0 ms")

    slider_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='t [ms]',
        valmin=t_min_ms,
        valmax=t_max_ms,
        valinit=t_sing_ms,
        valstep=(t_window_ms[1] - t_window_ms[0])
    )

   
    t_snap_min_ms = -Delta_t_ms / 3
    t_snap_max_ms = +Delta_t_ms / 3
    t_samples_ms = np.linspace(t_snap_min_ms, t_snap_max_ms, n_times)

    cmap_map = cm.get_cmap('turbo')
    norm = plt.Normalize(t_snap_min_ms, t_snap_max_ms)

    for t_sel in t_samples_ms:
        idx = np.argmin(np.abs(t_axis_ms - t_sel))
        ax3.plot(s_axis, centered[idx], color=cmap_map(norm(t_sel)), lw=1.5)

    sm = cm.ScalarMappable(cmap=cmap_map, norm=norm)
    cbar3 = fig.colorbar(sm, ax=ax3)
    cbar3.set_label("t [ms]", fontsize=fontsize_axlabel)
    cbar3.ax.axhline(0, color='k', lw=2)

    ax3.set_xlabel("s", fontsize=fontsize_axlabel)
    ax3.set_ylabel(r"$\gamma(s)$", fontsize=fontsize_axlabel)
    ax3.set_ylim(y_min, y_max)


    def compute_xy(gamma):
        ds = 1.0 / len(gamma)
        x = np.cumsum(np.cos(gamma)) * ds
        y = np.cumsum(np.sin(gamma)) * ds
        return x, y

    ax4.set_title("Real-space shape", fontsize=fontsize_axlabel)
    ax4.set_xlabel("x [µm]", fontsize=fontsize_axlabel)
    ax4.set_ylabel("y [µm]", fontsize=fontsize_axlabel)
    ax4.set_aspect("equal", adjustable="box")

    # ticks
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax3.tick_params(axis='both', labelsize=tick_fontsize)
    ax4.tick_params(axis='both', labelsize=tick_fontsize)

    # colorbar
    cbar.ax.tick_params(labelsize=colorbar_ticksize)
    cbar3.ax.tick_params(labelsize=colorbar_ticksize)


    def update_right_panels(t_sel_ms):
        ax4.cla()
        ax4.set_aspect("equal", adjustable="box")

        for t_ in t_samples_ms:
            idx = np.argmin(np.abs(t_axis_ms - t_))
            x, y = compute_xy(centered[idx])
            ax4.plot(x, y, color=cmap_map(norm(t_)), lw=1.5)

        ax4.set_xlim(-0.1, 1.1)
        ax4.set_ylim(-0.75, 0.75)
        ax4.set_xlabel("x [µm]", fontsize=fontsize_axlabel)
        ax4.set_ylabel("y [µm]", fontsize=fontsize_axlabel)
        ax4.set_title("Real-space shape", fontsize=fontsize_axlabel)

    update_right_panels(t_sing_ms)

    # slider callback
    def update(val):
        t_sel_ms = slider.val
        idx = np.argmin(np.abs(t_axis_ms - t_sel_ms))
        line_mid.set_ydata(centered[idx])
        ax2.set_title(f"t = {t_sel_ms:.1f} ms")
        ax1.lines[0].set_xdata([t_sel_ms, t_sel_ms])
        update_right_panels(t_sel_ms)
        fig.canvas.draw_idle()

    # --- VIDEO MODE: produce a single frame ---
    if specific_t_ms is not None:
        t_sel_ms = specific_t_ms
        idx = np.argmin(np.abs(t_axis_ms - t_sel_ms))
        line_mid.set_ydata(centered[idx])
        ax2.set_title(f"t = {t_sel_ms:.1f} ms")
        ax1.lines[0].set_xdata([t_sel_ms, t_sel_ms])
        update_right_panels(t_sel_ms)

        if return_fig:
            return fig
        else:
            plt.show()
            return

    slider.on_changed(update)

    plt.subplots_adjust(bottom=0.15, wspace=0.35)
    plt.show()

def render_singularity_frame(
    fig, axes, artists,
    t_sel_ms, t_axis_ms, s_axis, centered,
    s_rel, si, charge,
    cmap_map, norm,
    phi_window, phase_shift,
    fontsize_axlabel
):
    ax1, ax2, ax3, ax4 = axes
    im1, vline1, line_mid = artists

    # --- Panel 1: move dashed line only
    vline1.set_xdata([t_sel_ms, t_sel_ms])

    # --- Panel 2: update γ(s,t)
    idx = np.argmin(np.abs(t_axis_ms - t_sel_ms))
    line_mid.set_ydata(centered[idx])
    ax2.set_title(f"t = {t_sel_ms:.1f} ms")

    # --- Panel 4: Fully dynamic real-space shape
    ax4.cla()
    ax4.set_aspect("equal", adjustable="box")

    for t_ in np.linspace(norm.vmin, norm.vmax, 20):
        idx = np.argmin(np.abs(t_axis_ms - t_))
        ds = 1.0 / centered.shape[1]
        x = np.cumsum(np.cos(centered[idx])) * ds
        y = np.cumsum(np.sin(centered[idx])) * ds
        ax4.plot(x, y, color=cmap_map(norm(t_)), lw=1.5)

    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.75, 0.75)
    ax4.set_xlabel("x [µm]", fontsize=fontsize_axlabel)
    ax4.set_ylabel("y [µm]", fontsize=fontsize_axlabel)
    ax4.set_title("Real-space shape", fontsize=fontsize_axlabel)

def init_singularity_video_figure(
    t_axis_ms, t_window_ms, s_rel, si, charge,
    phi_window, phase_shift,
    s_axis, centered,
    Delta_t_ms, n_times,
    lambda_assumed, fontsize_axlabel
):

    cmap_map = matplotlib.colormaps.get_cmap("turbo")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(22, 5),
        gridspec_kw={'width_ratios': [1.5, 1.2, 1.2, 1.2]}
    )

    fig.suptitle(
        f"Singularity at s={s_rel[si]:.2f}, charge={charge:+d}",
        fontsize=14
    )

    # Static phase panel
    im1 = render_static_phase_panel(
        ax1, phi_window, t_window_ms, s_rel,
        phase_shift, t_sing_ms=0, fontsize_axlabel=fontsize_axlabel
    )
    vline1 = ax1.axvline(x=0, color='w', linestyle='--', lw=1.5)

    #  Dynamic γ(s,t)
    idx0 = np.argmin(np.abs(t_axis_ms - 0))
    line_mid, = ax2.plot(s_axis, centered[idx0], 'k-', lw=2)
    ax2.set_xlabel("s", fontsize=fontsize_axlabel)
    ax2.set_ylabel(r"$\gamma(s)$", fontsize=fontsize_axlabel)
    ax2.set_ylim(-1.7, 1.7)
    ax2.set_title("t = 0 ms")

    #  Static γ cluster
    norm = render_static_gamma_panel(
        ax3, t_axis_ms, centered, s_axis,
        0, Delta_t_ms, n_times, cmap_map, fontsize_axlabel
    )

    #  Real-space shape — dynamic; blank for now

    return fig, (ax1, ax2, ax3, ax4), (im1, vline1, line_mid), cmap_map, norm

def render_static_gamma_panel(ax3, t_axis_ms, centered, s_axis,
                               t_sing_ms, Delta_t_ms, n_times, cmap_map, fontsize_axlabel):

    t_snap_min = -Delta_t_ms / 3
    t_snap_max = +Delta_t_ms / 3
    t_samples = np.linspace(t_snap_min, t_snap_max, n_times)
    norm = plt.Normalize(t_snap_min, t_snap_max)

    for t_sel in t_samples:
        idx = np.argmin(np.abs(t_axis_ms - t_sel))
        ax3.plot(s_axis, centered[idx], color=cmap_map(norm(t_sel)), lw=1.5)

    sm = cm.ScalarMappable(cmap=cmap_map, norm=norm)
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label("t [ms]", fontsize=fontsize_axlabel)
    cbar.ax.axhline(0, color='k', lw=2)

    ax3.set_xlabel("s", fontsize=fontsize_axlabel)
    ax3.set_ylabel(r"$\gamma(s)$", fontsize=fontsize_axlabel)
    ax3.set_ylim(np.min(centered), np.max(centered))
    ax3.set_title("γ(s,t)", fontsize=fontsize_axlabel)

    return norm

def render_static_phase_panel(ax1, phi_window, t_window_ms, s_rel,
                              phase_shift, t_sing_ms, fontsize_axlabel):

    phase = np.angle(np.exp(1j * (phi_window.T - phase_shift))) % (2*np.pi)

    im = ax1.imshow(
        phase,
        cmap="hsv",
        origin="lower",
        aspect="auto",
        extent=[t_window_ms[0], t_window_ms[-1], s_rel[0], s_rel[-1]],
        interpolation="none"
    )

    ax1.axvline(x=t_sing_ms, color='w', linestyle='--', lw=1.5)
    ax1.set_xlabel("t [ms]", fontsize=fontsize_axlabel)
    ax1.set_ylabel("s", fontsize=fontsize_axlabel)

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(r"Phase $\phi$", fontsize=fontsize_axlabel)
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])

    return im



def export_singularity_video(
    t_sing, si, charge, t_axis, s_rel, phi_aligned, centered,
    Delta_t, lambda_assumed, n_times, fontsize_axlabel,
    video_filename="singularity.mp4",
    fps=1
):
    t_axis_ms = (t_axis - t_sing) * 1000.0
    Delta_t_ms = Delta_t * 1000.0

    t_min_ms = -Delta_t_ms
    t_max_ms = +Delta_t_ms
    t_values = np.arange(t_min_ms, t_max_ms+1, 1)

    mask = (t_axis_ms >= t_min_ms) & (t_axis_ms <= t_max_ms)
    phi_window = phi_aligned[mask]
    t_window_ms = t_axis_ms[mask]

    phase_shift = (
        np.linspace(0, 1, phi_window.shape[1]) * 2*np.pi / lambda_assumed
    )[:, None]

    s_axis = np.linspace(0, 1, centered.shape[1])

    fig, axes, artists, cmap_map, norm = init_singularity_video_figure(
        t_axis_ms, t_window_ms, s_rel, si, charge,
        phi_window, phase_shift, s_axis, centered,
        Delta_t_ms, n_times, lambda_assumed, fontsize_axlabel
    )

    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="Python"))

    with writer.saving(fig, video_filename, dpi=150):
        for t_ms in t_values:
            render_singularity_frame(
                fig, axes, artists,
                t_ms, t_axis_ms, s_axis, centered,
                s_rel, si, charge,
                cmap_map, norm,
                phi_window, phase_shift,
                fontsize_axlabel
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Video saved to {video_filename}")

def plot_kymo(phi_field, centered, dt,Ns, lambda_assumed=2.0, cutoff=3, ):
    """
    Extended 4-panel kymograph plot:
      1. Phase kymograph with singularities
      2. Tangent angle γ(s,t) with interactive slider
      3. Color-coded γ(s,t) curves
      4. Real-space (x,y) reconstruction from γ(s,t)
    """

    # Ensure 2D (time × space)
    phi_field = np.asarray(phi_field)
    if phi_field.ndim != 2:
        raise ValueError(f"Expected 2D (time × space), got {phi_field.shape}")
    nx, ny = phi_field.shape

    # --- Wrap phase into [-π, π)
    phi = np.angle(np.exp(1j * phi_field))

    
    mid_idx = ny // 2
    phi_ref = phi[:, mid_idx]
    rel_phase = np.angle(np.exp(1j * (phi - phi_ref[:, None])))
    mean_diff = np.angle(np.nanmean(np.exp(1j * rel_phase), axis=0))
    phi_aligned = np.angle(np.exp(1j * (phi - mean_diff[None, :])))

    dphi_right = np.angle(np.exp(1j * (phi_aligned[1:, :-1] - phi_aligned[:-1, :-1])))
    dphi_up    = np.angle(np.exp(1j * (phi_aligned[1:, 1:]  - phi_aligned[1:, :-1])))
    dphi_left  = np.angle(np.exp(1j * (phi_aligned[:-1, 1:] - phi_aligned[1:, 1:])))
    dphi_down  = np.angle(np.exp(1j * (phi_aligned[:-1, :-1] - phi_aligned[:-1, 1:])))
    winding = dphi_right + dphi_up + dphi_left + dphi_down
    charges = np.rint(winding / (2 * np.pi)).astype(int)

    ii, jj = np.meshgrid(np.arange(charges.shape[0]),
                         np.arange(charges.shape[1]),
                         indexing="ij")
    pos_coords = np.column_stack([ii[charges == 1], jj[charges == 1]])
    neg_coords = np.column_stack([ii[charges == -1], jj[charges == -1]])

    target_shape = phi_aligned.shape
    diff_rows = centered.shape[0] - target_shape[0]
    centered = centered[diff_rows // 2 : centered.shape[0] - (diff_rows - diff_rows // 2)]
    centered = centered - centered[:, [0]]

    # Remove close +/− pairs
    if len(pos_coords) > 0 and len(neg_coords) > 0:
        pos_keep = np.ones(len(pos_coords), dtype=bool)
        neg_keep = np.ones(len(neg_coords), dtype=bool)
        dists = cdist(pos_coords, neg_coords)
        while True:
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[i, j] >= cutoff:
                break
            pos_keep[i] = False
            neg_keep[j] = False
            dists[i, :] = np.inf
            dists[:, j] = np.inf
        pos_coords = pos_coords[pos_keep]
        neg_coords = neg_coords[neg_keep]
    s_min = 0.15
    s_max = 0.70

    if len(pos_coords) > 0:
        s_norm = pos_coords[:, 1] / Ns
        keep = (s_norm >= s_min) & (s_norm <= s_max)
        pos_coords = pos_coords[keep]

    if len(neg_coords) > 0:
        s_norm = neg_coords[:, 1] / Ns
        keep = (s_norm >= s_min) & (s_norm <= s_max)
        neg_coords = neg_coords[keep]

    n_t, n_s = phi_aligned.shape
    s_rel = np.linspace(0, 1, n_s)
    t_axis = np.arange(n_t) * dt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.angle(np.exp(1j * (phi_aligned.T - (np.linspace(0, 1, n_s) * 2 * np.pi / lambda_assumed)[:, None]))) % (2 * np.pi),
        cmap="hsv", origin="lower", aspect="auto",
        extent=[t_axis[0], t_axis[-1], s_rel[0], s_rel[-1]],
        interpolation="none"
    )
    cbar = plt.colorbar(im, ax=ax, label=r"Phase $\phi$ [rad]")
    if pos_coords.size > 0:
        ax.scatter(t_axis[pos_coords[:, 0]], s_rel[pos_coords[:, 1]],
                   c="black", marker="x", label="+1", s=70, lw=2)
    if neg_coords.size > 0:
        ax.scatter(t_axis[neg_coords[:, 0]], s_rel[neg_coords[:, 1]],
                   facecolors="none", edgecolors="black", marker="o", label="−1", s=90, lw=2)
    ax.set_xlabel("Time t [s]", fontsize=14)
    ax.set_ylabel("s (relative position)", fontsize=14)
    ax.legend(loc="upper right", fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

    fontsize_axlabel = 14
    Delta_t = 0.05
    n_times = 20

    if pos_coords.size + neg_coords.size == 0:
        return

    all_coords = np.vstack([
        np.column_stack([pos_coords, np.full(len(pos_coords), +1)]),
        np.column_stack([neg_coords, np.full(len(neg_coords), -1)])
    ])

    for ti, si, charge in all_coords:
        t_sing = t_axis[ti]
        plot_singularity_interactive(
            t_sing=t_sing,
            si=si,
            charge=charge,
            t_axis=t_axis,
            s_rel=s_rel,
            phi_aligned=phi_aligned,
            centered=centered,
            Delta_t=Delta_t,
            lambda_assumed=lambda_assumed,
            n_times=n_times,
            fontsize_axlabel=fontsize_axlabel
        )

        export_singularity_video(
            t_sing=t_sing,
            si=si,
            charge=charge,
            t_axis=t_axis,
            s_rel=s_rel,
            phi_aligned=phi_aligned,
            centered=centered,
            Delta_t=Delta_t,
            lambda_assumed=lambda_assumed,
            n_times=n_times,
            fontsize_axlabel=fontsize_axlabel,
            video_filename=f"singularity_t{t_sing*1000:.0f}ms_s{si}.mp4",
            fps=4        # because 1 ms → 1 second in video
        )

def exp_group_plot(group_folders):
    for i,folder in enumerate(group_folders):
        phi_file = folder / "phi_omega_a.npz"
        if not phi_file.exists():
            continue
       
        data = np.load(phi_file, allow_pickle=True)
        # phi_g, omega_g, amp_g = data["global_phi_omega_a"]
        phi_l, omega_l, amp_l = data["local_phi_omega_a"]

        centered = data["centered"]


        plot_kymo(phi_l,centered,dt= 0.001, Ns=24)

def experimental_plot():
    """
    Group experimental segments by (genotype, ATP, KCL) using all_sharma_results.csv,
    then compute/save fluctuation results per group folder under OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT.
    """
    exp_root = Path(OUTPUT_DATA_SHARME_DIR)
    

    csv_path = exp_root / "all_sharma_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {csv_path}")

    groups = _build_experiment_groups(exp_root, csv_path)
    if not groups:
        print("⚠️ No experimental groups to process.")
        return

    for i,((genotype, ATP, KCL), group_folders) in tqdm(enumerate(groups.items()), desc="Processing experimental groups"):
        exp_group_plot(group_folders)

from utils_rdf.open_res import read_spde
def compute_phase_quantities(group_folders): 
    base_mat_dir = Path("") # add path
    for folder in group_folders:
        print(folder)
        phi_file = folder / "phi_omega_a.npz"
        if not phi_file.exists():
            continue
    
        data = np.load(phi_file, allow_pickle=True)
        phi_l, omega_l, amp_l = data["local_phi_omega_a"]


        folder_name = folder.name
        data = read_spde(base_mat_dir /f"{folder_name}.gz")

        gamma_mat = data["gamma_mat"]
        
        gamma_mat = np.asarray(data["gamma_mat"])

        gamma_mat = gamma_mat[INITIAL_TRANSIENT_TIME_SIMULATION:, :]

  
        
        plot_kymo(phi_l, gamma_mat, dt=DT_SIMULATION/0.45, Ns=100)

def load_metadata(summary_dir):
    """Load sample metadata (Nmotor, mu_a, folder path)."""
    meta_rows = []

    for folder in sorted(Path(summary_dir).iterdir()):
        if not folder.is_dir():
            continue

        param_file = folder / "parameters.npz"
        if not param_file.exists():
            continue

        try:
            params = dict(np.load(param_file, allow_pickle=True))

            def to_scalar(x):
                if isinstance(x, np.ndarray) and x.ndim == 0:
                    return x.item()
                elif isinstance(x, (list, tuple)) and len(x) == 1:
                    return x[0]
                else:
                    return x

            Nmotor = to_scalar(params.get("Nmotor", np.nan))
            mu_a = to_scalar(params.get("mu_a", np.nan))
           
            meta_rows.append({
                "sample": folder.name,
                "Nmotor": Nmotor,
                "mu_a": mu_a,
                "path": folder
            })

        except Exception as e:
            print(f"⚠️ Skipping {folder.name} (could not read params): {e}")

    return pd.DataFrame(meta_rows)


def sim_plot():
    meta_df = load_metadata(OUTPUT_DATA_NEW_PARAMETER_DIR)

    if meta_df.empty:
        print(f"No valid metadata found in {OUTPUT_DATA_NEW_PARAMETER_DIR}")
        return pd.DataFrame()


    groups = list(meta_df.groupby(["Nmotor", "mu_a"]))
    for (Nmotor, mu_a), group_df in tqdm(groups, desc="Processing groups"):
        group_folders = [Path(p) for p in group_df["path"]]
        compute_phase_quantities(group_folders)

def main():
    sim_plot()
    experimental_plot()



if __name__ == "__main__":
    main()
