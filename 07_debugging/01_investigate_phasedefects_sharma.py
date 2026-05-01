import os
from cv2 import phase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datastructures.special_source_names import ORIGINAL
from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SEGMENTS,
    PHASE_DEFECTS_SEGMENTS,
    TANGENT_ANGLE_SERIES,
)
from cilia.transformers.local_phase_transformer import segment_local_phase_t
from cilia.transformers.gauge_transformer import base_gauge_sym_segments_t


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


def get_items_by_key(realization, key):
    return [
        item for item in realization.data_items.values()
        if item.data_name.key == key
    ]


def split_tangent_angle_items(realization):
    orig = []
    gauged = []

    for item in get_items_by_key(realization, TANGENT_ANGLE_SEGMENTS.key):
        if ORIGINAL.key in item.dependencies:
            orig.append(item)
        elif item.algorithm == base_gauge_sym_segments_t.algorithm:
            gauged.append(item)

    return orig, gauged


def get_phase_items(realization):
    matches = [
        item for item in realization.data_items.values()
        if item.data_name.key == "local_phase_segments"
        and "segment_local_phase_t" in item.algorithm
    ]

    return matches


def get_defect_items(realization):
    return get_items_by_key(realization, PHASE_DEFECTS_SEGMENTS.key)


# ============================================================
# Plotting
# ============================================================

def plot_segment_bundle(
    tangent_orig,
    tangent_gauged,
    phase,
    amplitude,
    defects_dict,
    segment_idx,
):
    clean = defects_dict.get("clean", np.zeros((0, 3)))
    raw = defects_dict.get("raw", np.zeros((0, 3)))
    good_interval = defects_dict.get("good_s_interval", None)
    T_eff = defects_dict.get("effective_T", None)

    Nt, Ns = phase.shape

    # -------------------------------------------------
    # ✅ Proper 2D detrending (time + space)
    # -------------------------------------------------
    ta_detrended = tangent_orig.copy()

    # remove time-dependent drift (per frame)
    ta_detrended = ta_detrended - np.nanmean(ta_detrended, axis=1, keepdims=True)

    # remove spatial bias (global over s)
    ta_detrended = ta_detrended - np.nanmean(ta_detrended, axis=0, keepdims=True)

    # -------------------------------------------------
    # Defect counting region
    # -------------------------------------------------
    s_lo = int(0.15 * Ns)
    s_hi = int(0.7 * Ns)

    # -------------------------------------------------
    # Count defects
    # -------------------------------------------------
    n_pos = 0
    n_neg = 0

    if clean.size > 0:
        s = clean[:, 1]
        q = clean[:, 2]

        mask = (s >= s_lo) & (s <= s_hi)

        n_pos = np.sum(q[mask] > 0)
        n_neg = np.sum(q[mask] < 0)

    print(f"[SEG {segment_idx}] pos={n_pos}, neg={n_neg}, T_eff={T_eff}")

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    def plot_with_all(ax, data, title, cmap):
        if data is None:
            ax.axis("off")
            ax.set_title(title + " (N/A)")
            return None
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )

        ax.set_title(title)

        # --- RAW defects
        if raw.size > 0:
            ax.scatter(
                raw[:, 0],
                raw[:, 1],
                marker="o",
                color="grey",
                alpha=0.3,
                s=20,
                label="raw",
            )

        # --- CLEAN defects
        if clean.size > 0:
            t = clean[:, 0]
            s = clean[:, 1]
            q = clean[:, 2]

            pos = q > 0
            neg = q < 0

            ax.scatter(
                t[pos], s[pos],
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                label="pos",
            )

            ax.scatter(
                t[neg], s[neg],
                marker="x",
                color="black",
                linewidths=1.5,
                label="neg",
            )

        # --- GOOD interval
        if good_interval is not None and good_interval[0] >= 0:
            ax.axhline(good_interval[0], color="green", linestyle="--")
            ax.axhline(good_interval[1], color="green", linestyle="--")

        # --- COUNT interval
        ax.axhline(s_lo, color="red", linestyle=":", linewidth=2)
        ax.axhline(s_hi, color="red", linestyle=":", linewidth=2)

        return im

    # -------------------------------------------------
    # Layout (updated colormaps)
    # -------------------------------------------------
    plot_with_all(axes[0, 0], tangent_orig, f"Original TA (seg {segment_idx})", cmap="viridis")
    plot_with_all(axes[0, 1], ta_detrended, "TA detrended (time + space)", cmap="viridis")

    plot_with_all(axes[1, 0], tangent_gauged, "Gauged TA", cmap="viridis")
    plot_with_all(axes[1, 1], phase, "Phase", cmap="twilight")  # ✅ cyclic

    plot_with_all(axes[2, 0], amplitude, "Amplitude", cmap="viridis")

    axes[2, 1].axis("off")

    axes[0, 0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def get_exp_path(dataset, realization):
    return os.path.join(dataset.path, realization.experiment_id)

def inspect_experiment(dataset, exp_id):
    realization = get_realization(dataset, exp_id)
    exp_path = get_exp_path(dataset, realization)

    orig_items, gauged_items = split_tangent_angle_items(realization)
    phase_items = get_phase_items(realization)
    #print(phase_items)
    defect_items = get_defect_items(realization)

    if not (orig_items and gauged_items and phase_items and defect_items):
        print("⚠️ Missing required data for inspection")
        return

    orig_data = orig_items[0].resolve(dataset, exp_path)
    gauged_data = gauged_items[0].resolve(dataset, exp_path)
    phase_data = phase_items[0].resolve(dataset, exp_path)
    defect_data = defect_items[0].resolve(dataset, exp_path)
    for item in realization.data_items.values():
        if item.data_name.key == "dt_frame" :
            dt_item=item
            dt_frame = dt_item.resolve(dataset, exp_path)
            break

    for i, seg in enumerate(phase_data):
        phase = np.asarray(seg["phase"])
        if not np.isfinite(phase).any():
            print(f"[SKIP] segment {i}: phase is all NaN")
            continue

        amp = np.asarray(seg["amplitude"])

        tangent_orig = np.asarray(orig_data[i])
        tangent_gauged = np.asarray(gauged_data[i])

        defects = defect_data[i]
        # if i<22:
        #     continue
        print('dt_frame',dt_frame)
        # plot_simulation(
        #     dt_frame,
        #     tangent_gauged,
        #     phase,
        #     amp,
        #     defects,
        #     idx0=600, idx1=651,#idx0=2190, idx1=2241#
        #     use_time=True,
        # )

        animate_simulation(
            dt_frame,
            tangent_gauged,
            phase,
            amp,
            defects,
            idx0=2190,
            idx1=2241,
        )

        # plot_segment_bundle(
        #     tangent_orig,
        #     tangent_gauged,
        #     phase,
        #     amp,
        #     defects,
        #     i,
        # )


def plot_polar_string_snapshots(
    ax,
    phase,
    amplitude,
    times_main=None,      # explicit times to highlight
    step=None,            # optional background sampling
    cmap="viridis",
    alpha_main=0.9,
    alpha_bg=0.2,
    lw_main=2.0,
    lw_bg=1.0,
    start_marker="o",
    end_marker="^",
    use_time=False,
    dt_frame=1.0,
):
    """
    Polar plot of strings: r = a(s), theta = phi(s)

    Parameters
    ----------
    ax : matplotlib axis (must be polar!)
    times_main : array-like
        Specific time indices to highlight (colored)
    step : int or None
        If given, plot every `step` frame in grey as background
    """

    Nt = phase.shape[0]

    if times_main is None:
        times_main = np.arange(0, Nt, max(1, Nt // 10))

    times_main = np.array(times_main)

    cmap_obj = plt.get_cmap(cmap)
    if use_time:
        times_plot = times_main * dt_frame
    else:
        times_plot = times_main

    norm = mpl.colors.Normalize(# type: ignore
        vmin=times_plot.min(),
        vmax=times_plot.max()
    )

    # -------------------------------------------------
    # Background curves (grey)
    # -------------------------------------------------
    if step is not None and step > 0:
        times_bg = np.arange(0, Nt, step)

        for t in times_bg:
            phi = phase[t]
            amp = amplitude[t]

            mask = np.isfinite(phi) & np.isfinite(amp)
            if not np.any(mask):
                continue

            ax.plot(
                phi[mask],
                amp[mask],
                color="grey",
                alpha=alpha_bg,
                lw=lw_bg,
            )

    # -------------------------------------------------
    # Main curves (colored)
    # -------------------------------------------------
    for t in times_main:
        phi = phase[t]
        amp = amplitude[t]

        mask = np.isfinite(phi) & np.isfinite(amp)
        if not np.any(mask):
            continue

        value = t * dt_frame if use_time else t
        color = cmap_obj(norm(value))

        ax.plot(
            phi[mask],
            amp[mask],
            color=color,
            lw=lw_main,
            alpha=alpha_main,
        )

        # Direction markers
        ax.scatter(
            phi[mask][0],
            amp[mask][0],
            color=color,
            marker=start_marker,
            s=30,
            zorder=3,
        )

        ax.scatter(
            phi[mask][-1],
            amp[mask][-1],
            color=color,
            marker=end_marker,
            s=40,
            zorder=3,
        )

    # -------------------------------------------------
    # Polar grid styling
    # -------------------------------------------------
    #ax.set_title(title)

    # radial limits
    rmax = np.nanmax(amplitude)
    ax.set_ylim(0, 2)

    # radial grid (r=0, r=1 if meaningful)
    ax.set_yticks([0, 1])
    ax.set_thetagrids(np.arange(0, 360, 45))
    ax.set_yticklabels([])   # remove radial labels
    ax.set_xticklabels([])   # remove angular labels

    ax.grid(True, alpha=0.5)

    # -------------------------------------------------
    # Colorbar
    # -------------------------------------------------
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj) # type: ignore
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("t [ms]" if use_time else "frame", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    

def integrate_phase_gradient(dphi, axis=1):
    """
    Reconstruct phase by integrating gradient along s.
    Sets phi(s=0) = 0 (gauge choice).
    """
    phi = np.cumsum(dphi, axis=axis)
    zeros = np.zeros_like(phi.take(indices=[0], axis=axis))
    phi = np.concatenate([zeros, phi], axis=axis)
    return phi

def compute_mean_phase_profile(phase):
    """
    Compute smooth mean phase profile via gradient averaging.
    """
    dphi = np.diff(np.unwrap(phase, axis=1), axis=1)
    dphi_mean = np.nanmean(dphi, axis=0, keepdims=True)
    phi_mean = integrate_phase_gradient(dphi_mean, axis=1)
    phi_mean = phi_mean - np.nanmean(phi_mean, axis=1, keepdims=True)
    return phi_mean  # shape (1, Ns)

def phase_remove_profile_and_center(phase, phi_mean):
    """
    Used for Plot 2:
    - subtract mean phase profile
    - then fix mean phase per frame to zero
    """
    phase_corr = np.unwrap(phase, axis=1) - phi_mean
    phase_corr = phase_corr - np.nanmean(phase_corr, axis=1, keepdims=True)
    return phase_corr

def plot_simulation(
    dt_frame,
    tangent_angle,
    phase,
    amplitude,
    defects_dict,
    idx0=0, idx1=-1,
    use_time=True,
):
    dt_frame=dt_frame*1000
    tangent_angle = tangent_angle[idx0:idx1]
    phase = phase[idx0:idx1]
    amplitude = amplitude[idx0:idx1]
    clean = defects_dict.get("clean", np.zeros((0, 3)))
    raw = defects_dict.get("raw", np.zeros((0, 3)))
    #good_interval = defects_dict.get("good_s_interval", None)
    T_eff = defects_dict.get("effective_T", None)
    Nt, Ns = phase.shape

    # Defect counting region
    s_lo = int(0.15 * Ns)
    s_hi = int(0.7 * Ns)
    # s_hi =Ns-1
    # s_lo = int(0.8 * Ns)

    # Count defects
    n_pos = 0
    n_neg = 0

    if clean.size > 0:
        s = clean[:, 1]
        q = clean[:, 2]

        mask = (s >= s_lo) & (s <= s_hi)

        n_pos = np.sum(q[mask] > 0)
        n_neg = np.sum(q[mask] < 0)

    print(f" pos={n_pos}, neg={n_neg}, T_eff={T_eff}")


    # Plotting

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    def plot_with_all(ax, data, title, cmap, vmin=None, vmax=None):
        if data is None:
            ax.axis("off")
            ax.set_title(title + " (N/A)")
            return None
        Nt, Ns = data.shape

        if use_time:
            t = np.arange(Nt) * dt_frame
            extent = [t[0], t[-1], 0, 1]
        else:
            extent = [0, Nt, 0, 1]

        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        #ax.set_title(title)

        # --- RAW defects
        if raw.size > 0:
            t_raw = raw[:, 0] - idx0
            if use_time:
                t_raw = t_raw * dt_frame
            # ax.scatter(
            #     t_raw,
            #     raw[:, 1]/ Ns,
            #     marker="o",
            #     color="grey",
            #     alpha=0.3,
            #     s=20,
            #     label="raw",
            # )

        # --- CLEAN defects
        if clean.size > 0:
            t = clean[:, 0]-idx0
            if use_time:
                t = t * dt_frame
            s = clean[:, 1]/ Ns
            q = clean[:, 2]

            pos = q > 0
            neg = q < 0

            ax.scatter(
                t[pos], s[pos],
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                label="pos",
            )

            # ax.scatter(
            #     t[neg], s[neg],
            #     marker="x",
            #     color="black",
            #     linewidths=1.5,
            #     label="neg",
            # )

            # size of the cross
            dt_half = 2.5   # ms (horizontal half-length)
            ds_half = 0.1  # vertical half-length (in s/L units)

            for ti, si in zip(t[neg], s[neg]):
                # horizontal line
                ax.plot(
                    [ti - dt_half, ti + dt_half],
                    [si, si],
                    color="black",
                    linewidth=1,
                )

                # vertical line
                ax.plot(
                    [ti, ti],
                    [si - ds_half, si + ds_half],
                    color="black",
                    linewidth=1,
                )

        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(title, fontsize=14)
        # --- Custom colorbar ticks
        if "gamma" in title.lower():
            cbar.set_ticks([-1.3, 0.0, 1.3])

        elif "amplitude" in title.lower():
            cbar.set_ticks([0.0, 2.5])

        if title.lower().startswith("phase"):
            ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)
            #cbar.set_label("phase", fontsize=14)



        # --- COUNT interval
        ax.axhline(s_lo/Ns, color="white", linestyle=":", linewidth=2)
        ax.axhline(s_hi/Ns, color="white", linestyle=":", linewidth=2)

        if use_time:
            ax.set_xlabel(r"$t$ [ms]", fontsize=14)
        else:
            ax.set_xlabel("frame", fontsize=14)

        ax.set_ylabel(r"$s/L$", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        if use_time:
            ax.set_xticks(np.arange(0, 51, 10))
        return im

    
    # Left column: kymographs
    plot_with_all(axes[0, 0], tangent_angle, r"$\gamma(s,t)$", cmap="jet", vmin=-1.3, vmax=1.3)
    plot_with_all(axes[1, 0], phase, r"Phase $\phi(s,t)$", cmap="hsv")
    plot_with_all(axes[2, 0], amplitude, r"Local amplitude $\alpha(s,t)$", cmap="jet", vmin=0.0, vmax=2.5)

    for i in range(3):
        fig.delaxes(axes[i, 1])
        axes[i, 1] = fig.add_subplot(3, 2, 2*(i+1), projection="polar")

    phase=phase[:,s_lo:s_hi]
    amplitude=amplitude[:,s_lo:s_hi]

    phi_mean = compute_mean_phase_profile(phase)
    phase_centered = phase_remove_profile_and_center(phase, phi_mean)

    Nt = phase.shape[0]
    # simsetting:
    # times_main = np.arange(0,15,1, dtype=int) *5+230
    # step=3
    times_main = np.arange(0,6,1, dtype=int) *1+21
    step=1
    print(times_main)
    plot_polar_string_snapshots(
        axes[0, 1],
        phase,
        amplitude,
        times_main=times_main,
        step=step,   # grey background
        use_time=use_time,
        dt_frame=dt_frame,
        cmap="plasma",
    )

    plot_polar_string_snapshots(
        axes[1, 1],
        phase_centered + phi_mean,
        amplitude,
        times_main=times_main,
        step=step,
        use_time=use_time,
        dt_frame=dt_frame,
        cmap="plasma",
    )

    plot_polar_string_snapshots(
        axes[2, 1],
        phase_centered,
        amplitude,
        times_main=times_main,
        step=step,
        use_time=use_time,
        dt_frame=dt_frame,
        cmap="plasma",
    )

    #axes[0, 0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

import matplotlib.animation as animation

def reconstruct_xy(gamma, ds=1.0):
    """
    Reconstruct filament shape from tangent angle gamma(s)
    """
    theta = gamma
    x = np.cumsum(np.cos(theta)) * ds
    y = np.cumsum(np.sin(theta)) * ds

    # anchor at origin
    x = x - x[0]
    y = y - y[0]

    return x, y

import matplotlib.animation as animation
def animate_simulation(
    dt_frame,
    tangent_angle,
    phase,
    amplitude,
    defects_dict,
    idx0=0,
    idx1=-1,
    use_time=True,
    show_background=True,   
):

    dt_frame = dt_frame * 1000

    tangent_angle = tangent_angle[idx0:idx1]
    phase = phase[idx0:idx1]
    amplitude = amplitude[idx0:idx1]
    clean = defects_dict.get("clean", np.zeros((0, 3)))
    raw = defects_dict.get("raw", np.zeros((0, 3)))
    Nt, Ns = tangent_angle.shape

    # physical scaling
    L_total = 10.0  # µm
    ds = L_total / Ns

    # -------------------------------------------------
    # Figure layout
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)

    ax_kymo1 = fig.add_subplot(gs[0, 0])
    ax_kymo2 = fig.add_subplot(gs[1, 0])
    ax_kymo3 = fig.add_subplot(gs[2, 0])

    ax_gamma = fig.add_subplot(gs[0, 1])
    ax_shape = fig.add_subplot(gs[1:, 1])

    def draw_defects(ax):
        if clean.size == 0:
            return

        t = clean[:, 0] - idx0
        if use_time:
            t = t * dt_frame

        s = clean[:, 1] / Ns
        q = clean[:, 2]

        pos = q > 0
        neg = q < 0

        # --- positive defects (circles)
        ax.scatter(
            t[pos], s[pos],
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )

        # --- negative defects (custom +)
        dt_half = 2.5   # ms
        ds_half = 0.08

        for ti, si in zip(t[neg], s[neg]):
            ax.plot(
                [ti - dt_half, ti + dt_half],
                [si, si],
                color="black",
                linewidth=1,
                zorder=3,
            )
            ax.plot(
                [ti, ti],
                [si - ds_half, si + ds_half],
                color="black",
                linewidth=1,
                zorder=3,
            )

    # -------------------------------------------------
    # Kymographs
    # -------------------------------------------------
    def plot_kymo(ax, data, title, cmap, vmin=None, vmax=None, show_xlabel=True):
        t = np.arange(Nt) * dt_frame
        extent = [t[0], t[-1], 0, 1]

        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",  # IMPORTANT
        )

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 51, 10))

        if show_xlabel:
            ax.set_xlabel(r"$t$ [ms]", fontsize=16)
        else:
            ax.set_xticklabels([])

        ax.set_ylabel(r"$s/L$", fontsize=16)
        ax.tick_params(labelsize=14)

        # --- colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(title, fontsize=16)

        if "gamma" in title.lower():
            cbar.set_ticks([-1.3, 0.0, 1.3])

        elif "alpha" in title.lower():
            cbar.set_ticks([0.0, 2.5])

        elif "phi" in title.lower():
            ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)

        return im

    plot_kymo(ax_kymo1, tangent_angle, r"$\gamma(s,t)$", "jet", -1.3, 1.3, show_xlabel=False)
    plot_kymo(ax_kymo2, phase, r"$\phi(s,t)$", "hsv", show_xlabel=False)
    plot_kymo(ax_kymo3, amplitude, r"$\alpha(s,t)$", "jet", 0.0, 2.5, show_xlabel=True)
    draw_defects(ax_kymo1)
    draw_defects(ax_kymo2)
    draw_defects(ax_kymo3)

    # -------------------------------------------------
    # Moving time indicator
    # -------------------------------------------------
    time_lines = []
    for ax in [ax_kymo1, ax_kymo2, ax_kymo3]:
        line = ax.axvline(0, color="grey", linewidth=2)
        time_lines.append(line)

    # -------------------------------------------------
    # Right side
    # -------------------------------------------------
    s = np.linspace(0, 1, Ns)

    # γ(s)
    
    ax_gamma.set_xlim(0, 1)
    ax_gamma.set_ylim(-1.3, 1.3)
    ax_gamma.set_yticks([-1.3, 0.0, 1.3])
    ax_gamma.set_xlabel(r"$s/L$", fontsize=16)
    ax_gamma.set_ylabel(r"$\gamma(s)$", fontsize=16)
    ax_gamma.tick_params(labelsize=14)

    # shape (x,y)
    
    ax_shape.set_aspect("equal")

    ax_shape.set_xlim(0, 1.2*L_total)
    ax_shape.set_ylim(-1.2*L_total/2, 1.2*L_total/2)

    ax_shape.set_xlabel(r"$x\,[\mathrm{\mu m}]$", fontsize=16)
    ax_shape.set_ylabel(r"$y\,[\mathrm{\mu m}]$", fontsize=16)
    ax_shape.tick_params(labelsize=14)

    # -------------------------------------------------
    # Reconstruction
    # -------------------------------------------------
    def reconstruct_xy(gamma):
        x = np.cumsum(np.cos(gamma)) * ds
        y = np.cumsum(np.sin(gamma)) * ds
        x -= x[0]
        y -= y[0]
        return x, y

    # -------------------------------------------------
    # Precompute background curves
    # -------------------------------------------------

    if show_background:
        gamma_bg = tangent_angle.copy()
        # xy_bg = []
        for g in gamma_bg:
            xg, yg = reconstruct_xy(g)
            # xy_bg.append((xg, yg))
            ax_gamma.plot(
                s, g,
                color="grey",
                alpha=0.15,
                lw=1
            )
            ax_shape.plot(
                xg, yg,
                color="grey",
                alpha=0.15,
                lw=1
            )
    gamma_line, = ax_gamma.plot([], [], color="red", lw=2)
    shape_line, = ax_shape.plot([], [], color="red", lw=2)
    # -------------------------------------------------
    # Update
    # -------------------------------------------------
    def update(frame):
        t_val = frame * dt_frame
        gamma = tangent_angle[frame]

        gamma_line.set_data(s, gamma)

        x, y = reconstruct_xy(gamma)
        shape_line.set_data(x, y)

        for line in time_lines:
            line.set_xdata([t_val, t_val])

        return [gamma_line, shape_line] + time_lines

    # -------------------------------------------------
    # Animation
    # -------------------------------------------------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=300, #50 in simulation #300 exp
        blit=False,   # safer with colorbars
    )

    plt.tight_layout()
    ani.save("simulation.mp4", fps=4, dpi=200) #fps=20 simulation
    plt.show()

    return ani

def inspect_simulation(dataset, exp_id):
    realization = get_realization(dataset, exp_id)
    exp_path = get_exp_path(dataset, realization)
    #print(realization,exp_path)

    for item in get_items_by_key(realization, TANGENT_ANGLE_SERIES.key):
        if ORIGINAL.key in item.dependencies:
            tangent_angle_item=item
            break
    for item in realization.data_items.values():
        if item.data_name.key == "local_phase_series" and "local_phase_transformer" in item.algorithm:
            local_phase_item=item
            break
    for item in realization.data_items.values():
        if item.data_name.key == "phase_defects_series" :
            defect_items=item
            break
    for item in realization.data_items.values():
        if item.data_name.key == "dt_frame" :
            dt_item=item
            break

    if not (tangent_angle_item and local_phase_item and defect_items and dt_item):
        print("⚠️ Missing required data for inspection")
        return

    tangent_angle_data = tangent_angle_item.resolve(dataset, exp_path)
    local_phase_data = local_phase_item.resolve(dataset, exp_path)
    defect_data_data = defect_items.resolve(dataset, exp_path)
    dt_frame = dt_item.resolve(dataset, exp_path)


    phase = np.asarray(local_phase_data["phase"])
    if not np.isfinite(phase).any():
        print(f"[SKIP] phase is all NaN")
        return


    amp = np.asarray(local_phase_data["amplitude"])

    tangent_orig = np.asarray(tangent_angle_data)
    print(dt_frame)
    
    # plot_simulation(dt_frame,tangent_orig, phase, amp, defect_data_data, idx0=1000, idx1=1367)# idx0=58633, idx1=59000#
    animate_simulation(
        dt_frame,
        tangent_orig,
        phase,
        amp,
        defect_data_data,
        idx0=1000,
        idx1=1367,
    )


from open_res import read_spde
from cilia.datasets.data_loaders import register_data_loader
# ============================================================
# DATASET CLASS
# ============================================================
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

import pandas as pd

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
    # "39f9a29c98e8189b" unperturbed with defects in seg0 (pairs)
    # '5fcc8a13641dc628' unperturbed with defects seg 62 very nice
    # ''101d88ae042ce1cc' 100kcl very nice

    # '278bf5b0af886206' kcl 200 seg1 for die out
    # "f34b4da146115fec" kcl 300 just bad


    exp_id = '101d88ae042ce1cc'
    inspect_experiment(dataset, exp_id)

    # dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_3d_extraction')
    # dataset = SimulationDataset(dataset_path)
    # out_csv = f"./scalar_observables_cass_3d_extraction.csv"
    # df_sim=pd.read_csv(out_csv)
    # # print(df_sim.head())
    # exp_id='36ae3023208c098a'
    # inspect_simulation(dataset, exp_id)



if __name__ == "__main__":
    main()