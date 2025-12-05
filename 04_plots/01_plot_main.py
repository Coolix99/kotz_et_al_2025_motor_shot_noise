import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as colors
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm
from scipy.stats import linregress
import os
from scipy.spatial.distance import cdist

from utils_rdf.config import (OUTPUT_DATA_PHASESPACE_DIR, OUTPUT_DATA_EXTRACTION_DIR,
                              OUTPUT_DATA_SHARME_DIR, OUTPUT_FIGURES_DIR, OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR,
                              OUTPUT_DATA_FLUCTUATIONS_EXTRACTION,OUTPUT_DATA_NEW_PARAMETER_DIR,
                              OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT,OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER,
                              DT_SIMULATION, SPATIAL_CORRELATION_CSV, OUTPUT_DATA_WN_PHASESPACE_DIR, OUTPUT_DATA_CWN_PHASESPACE_DIR)

F_UPPER = 100
F_LOWER = 0
A_UPPER = 1.0
A_LOWER = 0.0
LAMBDA_LOWER = 1.0
LAMBDA_UPPER = 4.0
Q_LOWER = 0.1
Q_UPPER = 1e4

LAMBDA_THRESHOLD = 2.1

def load_data(csv_path):
    """Load the aggregated CSV data into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df

def group_and_average(df, x_key, y_key, value_key):
    """Group by (x, y) and compute mean of the value_key."""
    grouped = df.groupby([x_key, y_key])[value_key].mean().reset_index()
    pivot_table = grouped.pivot(index=y_key, columns=x_key, values=value_key)
    return pivot_table

def plot_phase_space(df, x_key, y_key, value_key, title=None, cmap="viridis"):
    """Plot 2D phase space with color-coded mean values."""
    pivot = group_and_average(df, x_key, y_key, value_key)
    pivot = pivot.sort_index(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,  
        xticklabels=pivot.columns.values,
        yticklabels=pivot.index.values,
        cmap=cmap,
        cbar_kws={"label": f"{value_key}"},
    )
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def get_analytic_values(eta,f_stern,zeta,mu,beta,num_modes_mu_a=5):
    n0 = eta / (eta + (1 - eta) * np.exp(f_stern))
    lambda_ = eta + (1 - eta) * np.exp(f_stern)
    kappa = (1 - eta) * n0 * np.exp(f_stern) * f_stern * zeta
    
    mu_a_crit_free = (mu+lambda_*beta) / (2 * kappa - 2 * (n0) * lambda_ * zeta)
    
    q_squared = [(np.pi / 2 + n * np.pi) ** 2 for n in range(num_modes_mu_a)]
    mu_a_crits = [(mu + q2 + lambda_ * beta) / (2 * kappa - 2 * n0 * lambda_ * zeta) for q2 in q_squared]
    
    return mu_a_crit_free,mu_a_crits

def plot_phase_space_interpolated_inv_N(
    df,
    value_key,
    title=None,
    cmap="viridis",
    mu_a_crit_free=None,
    mu_a_crits=None,
    scale=None,
    method='nearest',
    transitions=None,  
    plot_transition_scatter=True ,        
    fit_line=False,  
):
    """
    Interpolated phase–space over (mu_a, 1/Nmotor), optionally overlaying
    detected transition points from `transitions` DataFrame.
    """
    from scipy.spatial import cKDTree

    # compute group means
    grouped = df.groupby(["mu_a", "Nmotor"])[value_key].mean().reset_index()
    grouped["inv_Nmotor"] = 1.0 / grouped["Nmotor"]
    
    x = grouped["mu_a"].values
    y = grouped["inv_Nmotor"].values
    z = grouped[value_key].values

    # default scale = range_x / range_y
    if scale is None:
        scale = (x.max() - x.min()) / (y.max() - y.min())

    # scaled coords
    xs, ys = x, y * scale

    # regular grid in original coords
    xi = np.linspace(0, x.max(), 200)
    yi = np.linspace(0, y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Xi_s, Yi_s = Xi, Yi * scale

    # interpolate
    if method == 'nearest_l1':
        tree = cKDTree(np.column_stack([xs, ys]))
        pts = np.column_stack([Xi_s.ravel(), Yi_s.ravel()])
        _, idx = tree.query(pts, k=1, p=1)
        Zi = z[idx].reshape(Xi_s.shape)
    else:
        Zi = griddata((xs, ys), z, (Xi_s, Yi_s), method=method)

    # mask extrapolated by y‐intervals at each row
    x_data = np.sort(np.unique(x))
    y_data = np.sort(np.unique(y))
    mu_min_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].min() for yv in y_data}
    mu_max_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].max() for yv in y_data}

    mask = np.zeros_like(Zi, dtype=bool)
    tol = 1e-8
    for j, yv in enumerate(yi):
        diffs = np.abs(y_data - yv)
        i0 = np.argmin(diffs)
        if diffs[i0] < tol:
            mn, mx = mu_min_y[y_data[i0]], mu_max_y[y_data[i0]]
        else:
            below, above = y_data[y_data<yv], y_data[y_data>yv]
            if len(below) and len(above):
                low, high = below[-1], above[0]
                w_low = mu_max_y[low]-mu_min_y[low]
                w_high= mu_max_y[high]-mu_min_y[high]
                mn,mx = (mu_min_y[low], mu_max_y[low]) if w_low<w_high else (mu_min_y[high], mu_max_y[high])
            else:
                continue
        mask[j, :] = (Xi[j, :]>=mn) & (Xi[j, :]<=mx)
    Zi_masked = np.where(mask, Zi, np.nan)

    # plot heatmap
    plt.figure(figsize=(8,6))
    pcm = plt.pcolormesh(Xi, Yi, Zi_masked, shading="auto", cmap=cmap)
    plt.colorbar(pcm, label=value_key)

    # critical lines
    if mu_a_crit_free is not None:
        plt.axvline(mu_a_crit_free, linestyle="--", color="red", label="μₐᶜ (free)")
    if mu_a_crits is not None:
        for v in mu_a_crits:
            plt.axvline(v, linestyle=":", color="red", alpha=0.7)

    # overlay transition points
    if transitions is not None:
        mstyles = {
            "amplitude_band_10pct": "o",
            "log10_Q":               "s",
            "log10_wavelength":      "^",
            "peak_freq":           "D"
        }
        for q, marker in mstyles.items():
            pts_q = transitions.dropna(subset=[q, "Nmotor"])
            xa = pts_q[q].values
            ya = 1.0 / pts_q["Nmotor"].values
            color = "black" if q == value_key else "lightgrey"
            alpha = 1.0 if q == value_key else 0.6
            label = f"{q} transition" if q == value_key else None
            if plot_transition_scatter:
                plt.scatter(
                    xa, ya,
                    color=color,
                    marker=marker,
                    s=80,
                    alpha=alpha,
                    label=label
                )
    
    if fit_line and transitions is not None:
        # gather all (mu_a, 1/Nmotor) from every quantity
        qs = ["amplitude_band_10pct", "log10_Q", "log10_wavelength", "peak_freq"]
        xs, ys = [], []
        for q in qs:
            pts = transitions.dropna(subset=[q, "Nmotor"])
            xs.extend(pts[q].values)
            ys.extend((1.0 / pts["Nmotor"].values))

        if xs:
            m, n = np.polyfit(xs, ys, 1)
            y_line = m * xi + n
            plt.plot(xi, y_line, 'k--', linewidth=2, label="best‐fit line")

    plt.xlabel("μₐ")
    plt.ylabel("1 / Nmotor")
    if title:
        plt.title(title)
    plt.xlim(0, xi.max())
    plt.ylim(0, yi.max())
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

def detect_phase_transitions(df):
    """
    For each Nmotor > 100, mu_a > 500, detect transition points in various quantities.
    Aggregates multiple realizations at each mu_a by taking their mean before analysis.
    """
    # 1) filter
    df_f = df[(df["Nmotor"] < 5000)&(df["Nmotor"] > 100) & (df["mu_a"] > 400)].copy()
    
    df_f["log10_wavelength"] = np.log10(df_f["wavelength_rel"])
    df_f["log10_Q"] = np.log10(df_f["Q"])

    records = []
    for N, group in df_f.groupby("Nmotor"):
        # 2) aggregate by mu_a (mean over realizations)
        agg = (
            group
            .groupby("mu_a")[["amplitude_band_10pct", "log10_Q", "log10_wavelength", "peak_freq_hz"]]
            .mean()
            .reset_index()
            .sort_values("mu_a")
        )
        x = agg["mu_a"].values
        
        rec = {"Nmotor": N,
               "Nmotor_scaled":N*200}

        # amplitude_band_10pct: smallest local minimum
        y1 = agg["amplitude_band_10pct"].values
        local_mins = [i for i in range(1, len(y1)-1) if y1[i] < y1[i-1] and y1[i] < y1[i+1]]
        if local_mins:
            i_min = min(local_mins, key=lambda i: y1[i])
            rec["amplitude_band_10pct"] = x[i_min]
        else:
            rec["amplitude_band_10pct"] = np.nan

        # log_Q: global minimum
        y2 = agg["log10_Q"].values
        rec["log10_Q"] = x[np.argmin(y2)] if len(y2) else np.nan
        
        # log_wavelength: position of most negative first derivative
        y3 = agg["log10_wavelength"].values
        print(N,y3, x)
        slopes3 = np.gradient(y3, x)
        rec["log10_wavelength"] = x[np.argmin(slopes3)]

        # peak_freq_hz: position of largest first derivative
        y4 = agg["peak_freq_hz"].values
        slopes4 = np.gradient(y4, x)
        rec["peak_freq_hz"] = x[np.argmax(slopes4)]

        records.append(rec)

    return pd.DataFrame.from_records(records)

def plot_phase_space_interpolated_inv_N_ax(
    ax,
    df,
    value_key,
    cmap="viridis",
    mu_a_crit_free=None,
    mu_a_crits=None,
    transitions=None,
    fit_line=False,
    scale=None,
    method='nearest',
):
    """
    Like plot_phase_space_interpolated_inv_N but draws into a provided Axes 'ax'.
    """
    # compute group means
    grouped = df.groupby(["mu_a", "Nmotor"])[value_key].mean().reset_index()
    grouped["inv_Nmotor"] = 1.0 / grouped["Nmotor"]

    x = grouped["mu_a"].values
    y = grouped["inv_Nmotor"].values
    z = grouped[value_key].values

    # default scale = range_x / range_y
    if scale is None:
        scale = (x.max() - x.min()) / (y.max() - y.min())

    xs, ys = x, y * scale

    # grid
    xi = np.linspace(0, x.max(), 200)
    yi = np.linspace(0, y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Xi_s, Yi_s = Xi, Yi * scale

    # interpolate
    if method == 'nearest_l1':
        tree = cKDTree(np.column_stack([xs, ys]))
        pts = np.column_stack([Xi_s.ravel(), Yi_s.ravel()])
        _, idx = tree.query(pts, k=1, p=1)
        Zi = z[idx].reshape(Xi_s.shape)
    else:
        Zi = griddata((xs, ys), z, (Xi_s, Yi_s), method=method)

    # mask extrapolated
    y_data = np.sort(np.unique(y))
    mu_min_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].min() for yv in y_data}
    mu_max_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].max() for yv in y_data}

    mask = np.zeros_like(Zi, dtype=bool)
    tol = 1e-8
    for j, yv in enumerate(yi):
        diffs = np.abs(y_data - yv)
        i0 = np.argmin(diffs)
        if diffs[i0] < tol:
            mn, mx = mu_min_y[y_data[i0]], mu_max_y[y_data[i0]]
        else:
            below = y_data[y_data < yv]
            above = y_data[y_data > yv]
            if len(below) and len(above):
                low, high = below[-1], above[0]
                w_low = mu_max_y[low] - mu_min_y[low]
                w_high = mu_max_y[high] - mu_min_y[high]
                mn, mx = (mu_min_y[low], mu_max_y[low]) if w_low < w_high else (mu_min_y[high], mu_max_y[high])
            else:
                continue
        mask[j, :] = (Xi[j, :] >= mn) & (Xi[j, :] <= mx)
    Zi_masked = np.where(mask, Zi, np.nan)

    # plot
    c = ax.pcolormesh(Xi, Yi, Zi_masked, shading="auto", cmap=cmap)
    fig = ax.get_figure()
    fig.colorbar(c, ax=ax, label=value_key)

    # critical lines
    if mu_a_crit_free is not None:
        ax.axvline(mu_a_crit_free, linestyle="--", color="red")
    if mu_a_crits is not None:
        for v in mu_a_crits:
            ax.axvline(v, linestyle=":", color="red", alpha=0.7)

    # overlay transitions
    if transitions is not None:
        mstyles = {
            "peak_freq":           "D",
            "amplitude_band_10pct":"o",
            "log10_wavelength":      "^",
            "log10_Q":               "s",
        }
        for q, marker in mstyles.items():
            pts_q = transitions.dropna(subset=[q, "Nmotor"])
            xa = pts_q[q].values
            ya = 1.0 / pts_q["Nmotor"].values
            color = "black" if q == value_key else "lightgrey"
            alpha = 1.0 if q == value_key else 0.6
            ax.scatter(xa, ya, c=color, marker=marker, s=50, alpha=alpha)

    # optional global fit line
    if fit_line and transitions is not None:
        qs = ["peak_freq", "amplitude_band_10pct", "log10_wavelength", "log10_Q"]
        xs, ys = [], []
        for q in qs:
            pts = transitions.dropna(subset=[q, "Nmotor"])
            xs.extend(pts[q].values)
            ys.extend(1.0 / pts["Nmotor"].values)
        if xs:
            m, n = np.polyfit(xs, ys, 1)
            y_line = m * xi + n
            ax.plot(xi, y_line, 'k--', linewidth=2)

    ax.set_xlim(0, xi.max())
    ax.set_ylim(0, yi.max())
    ax.set_xlabel("μₐ")
    ax.set_ylabel("1 / Nmotor")

def plot_phase_space_interpolated_inv_N_ax_fig(
    ax,
    df,
    value_key,
    name=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    mu_a_crits=None,
    scale=None,
    method='nearest',
    fontsize_labels=14,
    fontsize_ticks=12,
    fontsize_colorbar=12,
    log_scale=False,
    usefull_lines=True
):
    """
    Like plot_phase_space_interpolated_inv_N but draws into a provided Axes 'ax'.
    """
    # Compute group means
    grouped = df.groupby(["mu_a", "Nmotor_scaled"])[value_key].mean().reset_index()
    grouped["inv_Nmotor"] = 1.0e4 / grouped["Nmotor_scaled"]

    x = grouped["mu_a"].values
    y = grouped["inv_Nmotor"].values
    z = grouped[value_key].values

    # Default scale
    if scale is None:
        scale = (x.max() - x.min()) / (y.max() - y.min())
    xs, ys = x, y * scale

    # Grid
    xi = np.linspace(0, x.max(), 35)
    yi = np.linspace(0, y.max(), 35)
    Xi, Yi = np.meshgrid(xi, yi)
    Xi_s, Yi_s = Xi, Yi * scale

    # Interpolate
    if method == 'nearest_l1':
        tree = cKDTree(np.column_stack([xs, ys]))
        pts = np.column_stack([Xi_s.ravel(), Yi_s.ravel()])
        _, idx = tree.query(pts, k=1, p=1)
        Zi = z[idx].reshape(Xi_s.shape)
    else:
        Zi = griddata((xs, ys), z, (Xi_s, Yi_s), method=method)

    # Mask extrapolated regions
    y_data = np.sort(np.unique(y))
    mu_min_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].min() for yv in y_data}
    mu_max_y = {yv: grouped.loc[np.isclose(grouped["inv_Nmotor"], yv), "mu_a"].max() for yv in y_data}

    mask = np.zeros_like(Zi, dtype=bool)
    tol = 1e-8
    for j, yv in enumerate(yi):
        diffs = np.abs(y_data - yv)
        i0 = np.argmin(diffs)
        if diffs[i0] < tol:
            mn, mx = mu_min_y[y_data[i0]], mu_max_y[y_data[i0]]
        else:
            below = y_data[y_data < yv]
            above = y_data[y_data > yv]
            if len(below) and len(above):
                low, high = below[-1], above[0]
                w_low = mu_max_y[low] - mu_min_y[low]
                w_high = mu_max_y[high] - mu_min_y[high]
                mn, mx = (mu_min_y[low], mu_max_y[low]) if w_low < w_high else (mu_min_y[high], mu_max_y[high])
            else:
                continue
        mask[j, :] = (Xi[j, :] >= mn) & (Xi[j, :] <= mx)
    Zi_masked = np.where(mask, Zi, np.nan)


    mu_min_interp = np.empty_like(yi)
    for j, yv in enumerate(yi):
        diffs = np.abs(y_data - yv)
        i0 = np.argmin(diffs)
        mu_min_interp[j] = mu_min_y[y_data[i0]]
    # build boolean mask where mu_a < mu_min  ⇒  no well‐defined oscillation
    grey_mask = Xi <= mu_min_interp[:, None]+100
    grey_overlay = np.where(grey_mask, 1.0, np.nan)
    grey_cmap = colors.ListedColormap(['lightgrey'])
    ax.pcolormesh(
        Xi, Yi, grey_overlay,
        shading="auto",
        cmap=grey_cmap,
        vmin=0, vmax=1,
        zorder=1
    )
   
    # Plot heatmap
    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        c = ax.pcolormesh(Xi, Yi, Zi_masked, shading="auto", cmap=cmap, norm=norm)
    else:
        c = ax.pcolormesh(Xi, Yi, Zi_masked, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    fig = ax.get_figure()
    cb = fig.colorbar(c, ax=ax, label=name)
    cb.ax.tick_params(labelsize=fontsize_ticks)
    cb.set_label(name, fontsize=fontsize_colorbar)

    # Critical lines
    if mu_a_crits is not None and usefull_lines:
        ax.axvline(mu_a_crits[0], linestyle=":", color="black", alpha=1.0)

    # Labels
    ax.set_xlim(0, xi.max())
    ax.set_ylim(0, yi.max())
    ax.set_xlabel("μₐ", fontsize=fontsize_labels)
    ax.set_ylabel(r"$10^4 / N_{motor}$", fontsize=fontsize_labels)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

def plot_figure_2(
    df,
    transitions,
    N0s,
    mu_a_crits=None,
    cmap='jet',
    fontsize_labels=14,
    fontsize_ticks=12,
    fontsize_colorbar=14,
    all_black=False,
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X'],
    colors_mua_scan = ['#b8b8b8','#5b5c5b', '#000000'],
    usefull_lines=True,
    fig_subdir="fig2"
):
    """
    4×2 overview:
      col 0: phase space (1/N vs μₐ)
      col 1: at fixed N0, quantity vs μₐ
    """

    quantities = [
        ("peak_freq_hz", r"Frequency $f_0$ $[Hz]$", F_LOWER, F_UPPER,False),          
        ("amplitude_band_10pct", r"Amplitude A [rad]", A_LOWER, A_UPPER, False),   
        ("wavelength_rel", r"Wavelength $\lambda/L$", LAMBDA_LOWER, LAMBDA_UPPER,False),      
        ("Q", r"Quality factor Q", Q_LOWER, Q_UPPER , True),         
    ]

    fig, axes = plt.subplots(
        4, 2, 
        figsize=(10, 16),
        gridspec_kw={"width_ratios": [0.5, 0.5]}  # left wider than right
    )

    for row, (q, name, vmin, vmax, log_scale) in enumerate(quantities):
        # Column 0: Phase space
        plot_phase_space_interpolated_inv_N_ax_fig(
            axes[row, 0], df, q,
            name=name,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            mu_a_crits=mu_a_crits,
            fontsize_labels=fontsize_labels,
            fontsize_ticks=fontsize_ticks,
            fontsize_colorbar=fontsize_colorbar,
            log_scale=log_scale,
            usefull_lines=usefull_lines
        )
        #axes[row, 0].set_box_aspect(0.8)  #set aspect ratio

        xi = np.linspace(0, df["mu_a"].max(), 200)
        qs = ["peak_freq_hz", "amplitude_band_10pct", "log10_wavelength", "log10_Q"]
        xs, ys = [], []
        for qt in qs:
            pts = transitions.dropna(subset=[qt, "Nmotor_scaled"])
            xs.extend(pts[qt].values)
            ys.extend(1.0e4 / pts["Nmotor_scaled"].values)
        if xs:
            m, n = np.polyfit(xs, ys, 1)
            y_line = m * xi + n
        if usefull_lines:
            axes[row, 0].plot(xi, y_line, 'k--', linewidth=2)

        # Column 1: Fixed N0
        for idx, N0 in enumerate(N0s):
            if q=='Q' and N0>1e10:
                continue
            color = 'black' if all_black else colors_mua_scan[idx % len(colors_mua_scan)]
            sub1 = df[df["Nmotor_scaled"] == N0].groupby("mu_a")[q].mean().reset_index()
           
            marker = markers[idx % len(markers)]
            # plot with marker
            line, = axes[row, 1].plot(
                sub1["mu_a"], sub1[q],
                linestyle='-',
                marker=marker,
                label=fr"$N_{{motor}}$ = {N0}",
                color=color
            )
           
            # styling
           
            if row==0 and usefull_lines:
                axes[row, 1].legend(fontsize=fontsize_ticks, loc='upper right')
            if log_scale:
                axes[row, 1].set_yscale('log')
            # horizontal line on left plot
            y0 = 1.0e4 / N0
            if usefull_lines:
                axes[row, 0].axhline(y0, color=color, linewidth=1.5)

            # compute transition μₐ
            mu_trans = (y0 - n) / m
            # dashed vertical on right
            if usefull_lines:
                axes[row, 1].axvline(mu_trans, color=color, linewidth=1.5, linestyle="dashed")

            # place the same marker on the left phase‐space at (μ_trans, 1/N0)
            if usefull_lines:
                axes[row, 0].plot(
                    50, y0,
                    marker=marker,
                    color=color,
                    markersize=12,
                    markeredgecolor='black'
                )
        axes[row, 1].set_xlabel("μₐ", fontsize=fontsize_labels)
        axes[row, 1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        axes[row, 1].set_ylim(vmin, vmax)
        axes[row, 1].set_xlim(0, df["mu_a"].max())
        if usefull_lines:
            axes[row, 1].axvline(mu_a_crits[0], linestyle=":", color="black", alpha=1.0)
        #axes[row, 1].set_box_aspect(0.9)  #set aspect ratio
          
    fig.tight_layout()
    outdir = os.path.join(OUTPUT_FIGURES_DIR, fig_subdir)
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, fig_subdir+"_python")
    fig.savefig(fname + ".svg")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=300)
    plt.show()

def _plot_family(
    df_data, q, row, ax_right, ax_phase,
    m, n,                  # line fit coefficients (phase-space separator)
    color, marker,    # styling
    all_black, 
    marker_size_phase_space,
    mua0, N0s=None,        # if N0s is None, auto-pick (N0*, mua0*) from df_data
    tol=0.01, 
    label_add=''
):
    # Make sure ratio exists and is consistent
    dfd = df_data.copy()
    dfd['Nscaled/mua'] = dfd['Nmotor_scaled'] / dfd['mu_a']

    families = []
    if N0s is not None:
        # Use provided (N0s, mua0)
        for N0 in N0s:
            families.append((float(N0), float(mua0)))
    else:
        # Auto-pick single family from this dataset:
        idx = (dfd['Nmotor_scaled']).astype(float).idxmax()
        N0_star  = float(dfd.loc[idx, 'Nmotor_scaled'])
        mua0_str = float(dfd.loc[idx, 'mu_a'])
        families.append((N0_star, mua0_str))
    
    for i, (N0_i, mua0_i) in enumerate(families):
        color = "black" if all_black else  color
        # transition μa from quadratic formula
        mu_trans = -n/(2*m) + np.sqrt((n*n)/(4*m*m) + (1e4*mua0_i)/(N0_i*m))

        # match constant ratio R0 = N0/mua0 within tolerance
        R0 = N0_i / mua0_i
        dfe = dfd[np.abs(dfd["Nscaled/mua"]/R0 - 1.0) <= tol]

        if dfe.empty:
            continue

        # include TW in grouping
        agg = (
            dfe.groupby(["Nmotor_scaled", "mu_a"])[[q, "TW"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("mu_a")
        )
        x = (agg["Nmotor_scaled"].to_numpy()) / N0_i   # N/N0
        y = agg[q].to_numpy()
        # mu_vals = agg["mu_a"].to_numpy()
        tw_vals = agg["TW"].to_numpy() > 0.5  # boolean array

        # --- Draw segments individually based on TW ---
        for j in range(len(x) - 1):
            linestyle = '-' if (tw_vals[j] and tw_vals[j + 1]) else ':'
            ax_right.plot(
                x[j:j+2], y[j:j+2],
                linestyle=linestyle,
                marker=marker,
                color=color,
                label=(rf"$N_0={N0_i:.0f}$, $\mu_{{a,0}}={mua0_i:g}$, {label_add}" if (row == 0 and  j == 0) else None)#N0s is None and
            )

        # left phase-space: draw reference line(s) and marker at (μa0, 1e4*μa0/N0)
        x_line = np.linspace(0.1*mua0_i, min(mu_trans, mua0_i), 2000)
        y_line = 1e4*mua0_i / x_line / N0_i
        ax_phase.plot(x_line, y_line, color=color, linewidth=1.5, linestyle=":")

        # solid segment after transition if it exists
        if mu_trans < mua0_i:
            x_line2 = np.linspace(mu_trans, mua0_i, 2000)
            y_line2 = 1e4*mua0_i / x_line2 / N0_i
            ax_phase.plot(x_line2, y_line2, color=color, linewidth=1.5)

            ax_phase.plot(
                x_line2[-1], y_line2[-1],
                marker=marker,
                color=color,
                markersize=marker_size_phase_space,
                markeredgecolor=color
            )
        else:
            ax_phase.plot(
                x_line[-1], y_line[-1],
                marker=marker,
                color=color,
                markersize=marker_size_phase_space,
                markeredgecolor=color
            )

def plot_figure_3_ABCD(df,axes_phasespace,cmap,mu_a_crits,fontsize_labels,
                       fontsize_ticks,fontsize_colorbar,transitions,axes_3,N0s_ext,markers,colors_ext,all_black,marker_size_phase_space,mua0_ext,
                       df_new_parameter,overlay_color,df_extraction):
    #region top 4 -> comparison
    quantities = [
        ("peak_freq_hz", r"Frequency $f_0$ $[Hz]$", F_LOWER, F_UPPER, "peak_freq_hz",False),          
        ("amplitude_band_10pct", r"Amplitude A [rad]", A_LOWER, A_UPPER, "amplitude_band_10pct",False),   
        ("wavelength_rel", r"Wavelength $\lambda/L$", LAMBDA_LOWER, LAMBDA_UPPER, "wavelength_rel",False),      
        ("Q", r"Quality factor Q", Q_LOWER, Q_UPPER, "Q", True),         
    ]
    agg_all = df.groupby(["mu_a", "Nmotor_scaled"])[[q[0] for q in quantities]].mean().reset_index()

    for row, (q, name, vmin, vmax, exp_col, log_scale) in enumerate(quantities):
        #  Phase space (left column of the 4-wide header row)
        plot_phase_space_interpolated_inv_N_ax_fig(
            axes_phasespace[row], df, q,
            name=name,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            mu_a_crits=mu_a_crits,
            fontsize_labels=fontsize_labels,
            fontsize_ticks=fontsize_ticks,
            fontsize_colorbar=fontsize_colorbar,
            log_scale=log_scale,
        )

        # linear fit used for transition construction
        xi = np.linspace(0, df["mu_a"].max(), 200)
        qs_fit = ["peak_freq_hz", "amplitude_band_10pct", "log10_wavelength", "log10_Q"]
        xs, ys = [], []
        for qt in qs_fit:
            pts = transitions.dropna(subset=[qt, "Nmotor_scaled"])
            xs.extend(pts[qt].values)
            ys.extend(1.0e4 / pts["Nmotor_scaled"].values)
        if xs:
            m, n = np.polyfit(xs, ys, 1)
            y_line = m * xi + n
            axes_phasespace[row].plot(xi, y_line, 'k--', linewidth=2)
        else:
            # fallback to avoid NameError if transitions is empty
            m, n = 1.0, 0.0

        #  Right panel selection for this row
        pos = (row//2, row%2)
        ax_right = axes_3[pos]
        ax_phase = axes_phasespace[row]

        # ---------- families from df_extraction (your original set) ----------
        df_extraction = df_extraction.copy()
        df_extraction['Nscaled/mua'] = df_extraction['Nmotor_scaled'] / df_extraction['mu_a']

        for idx, N0 in enumerate(N0s_ext):
            marker = markers[idx % len(markers)]
            _plot_family(
                df_data=df_extraction, q=q,row=row,
                ax_right=ax_right, ax_phase=ax_phase,
                m=m, n=n,
                color=colors_ext[idx % len(colors_ext)],
                marker=marker,
                all_black=all_black, 
                marker_size_phase_space=marker_size_phase_space,
                mua0=mua0_ext, N0s=[N0], tol=0.01,
            )

        # ---------- BLUE overlay from df_new_parameter (auto-pick N0*, μa0*) ----------
        if df_new_parameter is not None and not df_new_parameter.empty:          
            marker_blue = 'o'  # or pick another symbol if you prefer
            _plot_family(
                df_data=df_new_parameter, q=q,row=row,
                ax_right=ax_right, ax_phase=ax_phase,
                m=m, n=n,
                color=overlay_color,     # <-- will be blue by default
                marker=marker_blue,
                all_black=False,   # ensure we actually draw blue
                marker_size_phase_space=marker_size_phase_space,
                mua0=None, N0s=None,          # <-- auto-pick (N0*, μa0*) from df_new_parameter
                tol=0.01,
                label_add="New Param"
            )
            
        # ---------- styling for the right panel ----------
        ax_right.set_xlabel(r"$N/N_0$", fontsize=fontsize_labels)
        ax_right.set_ylabel(name, fontsize=fontsize_labels)
        ax_right.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        ax_right.set_ylim(vmin, vmax)
        if log_scale:
            ax_right.set_yscale('log')
        ax_right.set_xlim(0.5, 1.0)
        ax_right.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
        if mu_a_crits is not None:
            ax_right.axvline(mu_a_crits[0], linestyle=":", color="black", alpha=1.0)

        # === ADD EXPERIMENTAL OVERLAY (WT, ATP=750) ON THE RIGHT PANEL ===
        # Map KCl [mM] -> N/N0 for WT (given)
        kcl_to_relN = {0: 1.00, 50: 0.96, 100: 0.93, 200: 0.87, 300: 0.80, 400: 0.74}
        kcl_keys = np.array(sorted(kcl_to_relN.keys()))

        # Load experimental CSV (produced elsewhere)
        csv_path = OUTPUT_DATA_SHARME_DIR/"all_sharma_results.csv"
        
        df_exp = pd.read_csv(csv_path)
        
        # Filter: genotype WT and ATP=750
        df_exp = df_exp[(df_exp["genotype"] == "WT") & (df_exp["ATP"] == 750)]

        # Map KCL to N/N0 (robust to float KCL by snapping to nearest known level)
        kcl_vals = pd.to_numeric(df_exp["KCL"], errors="coerce")
        nearest = kcl_keys[np.argmin(np.abs(kcl_vals.to_numpy()[:, None] - kcl_keys[None, :]), axis=1)]
        relN = np.vectorize(kcl_to_relN.get)(nearest)
        df_exp = df_exp.assign(N_over_N0=relN)

        # Pick which experimental column to use for this quantity (from quantities tuple)
        y_exp = pd.to_numeric(df_exp[exp_col], errors="coerce")

        # Match simulation quantity semantics:
        # - log10_*: take log10 of the experimental value
        if q.startswith("log10_"):
            y_exp = np.log10(y_exp.replace(0, np.nan))

        df_plot = pd.DataFrame({"N_over_N0": df_exp["N_over_N0"], "y": y_exp}).replace([np.inf, -np.inf], np.nan).dropna()

        if not df_plot.empty:
            agg = df_plot.groupby("N_over_N0")["y"].agg(["mean", "std", "count"]).reset_index()
            agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

            # Error bars on the right column
            axes_3[pos].errorbar(
                agg["N_over_N0"],
                agg["mean"],
                yerr=agg["sem"],
                fmt="o",
                capsize=4,
                linewidth=2,
                markersize=6,
                label="Exp (WT, ATP = 750 μM)" if row == 0 else None,
                zorder=10,
                color='red'
            )

        if row==0:
            pass
            #axes_3[pos].legend(fontsize=fontsize_ticks)#, loc='upper right'

def plot_figure_3_E(
    ax,
    colors_ext,
    markers,
    fontsize_labels=16,
    fontsize_ticks=14,
    csv_path=SPATIAL_CORRELATION_CSV,
):
    """
    Plot correlation length (ξ) vs relative motor extraction (N/N0) using
    the CSV produced by collect_corr_length_to_csv(...).

    Style:
      • Simulations (20k, 100k, new param): connected markers, line style depends on TW.
      • Experiments: red error bars (no connecting line).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Correlation-length CSV not found: {csv_path}\n"
            "Run collect_corr_length_to_csv(...) first."
        )

    df = pd.read_csv(csv_path)

    def color_for_scenario(name: str):
        s = name.lower()
        if "sim n₀=17k" in s:
            return colors_ext[0],markers[0]
        elif "sim n₀=100k" in s:
            return colors_ext[1], markers[1]
        elif "new param" in s or "127" in s:
            return "tab:blue", 'o'
        elif "exp" in s or "wt" in s:
            return "red", 'o'
        else:
            return "black", 'o'

    # Ensure TW column exists; if missing, default to False
    
    
    df["TW"] = False
    df.loc[(df['scenario'] == 'Sim N₀=100k, μₐ≈1570') & (df['relN'] >= 0.66), "TW"] = True
    df.loc[(df['scenario'] == 'Sim new parameter') & (df['relN'] >= 0.66), "TW"] = True


    for scenario, g in df.groupby("scenario"):
        color, marker = color_for_scenario(scenario)
        g = g.sort_values("relN")
        x = g["relN"].to_numpy()
        y = g["corr_len"].to_numpy()
        e = g["corr_len_sem"].to_numpy()
        tw = g["TW"].astype(bool).to_numpy()

        if "exp" in scenario.lower() or "wt" in scenario.lower():
            # Experimental: only error bars (no line)
            ax.errorbar(
                x, y, yerr=e,
                fmt=marker,
                color=color,
                capsize=4,
                linewidth=2,
                markersize=6,
                label=scenario,
                zorder=10,
            )
        else:
            # Simulations: connected by solid or dotted lines based on TW
            for i in range(len(x) - 1):
                linestyle = '-' if (tw[i] and tw[i + 1]) else ':'
                ax.plot(
                    x[i:i + 2], y[i:i + 2],
                    linestyle=linestyle,
                    color=color,
                    lw=1.8,
                    marker=marker,
                    ms=5,
                    label=scenario if i == 0 else None,
                    zorder=4
                )
            # Add error bars on top of markers
            ax.errorbar(
                x, y, yerr=e,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                alpha=0.7,
                zorder=3
            )

    # Axes styling
    ax.set_xlabel(r"$N/N_0$", fontsize=fontsize_labels)
    ax.set_ylabel(r"Correlation length $\xi/L$", fontsize=fontsize_labels)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlim(0.5, 1.0)
    ax.set_yscale("log")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        pass
        #ax.legend(fontsize=fontsize_ticks, frameon=True)

def plot_figure_3(
    df,df_extraction,df_new_parameter,
    transitions,
    N0s_ext,
    mua0_ext,
    mu_a_crits=None,
    cmap='jet',
    fontsize_labels=16,
    fontsize_ticks=14,
    fontsize_colorbar=16,
    marker_size_phase_space=7,
    all_black=True,
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X'],
    colors_ext =  ['#b8b8b8','#5b5c5b', '#000000'],
    overlay_color="tab:blue",
):

    df['TW']= df['wavelength_rel']<LAMBDA_THRESHOLD
    df_extraction['TW']= df_extraction['wavelength_rel']<LAMBDA_THRESHOLD
    df_new_parameter['TW']= df_new_parameter['wavelength_rel']<LAMBDA_THRESHOLD
    
    fig_phasespace, axes_phasespace = plt.subplots(1, 4, figsize=(20, 5))

    fig_3, axes_3 = plt.subplots(3, 2, figsize=(10, 12))


    plot_figure_3_ABCD(df,axes_phasespace,cmap,mu_a_crits,fontsize_labels,
                       fontsize_ticks,fontsize_colorbar,transitions,axes_3,N0s_ext,markers,colors_ext,all_black,marker_size_phase_space,mua0_ext,
                       df_new_parameter,overlay_color,df_extraction)
       
    plot_figure_3_E(axes_3[2, 0], colors_ext, markers,fontsize_labels, fontsize_ticks)

    fig_phasespace.tight_layout()
    fig_3.tight_layout()

    outdir = os.path.join(OUTPUT_FIGURES_DIR, "fig3")
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "fig3_python")
    fig_3.savefig(fname + ".svg")
    fig_3.savefig(fname + ".pdf")
    fig_3.savefig(fname + ".png", dpi=300)
    plt.show()

def plot_inset_fig2_E_NQ(
    df,
    transitions,
    muas_for_N_scan,
    mu_a_crits=None,
    fontsize_labels=15,
    fontsize_ticks=13,
    all_black=True,
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X'],
    colors_N_scan =  ['#b8b8b8','#5b5c5b', '#000000'],
    usefull_lines=True,
    fig_subdir="fig2_insets"
):


    fig_N_Q, ax_N_Q = plt.subplots(figsize=(5, 5))
    (q, name, vmin, vmax,exp_col, log_scale) = ("Q", r"Quality factor Q", 1, 1000, "Q_given", True)
    qs = ["peak_freq_hz", "amplitude_band_10pct", "log10_wavelength", "log10_Q"]
    xs, ys = [], []
    for qt in qs:
        pts = transitions.dropna(subset=[qt, "Nmotor_scaled"])
        xs.extend(pts[qt].values)
        ys.extend(1.0e4 / pts["Nmotor_scaled"].values)
    if xs:
        m, n = np.polyfit(xs, ys, 1)
    # Column 1: Fixed mua
    for idx, mua in enumerate(muas_for_N_scan):
        sub1 = df[df["mu_a"] == mua].groupby("Nmotor_scaled")[q].mean().reset_index()
        marker = markers[idx % len(markers)]
        color = 'black' if all_black else colors_N_scan[idx % len(colors_N_scan)]
        # plot with marker
        line, = ax_N_Q.plot(
            sub1["Nmotor_scaled"], sub1[q],
            linestyle='-',
            marker=marker,
            label=fr"$\mu_a$ = {mua}",
            color=color
        )
        color = line.get_color()

        # styling
        ax_N_Q.set_xlabel(r"$N_{motor}$", fontsize=fontsize_labels)
        ax_N_Q.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        ax_N_Q.set_ylim(Q_LOWER, Q_UPPER)
        if log_scale:
            ax_N_Q.set_yscale('log')
        ax_N_Q.set_xlim(df["Nmotor_scaled"].min(), 5000.0*200)
        ax_N_Q.set_xscale('log')
        if usefull_lines:
            ax_N_Q.legend(fontsize=fontsize_ticks, loc='upper right')

      
        N_trans = 1e4/(mua*m+n)
        # dashed vertical on right
        if usefull_lines:
            ax_N_Q.axvline(N_trans, color=color, linewidth=1.5, linestyle="dashed")

        # place the same marker on the left phase‐space at
       
    if usefull_lines and mu_a_crits is not None:
        ax_N_Q.axvline(mu_a_crits[0], linestyle=":", color="black", alpha=1.0)

    #scaling law
    if usefull_lines:
        x_vals = np.logspace(
            np.log10(df["Nmotor_scaled"].min()),
            np.log10(10000*200),
            20
        )
        ax_N_Q.plot(
            x_vals,
            x_vals/5e3,
            linestyle=':',
            color='red',
            label=r'$\log_{10}(N_{\mathrm{motor\_scaled}})$'
        )

    fig_N_Q.tight_layout()
    outdir = os.path.join(OUTPUT_FIGURES_DIR, fig_subdir)
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, fig_subdir+"_E_NQ_python")
    fig_N_Q.savefig(fname + ".svg")

    plt.show()

def plot_inset_fig3_F(
    selections=None,
    csv_path: Path = Path(OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR) / "phase_defect_rates_vs_relN.csv",
    fontsize_labels=16,
    fontsize_ticks=14,
    colors_ext=['#b8b8b8', '#5b5c5b', '#000000'],
    figsize=(6, 5),
):
    

    # ---- Default selections if none provided ----
    if selections is None:
        selections = [
            ("Exp (WT, ATP=750 μM)", 0.74),
            ("Sim new parameter", 0.7411764705882353),
            ("Exp (WT, ATP=750 μM)", 1.0),
        ]

    # ---- Load CSV ----
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    # ---- Color mapping ----
    def color_for(scenario_name):
        name = scenario_name.lower()
        if "exp" in name or "wt" in name:
            return "red"
        if "new param" in name:
            return "tab:blue"
        if "100k" in name:
            return colors_ext[1]
        if "17k" in name:
            return colors_ext[0]
        return "black"

    # ---- Gather selected rows ----
    extracted = []
    for scenario_name, relN_target in selections:
        df_s = df[df["scenario"] == scenario_name]

        if df_s.empty:
            raise RuntimeError(f"Scenario not found: {scenario_name}")

        # match relN exactly (within floating tolerance)
        match = df_s[np.isclose(df_s["relN"], relN_target, atol=1e-6)]

        if match.empty:
            raise RuntimeError(
                f"No row for scenario '{scenario_name}' with relN={relN_target}"
            )

        extracted.append(match.iloc[0])

    sel = pd.DataFrame(extracted)

    # ---- Extract data for plotting ----
    labels = sel["scenario"].tolist()
    pos_values = sel["pos_mean"].to_numpy()
    pos_err = sel["pos_sem"].to_numpy()
    neg_values = sel["neg_mean"].to_numpy()
    neg_err = sel["neg_sem"].to_numpy()

    x = np.arange(len(sel))
    bar_width = 0.35

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    for i, scenario_name in enumerate(labels):
        color = color_for(scenario_name)

        # +1 defects
        ax.bar(
            x[i] - bar_width/2,
            pos_values[i],
            bar_width,
            yerr=pos_err[i],
            color=color,
            edgecolor="black",
            hatch="///",
            capsize=4,
            label="+1 defects" if i == 0 else "",
        )

        # –1 defects
        ax.bar(
            x[i] + bar_width/2,
            neg_values[i],
            bar_width,
            yerr=neg_err[i],
            color=color,
            edgecolor="black",
            hatch="...",
            capsize=4,
            label="−1 defects" if i == 0 else "",
        )

    # ---- Styling ----
    xticklabels = [
        f"{scen}\n(N = {int(round(relN * 100))}%)"
        for scen, relN in zip(sel["scenario"], sel["relN"])
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=fontsize_ticks)

    ax.set_yscale("log")
    ax.set_ylabel(r"Defect rate [s$^{-1}$]", fontsize=fontsize_labels)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.legend(fontsize=fontsize_labels - 2)

    fig.tight_layout()

    # ---- Save ----
    outdir = Path(OUTPUT_FIGURES_DIR) / "fig3_insets"
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / "fig3_F_phase_defect_rates_custom.svg")
    fig.savefig(outdir / "fig3_F_phase_defect_rates_custom.pdf")
    fig.savefig(outdir / "fig3_F_phase_defect_rates_custom.png", dpi=300)

    plt.show()

def plot_SI_global_noniso(colors_ext = ['#b8b8b8','#5b5c5b', '#000000'], fontsize_labels=16, fontsize_ticks=14):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def plot_aniso_relation(path, color, label, A_key="A", omega_key="omega", 
                            mask_thresh=0.2, linewidth=2, contour=True):
        """Helper to plot A–ω density contours and regression line."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        amps = np.asarray(data[A_key], float)
        omegas = np.asarray(data[omega_key], float)

        # Filter out extreme outliers
        mask = (np.abs(amps - 1) < mask_thresh) & (np.abs(omegas - 1) < mask_thresh)
        amps, omegas = amps[mask], omegas[mask]
        if amps.size == 0 or omegas.size == 0:
            print(f"⚠️ No valid data in {path}")
            return None

        # -- Plot density contours
        if contour:
            nbins = 35
            H, xedges, yedges = np.histogram2d(amps, omegas, bins=nbins, density=True)
            if np.any(H > 0):
                Xc, Yc = np.meshgrid(
                    0.5 * (xedges[:-1] + xedges[1:]),
                    0.5 * (yedges[:-1] + yedges[1:])
                )
                levels = np.percentile(H[H > 0], [50, 70, 85, 97.5])
                ax.contour(Xc, Yc, H.T, levels=levels, colors=color, linewidths=1)

        # -- Linear regression
        res = linregress(amps, omegas)
        print(f"{label}:")
        print(f"  slope = {res.slope:.4f}")
        print(f"  intercept = {res.intercept:.4f}")
        print(f"  slope std error = {res.stderr:.4g}")
        print(f"  p-value (significance of slope!=0) = {res.pvalue:.3e}")

        xx = np.linspace(amps.min(), amps.max(), 200)
        ax.plot(xx, res.slope * xx + res.intercept, color=color, linewidth=linewidth,
                label=f"{label},  δω = {res.slope:.2f} δA")

    # === Simulation ===
    plot_aniso_relation(
        OUTPUT_DATA_FLUCTUATIONS_EXTRACTION / "Nmotor_500.0_mu_1570.0" / "global_aniso.npz",
        color=colors_ext[1],
        label=r"Sim ($N=100000$, $\mu_a=1570$)"
    )

    # === Experimental (WT, ATP=750 μM) ===
    plot_aniso_relation(
        OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT / "genotype_WT_ATP_750_KCL_0" / "global_aniso.npz",
        color="lightcoral",
        label=r"Exp (WT, ATP=750 μM)"
    )

    # === New parameter dataset ===
    plot_aniso_relation(
        OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER / "Nmotor_85.0_mu_365.0" / "global_aniso.npz",
        color="blue",
        label=r"Sim (new param)"
    )

    # === Styling ===
    ax.set_xlabel(r"rel. amplitude $\delta A$", fontsize=fontsize_labels)
    ax.set_ylabel(r"rel. frequency $\delta\omega$", fontsize=fontsize_labels)
    ax.set_xlim(0.8, 1.2)
    ax.set_ylim(0.8, 1.2)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    percent_fmt = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    ax.xaxis.set_major_formatter(percent_fmt)
    ax.yaxis.set_major_formatter(percent_fmt)
    ax.legend(fontsize=fontsize_ticks - 1)
    ax.set_aspect('equal', adjustable='box')


    figure_path = OUTPUT_FIGURES_DIR / "SI"
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path / "SI_global_noniso_relation.svg")
    fig.savefig(figure_path / "SI_global_noniso_relation.pdf")
    fig.savefig(figure_path / "SI_global_noniso_relation.png", dpi=300)
    plt.show()

def plot_SI_global_noniso_phi(colors_ext = ['#b8b8b8','#5b5c5b', '#000000'], fontsize_labels=16, fontsize_ticks=14):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def plot_aniso_phi(path, color, label, linewidth=2):
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        phi_grid=data['phi_grid']
        chi_all=data['chi_all']
        for chi in chi_all:
            ax.plot(phi_grid, chi, color=color, alpha=0.2, linewidth=1)
        chi_mean = np.mean(np.vstack(chi_all), axis=0)
        ax.plot(phi_grid, chi_mean, color=color, linewidth=linewidth, label=label)

    # === Simulation ===
    plot_aniso_phi(
        OUTPUT_DATA_FLUCTUATIONS_EXTRACTION / "Nmotor_500.0_mu_1570.0" / "global_aniso_phase.npz",
        color=colors_ext[1],
        label=r"Sim ($N=100000$, $\mu_a=1570$)"
    )

    # === Experimental (WT, ATP=750 μM) ===
    plot_aniso_phi(
        OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT / "genotype_WT_ATP_750_KCL_0" / "global_aniso_phase.npz",
        color="lightcoral",
        label=r"Exp (WT, ATP=750 μM)"
    )

    # === New parameter dataset ===
    plot_aniso_phi(
        OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER / "Nmotor_85.0_mu_365.0" / "global_aniso_phase.npz", #todo replace with 85
        color="blue",
        label=r"Sim (new param)"
    )

    # === Styling ===
    ax.set_xlabel(r"Phase φ (rad)", fontsize=fontsize_labels)
    ax.set_ylabel(r"χ(φ)", fontsize=fontsize_labels)
    ax.set_xlim(0, 2*np.pi)
    #ax.set_ylim(-0.5, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.legend(fontsize=fontsize_ticks - 1)
    
    figure_path = OUTPUT_FIGURES_DIR / "SI"
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path / "SI_global_noniso_phi_scatter.svg")
    fig.savefig(figure_path / "SI_global_noniso_phi_scatter.pdf")
    fig.savefig(figure_path / "SI_global_noniso_phi_scatter.png", dpi=300)
    plt.show()

def plot_SI_phase_defect_rates_vs_N(
    colors_ext = ['#b8b8b8','#5b5c5b', '#000000'],
    cutoff=7.0,
    fontsize_labels=16,
    fontsize_ticks=14,
    csv_path: Path = Path(OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR) / "phase_defect_rates_vs_relN.csv",
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Precomputed CSV not found: {csv_path}\n"
            "Run collect_phase_defect_rates_to_csv(...) first."
        )

    df = pd.read_csv(csv_path)

    # Optional cutoff consistency check
    if "cutoff" in df.columns:
        used_cutoffs = np.unique(df["cutoff"].astype(float))
        if not np.any(np.isclose(used_cutoffs, float(cutoff))):
            print(f"⚠️ CSV computed with cutoff(s) {used_cutoffs}, plot uses {cutoff}")

    # --- Map scenario to color
    def color_for_scenario(name: str):
        name = name.lower()
        if "sim n₀=17k" in name:
            return colors_ext[0]
        elif "sim n₀=100k" in name:
            return colors_ext[1]
        elif "new param" in name or "127" in name:
            return "tab:blue"
        elif "exp" in name or "wt" in name:
            return "red"
        else:
            return "black"

    # --- Helper: sanitize values for log plot
    def sanitize_for_log(values, errors):
        """
        Remove zeros & adjust error bars so lower bound never exceeds
        2 orders of magnitude below the value.
        Returns: (values_new, err_low, err_high, overflow_mask)
        """
        vals = np.asarray(values, float)
        errs = np.asarray(errors, float)

        # Remove zero or negative values
        mask = vals > 0
        vals = vals[mask]
        errs = errs[mask]

        # Compute symmetric errors
        err_low = errs.copy()
        err_high = errs.copy()


        lower_limit = vals * 1e-1
        actual_lower = vals - err_low

        overflow = actual_lower < lower_limit
        # clamp the lower error to allowed minimum
        err_low = np.where(overflow, vals - lower_limit, err_low)

        return vals, err_low, err_high, overflow, mask

    # --- Plot per scenario ---
    for scenario, g in df.groupby("scenario"):

        color = color_for_scenario(scenario)
        g = g.sort_values("relN")
        x_full = g["relN"].to_numpy()

        # ---- POSITIVE DEFECTS ----
        vals, err_low, err_high, overflow, mask = sanitize_for_log(
            g["pos_mean"], g["pos_sem"]
        )
        x = x_full[mask]

        ax.errorbar(
            x, vals,
            yerr=[err_low, err_high],
            fmt="o-",
            lw=1.8, ms=5, capsize=4,
            color=color,
            label=f"{scenario} (+)"
        )

        # downward arrows for overflow
        for xi, yi, ov in zip(x, vals, overflow):
            if ov:
                ax.plot(xi, yi * 1e-1, marker="v", color=color, markersize=7)

        # ---- NEGATIVE DEFECTS ----
        vals, err_low, err_high, overflow, mask = sanitize_for_log(
            g["neg_mean"], g["neg_sem"]
        )
        x = x_full[mask]

        ax.errorbar(
            x, vals,
            yerr=[err_low, err_high],
            fmt="o--",
            lw=1.8, ms=5, capsize=4,
            color=color,
            label=f"{scenario} (−)"
        )

        # downward arrows for overflow
        for xi, yi, ov in zip(x, vals, overflow):
            if ov:
                ax.plot(xi, yi * 1e-1, marker="v", color=color, markersize=7)

    # --- Axes styling ---
    ax.set_xlabel(r"$N/N_0$ (%)", fontsize=fontsize_labels)
    ax.set_ylabel(r"Defect rate [s$^{-1}$]", fontsize=fontsize_labels)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=fontsize_ticks, ncol=1, frameon=True)

    # --- Save ---
    figure_path = OUTPUT_FIGURES_DIR / "SI"
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path / "SI_phase_defect_rates_vs_N.svg")
    fig.savefig(figure_path / "SI_phase_defect_rates_vs_N.pdf")
    fig.savefig(figure_path / "SI_phase_defect_rates_vs_N.png", dpi=300)
    plt.show()

def plot_SI_global_noniso_combined(
    colors_ext=['#b8b8b8', '#5b5c5b', '#000000'],
    fontsize_labels=20,
    fontsize_ticks=18,
    figsize=(16, 7),
):
    """
    Combined two-panel SI figure:
       Left:   PCA relation between α and ω/ω0  (with percent ticks)
       Right:  Phase-dependent χ(φ) shifted so that mean χ = PCA slope

    This replaces:
       plot_SI_global_noniso()
       plot_SI_global_noniso_phi()
    """

    from sklearn.decomposition import PCA
    import numpy as _np

    # ======================================================================
    # Panel A helper: PCA relation alpha–omega
    # ======================================================================
    def plot_panel_A(ax, path, color, label,
                     A_key="A", omega_key="omega",
                     mask_thresh=0.2, linewidth=3, contour=True):

        path = Path(path)
        data = np.load(path, allow_pickle=True)

        amps = _np.asarray(data[A_key], float)
        omegas = _np.asarray(data[omega_key], float)

        mask = (abs(amps - 1) < mask_thresh) & (abs(omegas - 1) < mask_thresh)
        amps = amps[mask]
        omegas = omegas[mask]

        if amps.size == 0:
            print(f"⚠ No valid amplitude/frequency data in {path}")
            return None

        alpha = amps
        omega_rel = omegas

        # ---- density contours ----
        if contour:
            nbins = 40
            H, xedges, yedges = _np.histogram2d(alpha, omega_rel, bins=nbins, density=True)
            if H.any():
                Xc, Yc = _np.meshgrid(
                    0.5 * (xedges[:-1] + xedges[1:]),
                    0.5 * (yedges[:-1] + yedges[1:])
                )
                levels = _np.percentile(H[H > 0], [50, 70, 85, 97.5])
                ax.contour(Xc, Yc, H.T, levels=levels, colors=color, linewidths=1.2)

        # ---- PCA ----
        X = _np.column_stack([alpha, omega_rel])
        X_centered = X - X.mean(axis=0)
        pca = PCA(n_components=2).fit(X_centered)
        vx, vy = pca.components_[0]

        slope = vy / vx if abs(vx) > 1e-12 else _np.inf

        print(f"{label}:")
        print(f"  PCA slope = {slope:.3f}")
        print(f"  PCA explained variance = {pca.explained_variance_ratio_[0]:.3f}")

        # ---- regression line ----
        center = X.mean(axis=0)
        xx = _np.linspace(alpha.min(), alpha.max(), 200)
        yy = center[1] + slope * (xx - center[0])
        ax.plot(xx, yy, color=color, linewidth=linewidth,
                label=f"{label}, " +r"$\chi_\text{global}=$"+f"{slope:.2f}")

        return slope

    # ======================================================================
    # Panel B helper: χ(φ)
    # ======================================================================
    def plot_panel_B(ax, path, color, label, slope, linewidth=3):
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        phi_grid = data["phi_grid"]
        chi_all = data["chi_all"]

        # shift χ curves by PCA slope
        chi_shifted = [chi + slope - np.mean(chi) for chi in chi_all]

        # scatter realizations
        for chi in chi_shifted:
            ax.plot(phi_grid, chi, color=color, alpha=0.2, linewidth=1)

        chi_mean = _np.mean(_np.vstack(chi_shifted), axis=0)
        ax.plot(phi_grid, chi_mean, color=color, linewidth=linewidth, label=label)

    # ======================================================================
    # Create figure with two panels
    # ======================================================================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

    # ======================================================================
    # Panel A (compute slopes)
    # ======================================================================
    slopes = {}

    slopes["sim100k"] = plot_panel_A(
        axA,
        OUTPUT_DATA_FLUCTUATIONS_EXTRACTION / "Nmotor_500.0_mu_1570.0" / "global_aniso.npz",
        color=colors_ext[1],
        label=r"Simulation (old parameters)"
    )
    slopes["newparam"] = plot_panel_A(
        axA,
        OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER / "Nmotor_85.0_mu_365.0" / "global_aniso.npz",
        color="blue",
        label=r"Simulation (new parameters)"
    )
    slopes["exp"] = plot_panel_A(
        axA,
        OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT / "genotype_WT_ATP_750_KCL_0" / "global_aniso.npz",
        color="lightcoral",
        label=r"Experiment"
    )

    # ---- styling: percent ticks ----
    percent_fmt = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    axA.set_xlabel(r"Relative instantaneous amplitude $\alpha$", fontsize=fontsize_labels)
    axA.set_ylabel(r"Relative instantaneous frequency $\omega/\omega_0$", fontsize=fontsize_labels)
    axA.set_xlim(0.8, 1.2)
    axA.set_ylim(0.8, 1.2)
    axA.xaxis.set_major_formatter(percent_fmt)
    axA.yaxis.set_major_formatter(percent_fmt)
    axA.tick_params(axis="both", labelsize=fontsize_ticks)
    axA.legend(fontsize=fontsize_ticks-2)
    axA.set_aspect("equal", adjustable="box")

    # ======================================================================
    # Panel B: χ(φ) shifted by PCA slope
    # ======================================================================
    plot_panel_B(
        axB,
        OUTPUT_DATA_FLUCTUATIONS_EXTRACTION / "Nmotor_500.0_mu_1570.0" / "global_aniso_phase.npz",
        color=colors_ext[1],
        label=r"Sim ($N=100000$, $\mu_a=1570$)",
        slope=slopes["sim100k"]
    )

    plot_panel_B(
        axB,
        OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT / "genotype_WT_ATP_750_KCL_0" / "global_aniso_phase.npz",
        color="lightcoral",
        label=r"Exp (WT, ATP=750 μM)",
        slope=slopes["exp"]
    )

    plot_panel_B(
        axB,
        OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER / "Nmotor_85.0_mu_365.0" / "global_aniso_phase.npz",
        color="blue",
        label=r"Sim (new param)",
        slope=slopes["newparam"]
    )

    # ---- Custom π ticks ----
    axB.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axB.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    axB.set_xlabel(r"Phase $\varphi$ (rad)", fontsize=fontsize_labels)
    axB.set_ylabel(r"$\chi(\varphi)$", fontsize=fontsize_labels)
    axB.tick_params(axis="both", labelsize=fontsize_ticks)
    axB.legend(fontsize=fontsize_ticks-2)

    fig.tight_layout()

    # ======================================================================
    # Save
    # ======================================================================
    outdir = Path(OUTPUT_FIGURES_DIR) / "SI"
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / "SI_global_noniso_combined.svg")
    fig.savefig(outdir / "SI_global_noniso_combined.pdf")
    fig.savefig(outdir / "SI_global_noniso_combined.png", dpi=300)

    plt.show()

def plot_defect_distributions_multiroot(
        conditions,
        bins=13,
        cutoff=7,
        figsize=(7, 9),
):
    """
    Plot total defect spatial histograms (rates 1/s) for selected conditions,
    vertically stacked, with histogram bars and logarithmic y-axis.

    Colors:
        experiment = red
        simulation = blue
    """

    results = {}

    # === LOAD ALL CONDITIONS ===
    for cond in conditions:
        label = cond["label"]
        root_dir = Path(cond["root"])
        folder_key = cond["folder"]
        dt = cond["dt"]
        Ns_small, Ns_big = cond["Ns"]

        # Detect experiment vs simulation
        cond_color = "red" if "genotype" in folder_key else "blue"

        # Find folder
        folder = next((f for f in root_dir.glob("*") if folder_key in f.name), None)
        if folder is None:
            print(f"⚠️ Folder not found for: {label}")
            continue

        path = folder / "phase_defects.npz"
        if not path.exists():
            print(f"⚠️ phase_defects.npz missing for: {label}")
            continue

        data = np.load(path, allow_pickle=True)
        if "all_results" not in data or "phi_aligned_list" not in data:
            print(f"⚠️ Invalid phase_defects.npz in {folder}")
            continue

        all_results = list(data["all_results"])
        phi_list = list(data["phi_aligned_list"])

        pos_all = []
        neg_all = []
        T=0
        # === PROCESS EACH TIME WINDOW ===
        for i, r in enumerate(all_results):

            charges = np.asarray(r.get("charges", []))
            if charges.size == 0:
                continue

            pos = np.asarray(r.get("pos_coords", []))
            neg = np.asarray(r.get("neg_coords", []))
            phi_field = phi_list[i] if i < len(phi_list) else None
            if phi_field is None:
                continue

            # Remove close opposite-charge pairs
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

            if pos.size > 0:
                pos_all.append(pos[:, 1])
            if neg.size > 0:
                neg_all.append(neg[:, 1])
            
            T+= float(phi_field.shape[0]) * dt 

        # Combine
        if len(pos_all) == 0 and len(neg_all) == 0:
            print(f"⚠️ No defects for: {label}")
            continue

        pos_all = np.concatenate(pos_all) if pos_all else np.array([])
        neg_all = np.concatenate(neg_all) if neg_all else np.array([])
        all_defects = np.concatenate([pos_all, neg_all]) if pos_all.size or neg_all.size else np.array([])


        results[label] = (all_defects, T, cond_color, Ns_small, Ns_big)

    # === PLOTTING ===
    fig, axes = plt.subplots(len(results), 1, figsize=figsize, sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (label, (defects, T, cond_color, Ns_small, Ns_big)) in zip(axes, results.items()):

        bin_edges = np.linspace(0, 1, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Convert to relative position
        s_rel = defects / Ns_small

        # Histogram (rate)
        hist, _ = np.histogram(s_rel, bins=bin_edges)
        rate = hist / (T * bin_width)

        ax.bar(
            centers, rate,
            width=bin_width * 0.9,
            color=cond_color,
            alpha=0.7,
            label=label
        )
         # Vertical lines (scaled)
        ax.axvline(0.15, color="gray", linestyle="--", alpha=0.8)
        ax.axvline(0.70, color="gray", linestyle="--", alpha=0.8)
        # --- FONT SIZES ADDED HERE ---
        ax.set_ylabel("Defect rate density [1/s]", fontsize=14)
        #ax.set_title(label, fontsize=18)

        # Tick label font size
        ax.tick_params(axis="both", which="major", labelsize=14)

        # Log scale as required
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e3)
        ax.grid(alpha=0.3)

    # Shared x label
    axes[-1].set_xlabel("s", fontsize=16)

    plt.tight_layout()
    plt.show()

def main():
    csv_path = Path(OUTPUT_DATA_PHASESPACE_DIR) / "collected_results_phasespace.csv"
    df = load_data(csv_path)
    mu_a_crit_free,mu_a_crits=get_analytic_values(0.096,2,df['zeta'][0],df['mu'][0],df['beta'][0], num_modes_mu_a=2)
    
    colormap='jet'
    print(sorted(df["Nmotor"].unique()))
    print(sorted(df["mu_a"].unique()))
   
    df["log_p1_p50"] = np.log(df["p1"] / df["p50"])
    # df = df[df["Nmotor"]<np.infty]
    
    df["Nmotor_scaled"] = (df["Nmotor"])*200
    #df_removed = df[(df["log_p1_p50"] <= -1.5) | (df["p50"] <= 0.001)].copy()
    #df = df[(df["log_p1_p50"] > -1.5) & (df["p50"] > 0.001)]
    df = df[(((df["mu_a"] < 1000) & (df["log_p1_p50"] > -1.7) & (df["p50"] > 0.001)) | (df["mu_a"] >= 1000)) & (df["mu_a"] <= 2000) & (df["Nmotor"] >= 50)]
    #print sorted
    print(sorted(df["Nmotor"].unique()))
    print(sorted(df["mu_a"].unique()))

    csv_path = Path(OUTPUT_DATA_EXTRACTION_DIR) / "collected_results_extraction.csv"
    df_extraction = load_data(csv_path)
    df_extraction["Nmotor_scaled"] = (df_extraction["Nmotor"])*200
    
    df_new_parameter = pd.read_csv(OUTPUT_DATA_NEW_PARAMETER_DIR / "collected_results_new_parameter.csv")
    df_new_parameter["Nmotor_scaled"] = (df_new_parameter["Nmotor"])*200
    df_new_parameter["peak_freq_hz"] = 0.85*df_new_parameter["peak_freq_hz"]
    df_new_parameter["amplitude_band_10pct"] = df_new_parameter["amplitude_band_10pct"]

    transitions = detect_phase_transitions(df)
    print(transitions)
    usefull_lines=False
    plot_figure_2(
        df,
        transitions,
        N0s=np.array((85,500,np.infty))*200,
        mu_a_crits=mu_a_crits,
        cmap=colormap,
        usefull_lines=usefull_lines
    )    
    
    plot_figure_3(
        df,df_extraction,df_new_parameter,
        transitions,
        N0s_ext=np.array((85*200,100000)),
        mua0_ext=1570,
        mu_a_crits=mu_a_crits,
        cmap=colormap,
        all_black=False,
    )
    
    ##########INSET
    # --- to fig 2
    plot_inset_fig2_E_NQ(
        df,
        transitions,
        muas_for_N_scan=np.array((500,1000,1570)),
        mu_a_crits=mu_a_crits,
        all_black=False,
        usefull_lines=usefull_lines
    )

    # --- to fig 3
    plot_inset_fig3_F()
    
    ########SI
    plot_SI_global_noniso_combined()
    plot_SI_phase_defect_rates_vs_N()
    plot_defect_distributions_multiroot(
        conditions=[
            {
                "label": "WT ATP750 KCL0",
                "root": OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT,
                "folder": "genotype_WT_ATP_750_KCL_0",
                "dt": 0.001,
                'Ns': (20,24),
            },
            {
                "label": "WT ATP750 KCL400",
                "root": OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT,
                "folder": "genotype_WT_ATP_750_KCL_400",
                "dt": 0.001,
                'Ns': (20,24),
            },
            {
                "label": "Sim Nmotor_63 μ=270.529",
                "root": OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER,
                "folder": "Nmotor_63.0_mu_270.529",
                "dt": DT_SIMULATION / 0.45,
                'Ns': (90,100),
            }
        ]
    )




    ###########WN
    csv_path = Path(OUTPUT_DATA_WN_PHASESPACE_DIR) / "collected_results_wn_phasespace.csv"
    df_wn = load_data(csv_path)
    df_det = df[df["Nmotor"] == np.inf].copy()  
    df_wn = pd.concat([df_wn, df_det], ignore_index=True)
    print(sorted(df_wn["Nmotor"].unique()))
    print(sorted(df_wn["mu_a"].unique()))
   
    df_wn["log_p1_p50"] = np.log(df_wn["p1"] / df_wn["p50"])
    df_wn["Nmotor_scaled"] = (df_wn["Nmotor"])*200
    df_wn = df_wn[(((df_wn["mu_a"] < 1000) & (df_wn["log_p1_p50"] > -2.0) & (df_wn["p50"] > 0.0007)) | (df_wn["mu_a"] >= 1000)) & (df_wn["mu_a"] <= 2000) & (df_wn["Nmotor"] >= 50)]
    #print sorted
    print(sorted(df_wn["Nmotor"].unique()))
    print(sorted(df_wn["mu_a"].unique()))

    transitions = detect_phase_transitions(df_wn)
    print(transitions)
    usefull_lines=False
    plot_figure_2(
        df_wn,
        transitions,
        N0s=np.array((85,500,np.infty))*200,
        mu_a_crits=mu_a_crits,
        cmap=colormap,
        usefull_lines=usefull_lines,
        fig_subdir='fig2_wn'
    )    
    plot_inset_fig2_E_NQ(
        df_wn,
        transitions,
        muas_for_N_scan=np.array((500,1000,1570)),
        mu_a_crits=mu_a_crits,
        all_black=False,
        usefull_lines=usefull_lines,
        fig_subdir='fig2_wn'
    )

    ###########CWN
    csv_path = Path(OUTPUT_DATA_CWN_PHASESPACE_DIR) / "collected_results_cwn_phasespace.csv"
    df_wn = load_data(csv_path)
    df_det = df[df["Nmotor"] == np.inf].copy()  
    df_wn = pd.concat([df_wn, df_det], ignore_index=True)
    print(sorted(df_wn["Nmotor"].unique()))
    print(sorted(df_wn["mu_a"].unique()))
   
    df_wn["log_p1_p50"] = np.log(df_wn["p1"] / df_wn["p50"])
    df_wn["Nmotor_scaled"] = (df_wn["Nmotor"])*200
    df_wn = df_wn[(((df_wn["mu_a"] < 1000) & (df_wn["log_p1_p50"] > -2.0) & (df_wn["p50"] > 0.0007)) | (df_wn["mu_a"] >= 1000)) & (df_wn["mu_a"] <= 2000) & (df_wn["Nmotor"] >= 50)]
    #print sorted
    print(sorted(df_wn["Nmotor"].unique()))
    print(sorted(df_wn["mu_a"].unique()))

    transitions = detect_phase_transitions(df_wn)
    print(transitions)
    usefull_lines=False
    plot_figure_2(
        df_wn,
        transitions,
        N0s=np.array((85,500,np.infty))*200,
        mu_a_crits=mu_a_crits,
        cmap=colormap,
        usefull_lines=usefull_lines,
        fig_subdir='fig2_cwn'
    )    
    plot_inset_fig2_E_NQ(
        df_wn,
        transitions,
        muas_for_N_scan=np.array((500,1000,1570)),
        mu_a_crits=mu_a_crits,
        all_black=False,
        usefull_lines=usefull_lines,
        fig_subdir='fig2_cwn'
    )
if __name__ == "__main__":
    main()