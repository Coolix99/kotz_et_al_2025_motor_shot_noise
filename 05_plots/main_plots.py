import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ================================
# CONFIG
# ================================
F_RANGE = (0, 100)
A_RANGE = (0, 1.0)
LAM_RANGE = (1.0, 4.0)
Q_RANGE = (0.1, 1e4)

kcl_to_relN = {
    0: 1.00,
    50: 0.96,
    100: 0.93,
    200: 0.87,
    300: 0.80,
    400: 0.74
}

LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 11

plt.rcParams.update({
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

def prepare_experiment_df(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path)

    df.columns = [c.rstrip("_") for c in df.columns]
    # keep only ATP=750
    df = df[df["ATP_uM"] == 750].copy()
    df = df[df["sexp"].isin(["WT_KCl", "WT_ATP"])].copy()
    # map KCl → relN
    kcl_keys = np.array(sorted(kcl_to_relN.keys()))
    kcl_vals = df["KCl_mM"].to_numpy()

    nearest = kcl_keys[
        np.argmin(np.abs(kcl_vals[:, None] - kcl_keys[None, :]), axis=1)
    ]
    df["relN"] = np.vectorize(kcl_to_relN.get)(nearest)

    df["condition"] = df["sexp"]

    df["D"] = df["Q"]/df["f"]

    if "defect_rate_pos" in df.columns and "defect_rate_neg" in df.columns:
        df = df.copy()
        df["defect_rate_total"] = df["defect_rate_pos"] + df["defect_rate_neg"]

    print(f"prepare_experiment_df: {len(df)} entries")

    return df

def prepare_simulation_2d_phasespace_df(csv_path):
    df = pd.read_csv(csv_path)
    df.loc[df["Nmotor"] < 0, "Nmotor"] = np.inf
    # --- scaling ---
    df['Nmotor_scaled'] = df['Nmotor'] * 200
    df['f'] = df['f'] * 250

   
    # FILTER BAD DATA 
    before = len(df)

    df = df[
        (df["Nmotor_scaled"] >= 1e4) &
        (df["percentile_1"] >= 0.1)
    ].copy()

    after = len(df)

    print(f"Filtered simulations: {before} → {after} "
          f"({100 * after / before:.1f}% kept)")

    # PARAMETER CONSISTENCY CHECK
    expected = {
        "mu": 10.0,
        "eta": 0.096,
        "zeta": 0.96,
        "beta": 2.0,
    }

    for key, val in expected.items():
        if key not in df.columns:
            raise ValueError(f"Missing column '{key}' in dataframe")

        unique_vals = np.sort(df[key].unique())

        if not np.allclose(unique_vals, val):
            raise ValueError(
                f"Parameter '{key}' is not constant!\n"
                f"Expected: {val}\n"
                f"Found: {unique_vals}"
            )

    print("✅ Parameters are consistent:",
          ", ".join([f"{k}={v}" for k, v in expected.items()]))

    return df


def plot_hydro_boxplot(df, df_cass_hydro, x_max=0.15):
    """
    Boxplot: experiment (shifted left)
    + simulation mean: horizontal line + square marker (shifted right)
    """

    # -------------------------
    # EXPERIMENTAL DATA
    # -------------------------
    free   = df["hydro_free"].dropna().values
    chlamy = df["hydro_chlamy"].dropna().values
    fixed  = df["hydro_fixed"].dropna().values

    data = [free, chlamy, fixed]
    labels = ["Free", "Chlamy", "Fixed"]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))

    # geometry
    base_positions = np.array([1, 2, 3])
    offset = 0.2
    box_positions = base_positions - offset
    marker_positions = base_positions + offset
    box_width = 0.35

    # -------------------------
    # BOXPLOT (shifted left)
    # -------------------------
    ax.boxplot(
        data,
        positions=box_positions,
        patch_artist=True,
        showfliers=False,
        widths=box_width,
        boxprops=dict(facecolor="#d9d9d9", color="black", linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )

    # -------------------------
    # SIMULATION MEANS
    # -------------------------
    sim_free   = df_cass_hydro["mean_R_free"].mean()
    sim_chlamy = df_cass_hydro["mean_R_chlamy"].mean()
    sim_fixed  = df_cass_hydro["mean_R_fixed"].mean()

    sim_vals = [sim_free, sim_chlamy, sim_fixed]

    # horizontal line width = box width
    half_w = box_width / 2

    for x, y in zip(marker_positions, sim_vals):
        # horizontal line
        ax.plot(
            [x - half_w, x + half_w],
            [y, y],
            color="black",
            linewidth=2,
            zorder=5,
        )

        # square marker on top
        ax.scatter(
            x,
            y,
            marker="s",
            s=40,
            color="black",
            zorder=6,
        )

    # -------------------------
    # FORMATTING
    # -------------------------
    ax.set_ylabel(r"$\langle \frac{|\mathcal{H}|}{|\mathcal{H}|+|\mathcal{E}|} \rangle$")

    ax.set_xticks(base_positions)
    ax.set_xticklabels(labels, rotation=30)

    ax.set_ylim(0.0, x_max)

    step = 0.05
    yticks = np.arange(0, x_max + step, step)
    ax.set_yticks(yticks)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    # clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(0.5, 3.5)

    plt.tight_layout()
    plt.show()

def compute_boxplot_stats(vals):
    """
    Reproduce matplotlib boxplot statistics:
      - Q1, median, Q3
      - whiskers based on 1.5 * IQR
    """

    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return None

    q1 = np.percentile(vals, 25)
    median = np.percentile(vals, 50)
    q3 = np.percentile(vals, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    whisker_low = np.min(vals[vals >= lower_bound])
    whisker_high = np.max(vals[vals <= upper_bound])

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }

def print_hydro_boxplot_stats(df_exp, df_sim):
    keys = [
        ("hydro_free",   "Free"),
        ("hydro_chlamy", "Chlamy"),
        ("hydro_fixed",  "Fixed"),
    ]

    print("\n=== HYDRO BOXPLOT VALUES (matplotlib exact) ===\n")

    for key, label in keys:

        vals = df_exp[key].dropna().values
        stats = compute_boxplot_stats(vals)

        print(f"{label} (EXP):")

        if stats is None:
            print("  no data\n")
            continue

        print(f"  whisker_low  = {stats['whisker_low']:.4f}")
        print(f"  Q1           = {stats['q1']:.4f}")
        print(f"  median       = {stats['median']:.4f}")
        print(f"  Q3           = {stats['q3']:.4f}")
        print(f"  whisker_high = {stats['whisker_high']:.4f}")

        # simulation mean (your horizontal line)
        sim_key = f"mean_R_{key.split('_')[1]}"
        sim_vals = df_sim[sim_key].dropna().values

        if len(sim_vals) > 0:
            print(f"  SIM mean     = {np.mean(sim_vals):.4f}")
        else:
            print("  SIM mean     = nan")

        print("")

def plot_quantity(
    ax,
    df,
    y_key,
    ylabel,
    vmin=None,
    vmax=None,
    log_scale=False,
    scatter=False,
    df_sim_2d=None,
    df_sim_3d=None,
):
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_key, "relN"])

    # =====================================================
    # EXPERIMENT (RED)
    # =====================================================
    if scatter:
        ax.scatter(
            df["relN"],
            df[y_key],
            alpha=0.4,
            s=20,
            color="red",
        )

    agg = df.groupby("relN")[y_key].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    ax.errorbar(
        agg["relN"],
        agg["mean"],
        yerr=agg["sem"],
        fmt="o",
        color="red",
        capsize=4,
        linewidth=2,
        markersize=6,
        label="experiment"
    )

    # =====================================================
    # generic simulation plotting helper
    # =====================================================
    def _plot_sim(df_s, color, label):
        if df_s is None or y_key not in df_s.columns:
            return

        df_s = df_s.copy()
        df_s = df_s.replace([np.inf, -np.inf], np.nan)

        # if "relN" not in df_s.columns:
        #     if "Nmotor" in df_s.columns:
        #         df_s["relN"] = df_s["Nmotor"] / 20.0
        #     else:
        #         return

        df_s = df_s.dropna(subset=[y_key, "relN"])
        if len(df_s) == 0:
            return

       
        agg_s = (
            df_s.groupby("relN")
            .agg({
                y_key: "mean",
                "lambda": "mean" if "lambda" in df_s.columns else "first",
            })
            .reset_index()
            .sort_values("relN")
        )

        x_vals = agg_s["relN"].values
        y_vals = agg_s[y_key].values

        if "lambda" in agg_s.columns:
            lam_vals = agg_s["lambda"].values
        else:
            lam_vals = np.full_like(x_vals, np.nan, dtype=float)

        threshold = 2.2

        for i in range(len(x_vals) - 1):
            x_seg = x_vals[i:i+2]
            y_seg = y_vals[i:i+2]

            if np.isfinite(lam_vals[i]) and np.isfinite(lam_vals[i+1]):
                linestyle = ":" if (lam_vals[i] > threshold or lam_vals[i+1] > threshold) else "-"
            else:
                linestyle = "-"

            ax.plot(
                x_seg,
                y_seg,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )

        ax.scatter(
            x_vals,
            y_vals,
            color=color,
            s=40,
            marker="o",
            label=label
        )

    # =====================================================
    # SIMULATIONS
    # =====================================================
    # _plot_sim(df_sim_2d, color="#1f77b4", label="simulation 2D")
    # _plot_sim(df_sim_3d, color="green", label="simulation 3D")
    _plot_sim(df_sim_2d, color="#adadad", label="Ncilium")
    _plot_sim(df_sim_3d, color="#525252", label="5Ncilium")

    # =====================================================
    # AXES
    # =====================================================
    ax.set_xlabel(r"$N_\text{remain}/N$", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.set_xlim(0.5, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)

    if log_scale:
        ax.set_yscale("log")

    if ylabel.startswith("Frequency"):
        ax.legend(fontsize=LEGEND_SIZE)

def plot_figure_3(df, df_sim_2d=None, df_sim_3d=None):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    quantities = [
        ("f", r"Frequency $f_0$ [Hz]", *F_RANGE, False),
        ("amplitude", r"Amplitude $A$ [rad]", *A_RANGE, False),
        ("lambda", r"Wavelength $\lambda/L$", *LAM_RANGE, False),
        ("Q", r"Quality factor Q", *Q_RANGE, True),
    ]

    for i, (key, label, vmin, vmax, log_scale) in enumerate(quantities):
        ax = axes[i // 2, i % 2]

        plot_quantity(
            ax,
            df,
            key,
            label,
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale,
            df_sim_2d=df_sim_2d,
            df_sim_3d=df_sim_3d,
        )

    axes[2, 0].axis("off")
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_phase_space_ax(
    ax,
    df,
    value_key,
    vmin,
    vmax,
    label,
    cmap="jet",
    log_scale=False,
    debug_scatter=False,
):
    grouped = (
        df.groupby(["mu_a", "Nmotor_scaled"])[value_key]
        .mean()
        .reset_index()
    )
    grouped["invN"] = 1e4 / grouped["Nmotor_scaled"]

    x = grouped["mu_a"].values
    y = grouped["invN"].values
    z = grouped[value_key].values

    # grid
    xi = np.linspace(0, x.max(), 120)
    yi = np.linspace(0, y.max(), 120)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), z, (Xi, Yi), method="nearest")

    
    # GREY MASK
    y_data = np.sort(np.unique(y))
    mu_min_y = {
        yv: grouped.loc[np.isclose(grouped["invN"], yv), "mu_a"].min()
        for yv in y_data
    }

    mu_min_interp = np.zeros_like(yi)
    for j, yv in enumerate(yi):
        idx = np.argmin(np.abs(y_data - yv))
        mu_min_interp[j] = mu_min_y[y_data[idx]]
    mu_min_interp = np.minimum.accumulate(mu_min_interp[::-1])[::-1]
    grey_mask = Xi < (mu_min_interp[:, None] + 50)  # small buffer

    

    # -------------------------------------------------
    # main plot
    # -------------------------------------------------
    Zi_masked = np.ma.array(Zi, mask=np.isnan(Zi))

    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(Xi, Yi, Zi_masked, cmap=cmap, norm=norm, shading="auto")
    else:
        pcm = ax.pcolormesh(Xi, Yi, Zi_masked, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="auto")

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(label, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    ax.set_xlabel(r"$\mu_a$")
    ax.set_ylabel(r"$10^4 / N_{\mathrm{motor}}$")

    if debug_scatter:
        ax.scatter(
            x,
            y,
            s=10,
            color="black",
            alpha=0.5,
            label="data"
        )
    # plot grey background
    ax.pcolormesh(
        Xi, Yi,
        0.3*np.where(grey_mask, 1, np.nan),
        cmap="Greys",
        shading="auto",
        vmin=0, vmax=1,
        alpha=1.0,
        #zorder=10,
    )
    return pcm

def plot_figure_2(
    df,
    transitions,
    N0s,
    mu_a_crits=None,
    cmap="jet",
):
    quantities = [
        ("f", r"Frequency $f_0$ [Hz]", 0, 100, False),
        ("amplitude", r"Amplitude $A$ [rad]", 0, 1.0, False),
        ("lambda", r"Wavelength $\lambda/L$", 1.0, 4.0, False),
        ("Q", r"Quality factor $Q$", 0.1, 1e4, True),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(9, 16))

    # -----------------------------------------------------
    # compute global transition fit
    # -----------------------------------------------------
    qs = ["f", "amplitude", "log10_Q"]
    xs, ys = [], []
    m, n = None, None
    if transitions is not None:
        for q in qs:
            pts = transitions.dropna(subset=[q, "Nmotor_scaled"]).copy()
            pts["invN"] = 1e4 / pts["Nmotor_scaled"]

            xs.extend(pts[q].values)
            ys.extend(pts["invN"].values)

        if len(xs) > 0:
            m, n = np.polyfit(xs, ys, 1)

    # styles
    colors = ["#b8b8b8", "#5b5c5b", "#000000"]
    markers = ["o", "s", "^"]

    # special setup for Q right panel
    mu_a_qvals = [500, 1000, 1570]
    q_colors = {
        500: colors[0],
        1000: colors[1],
        1570: colors[2],
    }

    # -----------------------------------------------------
    # LOOP OVER QUANTITIES
    # -----------------------------------------------------
    for row, (q, label, vmin, vmax, log_scale) in enumerate(quantities):
        ax_phase = axes[row, 0]
        ax_right = axes[row, 1]

        # ---------------- LEFT: PHASE SPACE ----------------
        plot_phase_space_ax(
            ax_phase,
            df,
            q,
            vmin,
            vmax,
            label,
            cmap=cmap,
            log_scale=log_scale,
        )

        ax_phase.set_xlabel(r"Motor activity $\mu_a$", fontsize=LABEL_SIZE)
        ax_phase.set_ylabel(r"Motor number $N$", fontsize=LABEL_SIZE)
        ax_phase.tick_params(axis="both", labelsize=TICK_SIZE)
        ax_phase.set_xlim(0, 2000)

        yticks = [0.0, 0.1, 0.5, 1.0]
        yticklabels = [
            r"$\infty$",
            r"$10^5$",
            r"$2\cdot 10^4$",
            r"$10^4$",
        ]
        ax_phase.set_yticks(yticks)
        ax_phase.set_yticklabels(yticklabels)
        ax_phase.set_box_aspect(1)

        # critical line: only first one
        if mu_a_crits is not None and len(mu_a_crits) > 0:
            ax_phase.axvline(mu_a_crits[0], linestyle=":", color="black")
            ax_right.axvline(mu_a_crits[0], linestyle=":", color="black")

        # global fit line
        if m is not None:
            xi = np.linspace(0, 2000, 200)
            ax_phase.plot(xi, m * xi + n, "k--", linewidth=2)

        ax_phase.set_ylim((0, 1.0))
        ax_phase.set_xlim(0, 2000)

        # ---------------- RIGHT: STANDARD PANELS ----------------
        if q != "Q":
            for i, N0 in enumerate(N0s):
                color = colors[i % len(colors)]
                sub = (
                    df[df["Nmotor_scaled"] == N0]
                    .groupby("mu_a")[q]
                    .mean()
                    .reset_index()
                )

                marker = markers[i % len(markers)]

                # line
                ax_right.plot(
                    sub["mu_a"],
                    sub[q],
                    linestyle="-",
                    color=color,
                )

                # markers (explicit scatter for full control)
                ax_right.scatter(
                    sub["mu_a"],
                    sub[q],
                    marker=marker,
                    s=50,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                    label=f"N={N0:.0f}" if np.isfinite(N0) else r"N=$\infty$",
                )

                y0 = 1e4 / N0
                if m is not None and np.isfinite(y0):
                    mu_trans = (y0 - n) / m
                    ax_right.axvline(mu_trans, linestyle="--", color="black")
                ax_phase.axhline(y0, linewidth=1.5, color=color)

            ax_right.set_xlabel(r"Motor activity $\mu_a$", fontsize=LABEL_SIZE)
            ax_right.set_ylabel("")
            ax_right.tick_params(axis="both", labelsize=TICK_SIZE)
            ax_right.set_xlim(0, 2000)
            ax_right.set_ylim(vmin, vmax)

            if log_scale:
                ax_right.set_yscale("log")

            if row == 0:
                ax_right.legend(fontsize=LEGEND_SIZE)

        # ---------------- RIGHT: SPECIAL Q PANEL ----------------
        else:
            # left panel: vertical lines at selected mu_a instead of horizontal N lines
            for mu_val in mu_a_qvals:
                ax_phase.axvline(mu_val, linewidth=1.5, color=q_colors[mu_val])

            # right panel: Q vs motor number (log-log), excluding N = inf
            df_q = df[np.isfinite(df["Nmotor_scaled"])].copy()

            for mu_val in mu_a_qvals:
                sub = df_q[np.isclose(df_q["mu_a"], mu_val)].copy()
                if len(sub) == 0:
                    continue

                sub = (
                    sub.groupby("Nmotor_scaled")["Q"]
                    .mean()
                    .reset_index()
                    .sort_values("Nmotor_scaled")
                )

                if len(sub) == 0:
                    continue

                marker_map = {
                    500: "o",
                    1000: "s",
                    1570: "^",
                }

                color = q_colors[mu_val]
                marker = marker_map[mu_val]

                # line
                ax_right.plot(
                    sub["Nmotor_scaled"].values,
                    sub["Q"].values,
                    linestyle="-",
                    color=color,
                )

                # markers
                ax_right.scatter(
                    sub["Nmotor_scaled"].values,
                    sub["Q"].values,
                    marker=marker,
                    s=50,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                    label=rf"$\mu_a={mu_val}$",
                )

            # axes
            ax_right.set_xlabel(r"Motor number $N$", fontsize=LABEL_SIZE)
            ax_right.set_ylabel("")
            ax_right.tick_params(axis="both", labelsize=TICK_SIZE)

            ax_right.set_xscale("log")   # <-- log scale instead
            ax_right.set_yscale("log")

            ax_right.set_xlim(1e4, 1e6)  # <-- requested bounds
            ax_right.set_ylim(vmin, vmax)

            ax_right.legend(fontsize=LEGEND_SIZE)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.24)
            plt.show()

def plot_fig3F_inset(
    df_exp,
    df_sim=None,
    relN_values=None,
    fontsize_labels=14,
    fontsize_ticks=12,
    figsize=(5, 4),
    tol=1e-2,
):
    """
    Fig. 3F inset: defect rates (+1 / -1) at selected N/N0.
    Supports experiment + simulation.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if relN_values is None:
        relN_values = [1.0, 0.74]

    # -------------------------------------------------
    # Ensure relN exists for simulation
    # -------------------------------------------------
    if df_sim is not None:
        df_sim = df_sim.copy()

        if "relN" not in df_sim.columns:
            if "Nmotor" in df_sim.columns:
                df_sim["relN"] = df_sim["Nmotor"] / 20.0
            else:
                print("⚠️ Simulation: cannot construct relN → disabling simulation")
                df_sim = None

    # -------------------------------------------------
    # Check required columns
    # -------------------------------------------------
    required = ["defect_rate_pos", "defect_rate_neg"]

    def check_df(df, name):
        missing = [k for k in required if k not in df.columns]
        if missing:
            print(f"⚠️ {name} missing columns: {missing}")
            return False
        return True

    if not check_df(df_exp, "Experiment"):
        raise ValueError("Experiment data missing required defect columns")

    if df_sim is not None and not check_df(df_sim, "Simulation"):
        df_sim = None

    # -------------------------------------------------
    # Aggregation helper
    # -------------------------------------------------
    def aggregate(df, relN_targets):
        records = []

        for r in relN_targets:
            sub = df[np.isclose(df["relN"], r, atol=tol)]
            
            if sub.empty:
                continue

            rec = {"relN": r}

            for key in required:
                vals = sub[key].dropna().values
                print(vals)
                if len(vals) > 0:
                    rec[f"{key}_mean"] = np.mean(vals)
                    rec[f"{key}_sem"] = np.std(vals) / np.sqrt(len(vals))
                else:
                    rec[f"{key}_mean"] = np.nan
                    rec[f"{key}_sem"] = np.nan

            records.append(rec)

        return pd.DataFrame(records)

    # -------------------------------------------------
    # Aggregate data
    # -------------------------------------------------
    exp_agg = aggregate(df_exp, relN_values)

    sim_agg = None
    if df_sim is not None and not df_sim.empty:
        sim_agg = aggregate(df_sim, relN_values)

    if exp_agg.empty:
        print("⚠️ No experimental data available for inset")
        return

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.35
    x = np.arange(len(exp_agg))

    # --- EXPERIMENT (red) ---
    ax.bar(
        x - width / 2,
        exp_agg["defect_rate_pos_mean"],
        width,
        yerr=exp_agg["defect_rate_pos_sem"],
        color="red",
        alpha=0.7,
        hatch="///",
        edgecolor="black",
        capsize=4,
        label="+1 (exp)",
    )

    ax.bar(
        x + width / 2,
        exp_agg["defect_rate_neg_mean"],
        width,
        yerr=exp_agg["defect_rate_neg_sem"],
        color="red",
        alpha=0.7,
        hatch="...",
        edgecolor="black",
        capsize=4,
        label="−1 (exp)",
    )

    # --- SIMULATION (blue) ---
    if sim_agg is not None and not sim_agg.empty:
        print(sim_agg["defect_rate_pos_mean"])
        print(sim_agg["defect_rate_neg_mean"])
        ax.bar(
            x - width / 2,
            sim_agg["defect_rate_pos_mean"],
            width,
            yerr=sim_agg["defect_rate_pos_sem"],
            color="#1f77b4",
            alpha=0.5,
            edgecolor="black",
            capsize=4,
            label="+1 (sim)",
        )

        ax.bar(
            x + width / 2,
            sim_agg["defect_rate_neg_mean"],
            width,
            yerr=sim_agg["defect_rate_neg_sem"],
            color="#1f77b4",
            alpha=0.5,
            edgecolor="black",
            capsize=4,
            label="−1 (sim)",
        )

    # -------------------------------------------------
    # Labels
    # -------------------------------------------------
    xticklabels = [f"{int(r * 100)}%" for r in exp_agg["relN"]]

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel(r"$N/N_0$", fontsize=fontsize_labels)
    ax.set_ylabel(r"Defect rate [s$^{-1}$]", fontsize=fontsize_labels)

    ax.set_yscale("log")

    ax.tick_params(axis="both", labelsize=fontsize_ticks)

    ax.legend(fontsize=fontsize_ticks)

    plt.tight_layout()
    plt.show()

def compute_mu_a_crits(df):
    eta = df["eta"].iloc[0]
    zeta = df["zeta"].iloc[0]
    mu = df["mu"].iloc[0]
    beta = df["beta"].iloc[0]
    f_stern = 2.0 

    n0 = eta / (eta + (1 - eta) * np.exp(f_stern))
    lam = eta + (1 - eta) * np.exp(f_stern)
    kappa = (1 - eta) * n0 * np.exp(f_stern) * f_stern * zeta

    q_vals = [(np.pi/2 + n*np.pi)**2 for n in range(5)]

    mu_a_crits = [
        (mu + q + lam*beta) / (2*kappa - 2*n0*lam*zeta)
        for q in q_vals
    ]

    return mu_a_crits

def detect_phase_transitions(df):
    df = df.copy()
    df["log10_Q"] = np.log10(df["Q"])
    records = []
    for N, group in df.groupby("Nmotor_scaled"):
        agg = (
            group.groupby("mu_a")[["amplitude", "log10_Q", "lambda", "f"]]
            .mean()
            .reset_index()
            .sort_values("mu_a")
        )
        x = agg["mu_a"].values
        rec = {"Nmotor_scaled": N}
        # amplitude local minimum
        y = agg["amplitude"].values
        if len(y) > 5:
            # invert to find minima as peaks
            peaks, props = find_peaks(-y, prominence=0.01)

            if len(peaks) > 0:
                # choose most prominent minimum
                best = peaks[np.argmax(props["prominences"])]
                rec["amplitude"] = x[best]
            else:
                rec["amplitude"] = np.nan
        else:
            rec["amplitude"] = np.nan
        
        
        # Q minimum
        mask = x > 300
        if np.any(mask):
            x_sel = x[mask]
            y_sel = y[mask]

            # find local minima via peaks in -y
            peaks, props = find_peaks(-y_sel, prominence=0.01)

            if len(peaks) > 0:
                # pick most prominent minimum
                best = peaks[np.argmax(props["prominences"])]
                rec["log10_Q"] = x_sel[best]
            else:
                rec["log10_Q"] = np.nan
        else:
            rec["log10_Q"] = np.nan

        # frequency slope
        if N>1e4 and min(x)<250:
            y = agg["f"].values
            slopes = np.gradient(y, x)
            rec["f"] = x[np.argmax(slopes)]
            
            records.append(rec)

    return pd.DataFrame(records)

def compute_sbi_targets(df_exp, relN_norm=1.0, relN_pert=0.74, tol=1e-3):
    import numpy as np

    def extract_stats(df, key, relN):
        sub = df[np.isclose(df["relN"], relN, atol=tol)][key].dropna()

        if len(sub) == 0:
            raise ValueError(f"No data for relN={relN}, key={key}")

        mean = np.mean(sub)
        sem = np.std(sub) / np.sqrt(len(sub))

        return mean, sem

    # ---------------------------
    # ABSOLUTE VALUES
    # ---------------------------
    f_norm, f_norm_sem = extract_stats(df_exp, "f", relN_norm)
    f_pert, f_pert_sem = extract_stats(df_exp, "f", relN_pert)

    A_norm, A_norm_sem = extract_stats(df_exp, "amplitude", relN_norm)
    A_pert, A_pert_sem = extract_stats(df_exp, "amplitude", relN_pert)

    lam_norm, lam_norm_sem = extract_stats(df_exp, "lambda", relN_norm)
    lam_pert, lam_pert_sem = extract_stats(df_exp, "lambda", relN_pert)
    
    Q_norm, Q_norm_sem = extract_stats(df_exp, "Q", relN_norm)
    Q_pert, Q_pert_sem = extract_stats(df_exp, "Q", relN_pert)


    # ---------------------------
    # RATIOS (unchanged)
    # ---------------------------
    def ratio_sem(m1, s1, m2, s2):
        return abs(m1 / m2) * np.sqrt((s1 / m1)**2 + (s2 / m2)**2)

    def log10_sem_from_linear(mean, sem):
        if not np.isfinite(mean) or mean <= 0:
            return np.nan, np.nan
        log_mean = np.log10(mean)
        log_sem = sem / (mean * np.log(10))
        return log_mean, log_sem

    freq_ratio = f_pert / f_norm
    freq_ratio_sem = ratio_sem(f_pert, f_pert_sem, f_norm, f_norm_sem)

    amp_ratio = A_pert / A_norm
    amp_ratio_sem = ratio_sem(A_pert, A_pert_sem, A_norm, A_norm_sem)

    lam_ratio = lam_pert / lam_norm
    lam_ratio_sem = ratio_sem(lam_pert, lam_pert_sem, lam_norm, lam_norm_sem)

    # ---------------------------
    # GEOMETRIC MEAN + ERROR
    # ---------------------------
    def geom_mean_with_sem(x1, s1, x2, s2):
        g = np.sqrt(x1 * x2)

        rel_err = 0.5 * np.sqrt((s1 / x1)**2 + (s2 / x2)**2)
        s_g = g * rel_err

        return g, s_g

    f_geom, f_geom_sem = geom_mean_with_sem(f_norm, f_norm_sem, f_pert, f_pert_sem)
    A_geom, A_geom_sem = geom_mean_with_sem(A_norm, A_norm_sem, A_pert, A_pert_sem)

    logQ_norm, logQ_norm_sem = log10_sem_from_linear(Q_norm, Q_norm_sem)
    logQ_pert, logQ_pert_sem = log10_sem_from_linear(Q_pert, Q_pert_sem)

    # ---------------------------
    # PRINT CONFIG
    # ---------------------------
    print("\n=== SBI TARGETS ===\n")

    print("LIKELIHOOD_1_CONFIG = {")
    print("    \"wavelength\": {")
    print(f"        \"norm\": {lam_norm:.3f},")
    print(f"        \"pert\": {lam_pert:.3f},")
    print(f"        \"sigma\": {lam_ratio_sem:.3f},")
    print("    },")
    print("    \"amplitude_ratio\": {")
    print(f"        \"expected\": {amp_ratio:.3f},")
    print(f"        \"sigma\": {amp_ratio_sem:.3f},")
    print("    },")
    print("    \"frequency_ratio\": {")
    print(f"        \"expected\": {freq_ratio:.3f},")
    print(f"        \"sigma\": {freq_ratio_sem:.3f},")
    print("    },")
    print("}")

    print("\nLIKELIHOOD_2_CONFIG = {")
    print("    \"frequency\": {")
    print(f"        \"mean\": {f_geom:.2f},")
    print(f"        \"sigma\": {f_geom_sem:.2f},")
    print("    },")
    print("    \"amplitude\": {")
    print(f"        \"mean\": {A_geom:.3f},")
    print(f"        \"sigma\": {A_geom_sem:.3f},")
    print("    },")
    print("}")

    print("\nLIKELIHOOD_1_Q_CONFIG = {")
    print("    \"full\": {")
    print(f"        \"mean\": {logQ_norm:.3f},")
    print(f"        \"sigma\": {logQ_norm_sem:.3f},")
    print("    },")
    print("    \"reduced\": {")
    print(f"        \"mean\": {logQ_pert:.3f},")
    print(f"        \"sigma\": {logQ_pert_sem:.3f},")
    print("    },")
    print("}")

def plot_phase_diagnostics(df, cmap="viridis"):
    """
    Debug phase-space plots for:
        - percentile_1 (log scale)
        - variance_explained (linear)

    Same style as Fig. 2.
    """

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["percentile_1", "variance_explained", "mu_a", "Nmotor_scaled"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -------------------------
    # Percentile (LOG SCALE)
    # -------------------------
    key = "percentile_1"

    # avoid log(0)
    df_pos = df[df[key] > 0].copy()

    vmin = np.nanpercentile(df_pos[key], 5)
    vmax = np.nanpercentile(df_pos[key], 95)

    plot_phase_space_ax(
        axes[0],
        df_pos,
        key,
        vmin,
        vmax,
        r"$\mathrm{percentile}_1$ (log)",
        cmap=cmap,
        log_scale=True,   # 🔥 key change
    )

    # -------------------------
    # Variance explained (linear)
    # -------------------------
    key = "variance_explained"

    vmin = np.nanpercentile(df[key], 5)
    vmax = np.nanpercentile(df[key], 95)

    plot_phase_space_ax(
        axes[1],
        df,
        key,
        vmin,
        vmax,
        r"Variance explained (mode 1)",
        cmap=cmap,
        log_scale=False,
    )

    plt.tight_layout()
    plt.show()

def plot_SI_R2_limitcycle(
    df,
    matrix_size=3,
    vmin=0.5,
    vmax=1.0,
):
    """
    SI figure: mean R² matrix + full profile R².

    Uses df_exp directly (already processed dataframe).
    """

    # -----------------------------------
    # BUILD MEAN MATRIX
    # -----------------------------------
    M = np.full((matrix_size, matrix_size), np.nan, dtype=float)

    for i in range(matrix_size):
        for j in range(matrix_size):
            col = f"r2_{i}_{j}"
            if col in df.columns:
                vals = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
                M[i, j] = np.nanmean(vals)

    # -----------------------------------
    # FULL R²
    # -----------------------------------
    if "r2_full_profiles" not in df.columns:
        raise ValueError("Missing column 'r2_full_profiles'")

    r2_full = (
        df["r2_full_profiles"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )

    r2_mean = float(np.mean(r2_full)) if len(r2_full) else np.nan

    # -----------------------------------
    # COLORMAP
    # -----------------------------------
    cmap = LinearSegmentedColormap.from_list(
        "white_to_darkgreen",
        ["#ffffff", "#006400"]
    )

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=(7.5, 3.8),
        gridspec_kw={"width_ratios": [1.3, 0.5]}
    )

    # -----------------------------------
    # MATRIX HEATMAP
    # -----------------------------------
    ax0.set_box_aspect(1)

    im = ax0.imshow(
        M,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        aspect="equal"
    )

    ax0.set_xlabel("Phase modes", fontsize=13)
    ax0.set_ylabel("Amplitude modes", fontsize=13)

    ax0.set_xticks(range(matrix_size))
    ax0.set_yticks(range(matrix_size))

    ax0.set_xticklabels([f"{j+1}" for j in range(matrix_size)], fontsize=12)
    ax0.set_yticklabels([f"{i+1}" for i in range(matrix_size)], fontsize=12)

    # annotate values
    for i in range(matrix_size):
        for j in range(matrix_size):
            val = M[i, j]

            if not np.isfinite(val):
                txt = "nan"
                color = "black"
            else:
                txt = f"{val:.3f}"
                color = "white" if val > 0.6 else "black"

            ax0.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=10,
                color=color
            )

    # colorbar
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("R²", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # -----------------------------------
    # FULL R² PANEL
    # -----------------------------------
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    ax1.set_box_aspect(1)

    if np.isfinite(r2_mean):
        color = cmap((r2_mean - vmin) / (vmax - vmin))
    else:
        color = "white"

    rect = plt.Rectangle((0.25, 0.25), 0.5, 0.5, color=color) # type: ignore
    ax1.add_patch(rect)

    ax1.text(
        0.5, 0.5,
        f"{r2_mean:.3f}" if np.isfinite(r2_mean) else "nan",
        ha="center",
        va="center",
        fontsize=16,
        color=("white" if r2_mean > 0.6 else "black"),
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # -----------------------------------
    # TITLE
    # -----------------------------------
    fig.suptitle("SI: Limit-cycle reconstruction quality (Geyer fit)", fontsize=14)

    plt.tight_layout()
    plt.show()

def get_hpd_levels(H, probs = [0.5, 0.95, 0.99]):
    H_flat = H.flatten()
    H_sorted = np.sort(H_flat)[::-1]

    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]

    levels = []
    for p in probs:
        idx = np.searchsorted(cumsum, p)
        levels.append(H_sorted[idx])

    return levels

def plot_weighted_density_F0_v0(
    df_2d,
    df_3d,
    F0_key="F0",
    v0_key="v0",
    bins=30,
):

    def get_samples(df):
        F0 = df[F0_key].values
        v0 = df[v0_key].values

        mask = np.isfinite(F0) & np.isfinite(v0)
        return F0[mask], v0[mask]

    F0_2d, v0_2d = get_samples(df_2d)
    F0_3d, v0_3d = get_samples(df_3d)


    # histogram (posterior density)
    def compute_density(F0, v0):
        mask = (F0 > 0) & (v0 > 0)
        F0 = F0[mask]
        v0 = v0[mask]

        logF0 = np.log10(F0)
        logv0 = np.log10(v0)

        F0_min, F0_max = np.percentile(logF0, [1, 99])
        v0_min, v0_max = np.percentile(logv0, [1, 99])

        H, xedges, yedges = np.histogram2d(
            logF0,
            logv0,
            bins=bins,
            range=[[F0_min*0.8, F0_max*1.2], [v0_min*0.8, v0_max*1.2]],
            density=True,
        )

        H = gaussian_filter(H, sigma=0.75)
        H /= np.max(H) # type: ignore

        # convert grid back to linear space
        Xc = 10 ** (0.5 * (xedges[:-1] + xedges[1:]))
        Yc = 10 ** (0.5 * (yedges[:-1] + yedges[1:]))

        X, Y = np.meshgrid(Xc, Yc, indexing="ij")

        return H, X, Y
    
    H2d, X_2d, Y_2d = compute_density(F0_2d, v0_2d)
    H3d, X_3d, Y_3d = compute_density(F0_3d, v0_3d)


    # PRIOR 
    n_prior = 400
    x_prior = np.logspace(np.log10(0.7), np.log10(200), n_prior)
    y_prior = np.logspace(np.log10(0.7), np.log10(100), n_prior)
    Xp, Yp = np.meshgrid(x_prior, y_prior, indexing="ij")

    def log10_gaussian(x, mean_log10, sigma_log10):
        lx = np.log10(x)
        return np.exp(-0.5 * ((lx - mean_log10) / sigma_log10) ** 2)

    prior = (
        log10_gaussian(Xp, np.log10(3.0), 0.25)
        * log10_gaussian(Yp, np.log10(5.0), 0.4)
    )
    prior /= np.max(prior)


    levels_2d = get_hpd_levels(H2d, probs=[0.5, 0.95, 0.99])
    levels_3d = get_hpd_levels(H3d, probs=[0.5, 0.95, 0.99])

    # plot
    plt.figure(figsize=(5, 4))

    # PRIOR background
    # plt.pcolormesh(
    #     Xp, Yp, prior,
    #     cmap="Greys",
    #     shading="auto",
    #     alpha=1.0,
    # )

    # PRIOR contours
    sigma_levels = [3, 2, 1]
    levels_prior = [np.exp(-0.5 * s**2) for s in sigma_levels]
    levels_prior_sorted = sorted(levels_prior)
    print(levels_prior_sorted)
    levels_prior = get_hpd_levels(prior, probs=[0.5, 0.95, 0.99])
    levels_prior_sorted = sorted(levels_prior)
    print(levels_prior_sorted)
    plt.contourf(
        Xp, Yp, prior,
        levels=levels_prior_sorted + [1.0],  # include top level
        cmap="Greys",
        alpha=0.7,
    )
    # cs_prior = plt.contour(
    #     Xp, Yp, prior,
    #     levels=levels_prior,
    #     colors="black",
    #     linewidths=1.5,
    # )
    # fmt_prior = {lvl: f"{s}σ" for lvl, s in zip(levels_prior, sigma_levels)}
    # plt.clabel(cs_prior, fmt=fmt_prior, inline=True, fontsize=11)

    # POSTERIOR contours (2D vs 3D)
    def plot_post(H, X, Y, levels, color, label):
        levels = sorted(levels)
        cs = plt.contour(
            X, Y, H,
            levels=levels,
            colors=color,
            linewidths=2,
            alpha=0.7
        )

        # fmt = {lvl: f"{s}σ" for lvl, s in zip(levels, sigma_levels)}
        # plt.clabel(cs, fmt=fmt, inline=True, fontsize=11)

        # cleaner legend (no deprecated API)
        plt.plot([], [], color=color, label=label)

    plot_post(H2d, X_2d, Y_2d, levels_2d, "#1f77b4", "2D")
    plot_post(H3d, X_3d, Y_3d, levels_3d, "#369b36", "3D")

    # ---------------------------------------
    # REFERENCE POINTS
    # ---------------------------------------

    # 2D (blue)
    plt.scatter(
        5.4,
        51.3,
        color="#1f77b4",
        s=70,
        marker="o",
        # edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="2D best fit",
    )

    # 3D (green)
    plt.scatter(
        25.6,
        16.8,
        color="#369b36",
        s=70,
        marker="o",
        # edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="3D best fit",
    )
    # cass
    plt.scatter(
        65.9,
        52.1,
        color="#000000",
        s=70,
        marker="o",
        # edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="3D best fit",
    )



    # ---------------------------------------
    # DEBUG SCATTER
    # ---------------------------------------
    # plt.scatter(
    #     F0_2d,
    #     v0_2d,
    #     s=5,
    #     alpha=0.2,
    #     color="blue",
    #     label="2D samples",
    # )

    # plt.scatter(
    #     F0_3d,
    #     v0_3d,
    #     s=5,
    #     alpha=0.2,
    #     color="green",
    #     label="3D samples",
    # )

    # axes
    plt.xlabel(r"$F_0$ [pN]", fontsize=LABEL_SIZE)
    plt.ylabel(r"$v_0$ [$\mu$m/s]", fontsize=LABEL_SIZE)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(1.0, 100)
    plt.ylim(1.0, 100)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    # plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def corner_plot_parameters(
    df_2d=None,
    df_3d=None,
    params=("K_d2","B","F0","Fc","v0","v0_eps0","pi0","b"),
    bins=40,
    sigma_smooth=1.0,
    plot_cloud=True,
    plot_contours=True,
    use_2d=True,
    use_3d=True,
):
    param_info = {
        "B": (r"$B$", (500, 1500)),
        "F0": (r"$F_0$", (1, 100)),
        "Fc": (r"$F_c$", (1, 20)),
        "v0": (r"$v_0$", (10, 100)),
        "v0_eps0": (r"$v_0/\epsilon_0$", (0.1, 1)),
        "pi0": (r"$\pi_0$", (10, 1000)),
        "b": (r"$b$", (0.01, 5)),
        "K_d2": (r"$a^2K$", (70, 90)),  
    }
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    n = len(params)

    fig, axes = plt.subplots(n, n, figsize=(2.5*n, 2.5*n))

    # helper
    def get_samples(df, p1, p2):
        x = df[p1].values
        y = df[p2].values
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return x[mask], y[mask]

    def compute_density(x, y):
        lx = np.log10(x)
        ly = np.log10(y)

        xmin, xmax = np.percentile(lx, [2, 98])
        ymin, ymax = np.percentile(ly, [2, 98])
        dx = xmax - xmin
        dy = ymax - ymin

        xmin_r = xmin - 0.8 * dx
        xmax_r = xmax + 0.8 * dx

        ymin_r = ymin - 0.8 * dy
        ymax_r = ymax + 0.8 * dy
        H, xedges, yedges = np.histogram2d(
            lx, ly,
            bins=bins,
            range=[[xmin_r, xmax_r], [ymin_r, ymax_r]],
            density=True
        )

        H = gaussian_filter(H, sigma=sigma_smooth)
        H /= np.max(H) # type: ignore

        Xc = 10 ** (0.5 * (xedges[:-1] + xedges[1:]))
        Yc = 10 ** (0.5 * (yedges[:-1] + yedges[1:]))

        return H, Xc, Yc

    # sigma contour levels
    sigma_levels = [3, 2, 1]
    levels = [np.exp(-0.5 * s**2) for s in sigma_levels]

    for i in range(n):
        for j in range(n):

            ax = axes[i, j]

            # only lower triangle
            if i <= j:
                ax.axis("off")
                continue

            p_y = params[i]
            p_x = params[j]

            # ---- 2D ----
            if use_2d and df_2d is not None:
                x2d, y2d = get_samples(df_2d, p_x, p_y)

                if plot_cloud:
                    ax.scatter(
                        x2d, y2d,
                        s=5, alpha=0.2,
                        color="#1f77b4"
                    )

                if plot_contours and len(x2d) > 50:
                    H, Xc, Yc = compute_density(x2d, y2d)
                    X, Y = np.meshgrid(Xc, Yc, indexing="ij")

                    ax.contour(
                        X, Y, H,
                        levels=levels,
                        colors="#1f77b4",
                        linewidths=1.5
                    )

            # ---- 3D ----
            if use_3d and df_3d is not None:
                x3d, y3d = get_samples(df_3d, p_x, p_y)

                if plot_cloud:
                    ax.scatter(
                        x3d, y3d,
                        s=5, alpha=0.2,
                        color="#369b36"
                    )

                if plot_contours and len(x3d) > 50:
                    H, Xc, Yc = compute_density(x3d, y3d)
                    X, Y = np.meshgrid(Xc, Yc, indexing="ij")

                    ax.contour(
                        X, Y, H,
                        levels=levels,
                        colors="#369b36",
                        linewidths=1.5
                    )

            # log scale everywhere
            ax.set_xscale("log")
            ax.set_yscale("log")

            # apply limits if defined
            if param_info[p_x][1] is not None:
                ax.set_xlim(*param_info[p_x][1])
            if param_info[p_y][1] is not None:
                ax.set_ylim(*param_info[p_y][1])

            # --- hide ALL tick labels by default ---
            ax.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                direction="in",
                top=True,
                right=True,
            )

            # --- re-enable only outer labels ---
            if i == n - 1:
                ax.set_xlabel(param_info[p_x][0], fontsize=12)
                ax.tick_params(axis="x", which="both", labelbottom=True)

            if j == 0:
                ax.set_ylabel(param_info[p_y][0], fontsize=12)
                ax.tick_params(axis="y", which="both", labelleft=True)
                

    plt.subplots_adjust(
        top=1,
        bottom=0.06,
        left=0.046,
        right=1,
        hspace=0.086,
        wspace=0.093,
    )
    plt.show()

def plot_figure_3_threepanel_dualaxis(
    df_exp,
    df_sim_2d=None,
    df_sim_3d=None,
    ylims_exp=None,
    ylims_sim=None,
    ylabels_sim=None,   # <-- NEW
    lambda_threshold=2.2,  # <-- same logic as before
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    quantities = [
        ("amplitude_t_cov", r"$\mathrm{CoV}_A$ [%]"),
        ("lambda_t_cov", r"$\mathrm{CoV}_\lambda$ [%]"),
        ("corr_length", r"$\xi/L$"),
    ]

    for i, (key, default_label) in enumerate(quantities):
        ax = axes[i]
        ax_exp = ax.twinx()

        # =========================
        # EXPERIMENT (RIGHT AXIS)
        # =========================
        df_e = df_exp.replace([np.inf, -np.inf], np.nan).dropna(subset=[key, "relN"])

        agg_e = df_e.groupby("relN")[key].agg(["mean", "std", "count"]).reset_index()
        agg_e["sem"] = agg_e["std"] / np.sqrt(agg_e["count"].clip(lower=1))

        ax_exp.errorbar(
            agg_e["relN"],
            agg_e["mean"],
            yerr=agg_e["sem"],
            fmt="o",
            color="red",
            capsize=4,
            linewidth=2,
            markersize=6,
            label="experiment",
        )

        # =========================
        # SIMULATIONS (LEFT AXIS)
        # =========================
        def _plot_sim(df_s, color, label):
            if df_s is None or key not in df_s.columns:
                return

            df_s = df_s.replace([np.inf, -np.inf], np.nan)

            # require lambda for linestyle logic
            if "lambda" not in df_s.columns:
                return

            df_s = df_s.dropna(subset=[key, "relN", "lambda"])
            if len(df_s) == 0:
                return

            agg_s = (
                df_s.groupby("relN")
                .agg({
                    key: "mean",
                    "lambda": "mean",
                })
                .reset_index()
                .sort_values("relN")
            )

            x = agg_s["relN"].values
            y = agg_s[key].values
            lam = agg_s["lambda"].values

            # segment-wise linestyle
            for j in range(len(x) - 1):
                linestyle = ":" if (lam[j] > lambda_threshold or lam[j+1] > lambda_threshold) else "-"

                ax.plot(
                    x[j:j+2],
                    y[j:j+2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                )

            # markers
            ax.scatter(
                x,
                y,
                color=color,
                s=40,
                label=label
            )

        _plot_sim(df_sim_2d, "#1f77b4", "simulation 2D")
        _plot_sim(df_sim_3d, "green", "simulation 3D")

        # =========================
        # AXES
        # =========================
        ax.set_xlabel(r"$N_\mathrm{remain}/N$", fontsize=LABEL_SIZE)

        # ---- LEFT Y LABEL ONLY ----
        label = ylabels_sim[key] if ylabels_sim and key in ylabels_sim else default_label
        ax.set_ylabel(label, fontsize=LABEL_SIZE)

        # explicitly remove right label
        ax_exp.set_ylabel("")

        ax.set_xlim(0.5, 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

        # log scale for corr_length
        if key == "corr_length":
            ax.set_yscale("log")
            ax_exp.set_yscale("log")

        # limits
        if ylims_sim and key in ylims_sim:
            ax.set_ylim(*ylims_sim[key])

        if ylims_exp and key in ylims_exp:
            ax_exp.set_ylim(*ylims_exp[key])

        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax_exp.tick_params(axis="y", labelsize=TICK_SIZE)

        ax.spines["top"].set_visible(False)
        ax_exp.spines["top"].set_visible(False)

        if i == 0:
            ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
            ax_exp.legend(fontsize=LEGEND_SIZE, loc="upper right")

    plt.tight_layout()
    plt.show()

def add_cov_columns(df):
    df = df.copy()

    # avoid division by zero / invalid values
    for key in ["amplitude_t", "lambda_t"]:
        std_col = f"{key}_std"
        mean_col = f"{key}_mean"
        cov_col = f"{key}_cov"

        if std_col in df.columns and mean_col in df.columns:
            mean = df[mean_col].replace(0, np.nan)

            df[cov_col] = 100.0 * df[std_col] / mean  # <-- % CoV

    return df

def print_parameter_stats(df, name, params):
    print(f"\n=== {name} parameter statistics ===\n")

    for p in params:
        if p not in df.columns:
            print(f"{p}: missing")
            continue

        vals = df[p].values
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            print(f"{p}: no valid data")
            continue

        mean = np.mean(vals)
        std = np.std(vals)

        # also useful: log10 stats (since plots are log-scale)
        log_vals = np.log10(vals[vals > 0])
        if len(log_vals) > 0:
            log_mean = np.mean(log_vals)
            log_std = np.std(log_vals)
        else:
            log_mean = np.nan
            log_std = np.nan

        print(f"{p}:")
        print(f"  mean       = {mean:.4g}")
        print(f"  std        = {std:.4g}")
        print(f"  log10 mean = {log_mean:.4f}")
        print(f"  log10 std  = {log_std:.4f}")
        print("")

def append_infinite_motor_rows(df_target, df_source):
    # get mu_a values present in target
    mu_a_target = df_target["mu_a"].unique()

    # select infinite rows from source
    inf_rows = df_source[np.isinf(df_source["Nmotor_scaled"])].copy()

    if len(inf_rows) == 0:
        print("No Nmotor=inf rows found in source")
        return df_target

    # keep only matching mu_a
    inf_rows = inf_rows[np.isin(inf_rows["mu_a"], mu_a_target)]

    if len(inf_rows) == 0:
        print("No matching mu_a values for Nmotor=inf rows")
        return df_target

    df_out = pd.concat([df_target, inf_rows], ignore_index=True)

    print(f"Added {len(inf_rows)} Nmotor=inf rows (matched mu_a)")

    return df_out

def plot_defect_rates_singlepanel(
    df_exp,
    df_sim_2d=None,
    df_sim_3d=None,
    ylims=(1e-2, 20),
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # =========================
    # HELPER
    # =========================
    def _plot(df, color, label_prefix):
        if df is None:
            return

        df = df.replace([np.inf, -np.inf], np.nan)

        required = ["relN", "defect_rate_pos", "defect_rate_neg"]
        if not all(k in df.columns for k in required):
            return

        df = df.dropna(subset=required)
        if len(df) == 0:
            return

        agg = (
            df.groupby("relN")[["defect_rate_pos", "defect_rate_neg"]]
            .agg(["mean", "std", "count"])
        )

        # flatten columns
        agg.columns = ["_".join(col) for col in agg.columns]
        agg = agg.reset_index().sort_values("relN")

        for key, linestyle in zip(
            ["defect_rate_pos", "defect_rate_neg"],
            ["-", "--"]
        ):
            mean = agg[f"{key}_mean"].values
            std = agg[f"{key}_std"].values
            count = agg[f"{key}_count"].values

            sem = std / np.sqrt(np.clip(count, 1, None))
            x = agg["relN"].values

            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt="o",
                linestyle=linestyle,
                color=color,
                capsize=3,
                linewidth=2,
                markersize=5,
                label=f"{label_prefix} {'+1' if 'pos' in key else '-1'}"
            )

    # =========================
    # PLOTS
    # =========================
    _plot(df_exp, "red", "exp")
    _plot(df_sim_2d, "#1f77b4", "2D")
    _plot(df_sim_3d, "green", "3D")

    # =========================
    # AXES
    # =========================
    ax.set_xlabel(r"$N_\mathrm{remain}/N_\mathrm{cilium}$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Defect rate [s$^{-1}$]", fontsize=LABEL_SIZE)

    ax.set_xlim(0.5, 1.0)
    ax.set_yscale("log")

    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.show()

def print_power_stats(df):
    # ---------------------------------------------
    # Filter: ATP = 750, KCl = 0, WT experiments
    # ---------------------------------------------
    mask = (
        (df["ATP_uM"] == 750) &
        (df["KCl_mM"] == 0) &
        (df["sexp"].astype(str).str.startswith("WT_"))
    )

    df_sel = df[mask].copy()

    print(f"Selected {len(df_sel)} entries\n")

    if len(df_sel) == 0:
        print("⚠️ No matching data")
        return

    # ---------------------------------------------
    # Helper for stats
    # ---------------------------------------------
    def stats(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]

        N = len(x)
        if N == 0:
            return np.nan, np.nan, np.nan, 0

        mean = np.mean(x)
        std = np.std(x, ddof=1) if N > 1 else np.nan
        sem = std / np.sqrt(N) if N > 1 else np.nan

        return mean, std, sem, N

    # ---------------------------------------------
    # Compute stats (in fW)
    # ---------------------------------------------
    for key in ["power_free_fW", "power_chlamy_fW"]:
        mean, std, sem, N = stats(df_sel[key])

        print(f"{key}:")
        print(f"  N   = {N}")
        print(f"  mean = {mean:.3f} fW")
        print(f"  std  = {std:.3f} fW")
        print(f"  SEM  = {sem:.3f} fW\n")

# ================================
# MAIN
# ================================
def main():
    csv_path = Path("./scalar_observables.csv")

    df_exp = prepare_experiment_df(csv_path)
    print_power_stats(df_exp)
    
    print("Unique relN:", sorted(df_exp["relN"].unique()))
    compute_sbi_targets(df_exp)
    plot_SI_R2_limitcycle(df_exp)
    df_cass_hydro = pd.read_csv("hydro_cass_2d.csv")
    print_hydro_boxplot_stats(df_exp, df_cass_hydro)
    plot_hydro_boxplot(df_exp, df_cass_hydro)
    
    

    df_sim_2d_phasespace = prepare_simulation_2d_phasespace_df("./scalar_observables_phasespace.csv") 
    df_sim_2d_wn= prepare_simulation_2d_phasespace_df("./scalar_observables_wn.csv") 
    df_sim_2d_cwn = prepare_simulation_2d_phasespace_df("./scalar_observables_cwn.csv") 

    df_sim_2d_wn = append_infinite_motor_rows(df_sim_2d_wn, df_sim_2d_phasespace)
    df_sim_2d_cwn = append_infinite_motor_rows(df_sim_2d_cwn, df_sim_2d_phasespace)
    plot_phase_diagnostics(df_sim_2d_phasespace)
   
    N0s = np.array([85, 500, np.inf]) * 200
    mu_a_crits = compute_mu_a_crits(df_sim_2d_phasespace)
    transitions = detect_phase_transitions(df_sim_2d_phasespace)
    plot_figure_2(
        df_sim_2d_phasespace,
        transitions,
        N0s,
        mu_a_crits=mu_a_crits
    )

    plot_figure_2(
        df_sim_2d_wn,
        transitions,
        N0s,
        mu_a_crits=mu_a_crits
    )

    plot_figure_2(
        df_sim_2d_cwn,
        transitions,
        N0s,
        mu_a_crits=mu_a_crits
    )

    df_sim_3d = pd.read_csv("scalar_observables_cass_3d_extraction.csv")
    df_sim_2d = pd.read_csv("scalar_observables_cass_2d_extraction.csv")
    df_sim_3d['relN']=df_sim_3d['Nmotor']/20.0 
    df_sim_2d['relN']=df_sim_2d['Nmotor']/85.0 
    plot_figure_3(df_exp, df_sim_2d=df_sim_2d, df_sim_3d=df_sim_3d)
  
    df_exp    = add_cov_columns(df_exp)
    df_sim_2d = add_cov_columns(df_sim_2d)
    df_sim_3d = add_cov_columns(df_sim_3d)
    plot_figure_3_threepanel_dualaxis(
        df_exp,
        df_sim_2d=df_sim_2d,
        df_sim_3d=df_sim_3d,
        ylims_sim={
            "amplitude_t_cov": (0, 10),
            "lambda_t_cov": (0, 50),
            "corr_length": (50, 10000),
        },
        ylims_exp={
            "amplitude_t_cov": (0, 30),
            "lambda_t_cov": (0, 150),
            "corr_length": (50, 2000),
        },
        ylabels_sim={
            "amplitude_t_cov": r"CoV local amplitude [%]",
            "lambda_t_cov": r"CoV inst. wavelength [%]",
            "corr_length": r"Correlation length $\xi/L$",
        }
    )

    plot_defect_rates_singlepanel(
        df_exp,
        df_sim_2d=df_sim_2d,
        df_sim_3d=df_sim_3d,
    )

    df_sim_cass = pd.read_csv("scalar_observables_cass_2d_extraction_cass.csv")
    df_sim_cass5fold = pd.read_csv("scalar_observables_cass_2d_extraction_cass_5fold.csv")
    df_sim_cass['relN']=df_sim_cass['Nmotor']/85.0 
    df_sim_cass5fold['relN']=df_sim_cass5fold['Nmotor']/500.0 
    plot_figure_3(df_exp, df_sim_2d=df_sim_cass, df_sim_3d=df_sim_cass5fold)

    
    df_sampled_3d = pd.read_csv("gp_logposterior_samples_3d.csv")
    df_sampled_2d = pd.read_csv("gp_logposterior_samples_2d.csv")

    params = ("K_d2","B","F0","Fc","v0","v0_eps0","pi0","b")

    print_parameter_stats(df_sampled_2d, "2D", params)
    print_parameter_stats(df_sampled_3d, "3D", params)
    plot_weighted_density_F0_v0(df_sampled_2d,df_sampled_3d)
    corner_plot_parameters(df_sampled_2d, df_sampled_3d, plot_cloud=False)
   

if __name__ == "__main__":
    main()