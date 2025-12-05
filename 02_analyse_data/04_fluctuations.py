import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import re

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from utils_rdf.config import (
    OUTPUT_DATA_EXTRACTION_DIR,
    OUTPUT_DATA_NEW_PARAMETER_DIR,OUTPUT_DATA_FLUCTUATIONS_EXTRACTION, OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER,
    OUTPUT_DATA_SHARME_DIR, OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT,DT_SIMULATION,PHASE_DEFECT_RATES_CSV,
    SPATIAL_CORRELATION_CSV
)

# --- utility
def save_npz_safe(path, **arrays):
    """
    Save arrays to a compressed npz file, creating directories if needed.
    Handles ragged lists (arrays of different shapes) by converting them to dtype=object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_arrays = {}
    for key, val in arrays.items():
        try:
            # Attempt to convert to ndarray directly
            arr = np.asarray(val)
            # detect raggedness: ndim==1 and any element is an array of differing shape
            if arr.dtype == object:
                safe_arrays[key] = arr
            else:
                safe_arrays[key] = arr
        except Exception:
            # fallback: force object array
            safe_arrays[key] = np.array(val, dtype=object)

    try:
        np.savez_compressed(path, **safe_arrays)
    except ValueError as e:
        # fallback: force every value to object array
        print(f"⚠️ Ragged data in {path.name}, saving as object arrays ({e})")
        for k, v in arrays.items():
            safe_arrays[k] = np.array(v, dtype=object)
        np.savez_compressed(path, **safe_arrays)

def _sanitize_token(x):
    s = str(x)
    s = s.replace(" ", "")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = s.replace(":", "_")
    s = s.replace(",", "_")
    return s

def make_experiment_group_name(genotype, ATP, KCL):
    """
    Create a stable folder name for experimental groups: genotype_..._ATP_..._KCL_...
    Floats are formatted compactly; strings are sanitized.
    """
    def _fmt_num(v):
        # keep ints as int-like, small floats compact
        try:
            fv = float(v)
            if abs(fv - round(fv)) < 1e-9:
                return str(int(round(fv)))
            return f"{fv:g}"
        except Exception:
            return _sanitize_token(v)
    g = _sanitize_token(genotype)
    a = _fmt_num(ATP)
    k = _fmt_num(KCL)
    return f"genotype_{g}_ATP_{a}_KCL_{k}"

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


# ----------------------------------------------------------------------
# --- individual processings
# ----------------------------------------------------------------------

def global_aniso(all_A_global,all_omega_global, plot=True):
    # concatenate globals
    A = np.concatenate(all_A_global)
    omega = np.concatenate(all_omega_global)
    
    if plot:
        bins = 150
        H, xedges, yedges = np.histogram2d(A, omega, bins=bins, density=True)
        xcent = 0.5 * (xedges[:-1] + xedges[1:])
        ycent = 0.5 * (yedges[:-1] + yedges[1:])
        Xc, Yc = np.meshgrid(xcent, ycent, indexing="ij")

        Z = np.log10(H + 1e-12)
        vmin, vmax = np.nanpercentile(Z[np.isfinite(Z)], [10, 99.9])
        levels = np.linspace(vmin, vmax, 12)

        fig, ax = plt.subplots(figsize=(5, 4))
        cs = ax.contour(Xc, Yc, Z, levels=levels, cmap="viridis", linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        ax.set_xlabel("Global Amplitude A")
        ax.set_ylabel("Global ω")
        ax.set_title("Global amplitude vs frequency (log density)")
        ax.grid(True, alpha=0.3)

        # fit (ω−1) = χ·(A−1)
        valid = np.isfinite(A) & np.isfinite(omega)
        A_fit, w_fit = A[valid], omega[valid]
        A_shift, w_shift = A_fit - 1.0, w_fit - 1.0
        chi_global = np.dot(A_shift, w_shift) / np.dot(A_shift, A_shift)
        xline = np.linspace(*ax.get_xlim(), 200)
        ax.plot(xline, 1 + chi_global * (xline - 1), "r", lw=1.5,
                label=fr"(ω−1)={chi_global:.3g}(A−1)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {"A": A, "omega": omega}

def global_aniso_phase(all_phi_global, all_omega_global, all_A_global, plot = True):
    M = 5
    phi_grid = np.linspace(0, 2 * np.pi, 400)
    chi_all = []

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
    for phi_g, omega_g, amp_g in zip(all_phi_global, all_omega_global, all_A_global):
        da = amp_g - np.mean(amp_g)
        dw = omega_g - np.mean(omega_g)
        mask = (np.abs(da) < 0.2) & (np.abs(dw) < 0.2)
        phi_mask = np.mod(phi_g[mask], 2 * np.pi)
        if phi_mask.size < 50:
            continue

        cols = [da[mask]]
        for m in range(1, M + 1):
            cols.append(da[mask] * np.cos(m * phi_mask))
            cols.append(da[mask] * np.sin(m * phi_mask))
        X = np.column_stack(cols)
        theta, *_ = np.linalg.lstsq(X, dw[mask], rcond=None)
        
        #rotate theta
        def idx_am(m): return 2*m - 1
        def idx_bm(m): return 2*m

        a0 = theta[0]
        # extract a_m, b_m for m_align
        a_m = theta[idx_am(2)]
        b_m = theta[idx_bm(2)]
        delta = 0.5 * np.arctan2(-b_m, a_m)

        # rotate all modes consistently
        theta_rot = theta.copy()
        for m in range(1, M+1):
            am = theta[idx_am(m)]
            bm = theta[idx_bm(m)]
            c  = np.cos(m*delta)
            s  = np.sin(m*delta)
            a_new = am*c - bm*s
            b_new = am*s + bm*c
            theta_rot[idx_am(m)] = a_new
            theta_rot[idx_bm(m)] = b_new
        # a0 is invariant under phase shift
        theta_rot[0] = a0

        chi_phi = theta_rot[0] * np.ones_like(phi_grid)
        for m in range(1, M+1):
            chi_phi += theta_rot[idx_am(m)] * np.cos(m * phi_grid) \
                    + theta_rot[idx_bm(m)] * np.sin(m * phi_grid)
        chi_all.append(chi_phi)

        if plot:
            ax.plot(phi_grid, chi_phi, alpha=0.85)  # aligned curve
        
    chi_mean = np.mean(np.vstack(chi_all), axis=0)

    if plot:
        ax.plot(phi_grid, chi_mean, "k", lw=2, label="mean χ(φ)")
        ax.set_xlabel(r"Phase $\phi$")
        ax.set_ylabel(r"$\chi(\phi)$")
        ax.set_title("Phase-dependent stiffness χ(φ)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        "phi_grid": phi_grid,
        "chi_all": np.array(chi_all, dtype=object)
    }

def correlation_time_amp_omega(
    all_omega_global, all_A_global, plot=True, max_dt=100, s_rel_positions=(0.5, 1.0)
):
    """
    Compute temporal autocorrelation <A(s,t)A(s,t+dt)>_st and <ω(s,t)ω(s,t+dt)>_st
    for all datasets in the lists, average them, and plot with SEM error bars.
    Additionally, plot autocorrelations at specific spatial positions s_rel.

    Parameters
    ----------
    all_omega_global : list of np.ndarray
        Each array is shape (T,) or (T, n_s): temporal series of frequency.
    all_A_global : list of np.ndarray
        Each array is shape (T,) or (T, n_s): temporal series of amplitude.
    plot : bool, default=True
        If True, plot both global and position-specific correlations.
    max_dt : int, default=100
        Maximum lag (frames).
    s_rel_positions : tuple of floats
        Relative spatial positions (0–1) at which to compute correlations.

    Returns
    -------
    lags : np.ndarray
        Lag values (frames).
    results : dict
        Contains mean and SEM for global and position-specific correlations:
        {
            "A_mean": ..., "A_sem": ...,
            "omega_mean": ..., "omega_sem": ...,
            "A_pos": {s_rel: mean array}, "omega_pos": {s_rel: mean array}
        }
    """
    lags = np.arange(1, max_dt + 1)
    C_A_all, C_omega_all = [], []
    C_A_pos, C_omega_pos = {s_rel: [] for s_rel in s_rel_positions}, {s_rel: [] for s_rel in s_rel_positions}

    for A, w in tqdm(zip(all_A_global, all_omega_global),
                     total=len(all_A_global), desc="autocorr datasets", leave=False):
        A = np.asarray(A)
        w = np.asarray(w)
        if A.ndim == 1:
            A = A[:, None]
        if w.ndim == 1:
            w = w[:, None]

        T, n_s = A.shape
        if T <= max_dt + 1:
            continue

        # normalize to zero mean
        A = A - np.nanmean(A, axis=0, keepdims=True)
        w = w - np.nanmean(w, axis=0, keepdims=True)

        var_A = np.nanmean(A**2)
        var_w = np.nanmean(w**2)

        C_A = np.zeros_like(lags, dtype=float)
        C_w = np.zeros_like(lags, dtype=float)

        # global average
        for i, dt in enumerate(lags):
            A_t   = A[:-dt, :]
            A_tdt = A[dt:, :]
            w_t   = w[:-dt, :]
            w_tdt = w[dt:, :]

            C_A[i] = np.nanmean(A_t * A_tdt) / var_A
            C_w[i] = np.nanmean(w_t * w_tdt) / var_w

        C_A_all.append(C_A)
        C_omega_all.append(C_w)

        # position-specific correlations
        for s_rel in s_rel_positions:
            s_idx = int(np.clip(round(s_rel * (n_s - 1)), 0, n_s - 1))
            C_A_s = np.zeros_like(lags, dtype=float)
            C_w_s = np.zeros_like(lags, dtype=float)
            for i, dt in enumerate(lags):
                C_A_s[i] = np.nanmean(A[:-dt, s_idx] * A[dt:, s_idx]) / np.nanmean(A[:, s_idx]**2)
                C_w_s[i] = np.nanmean(w[:-dt, s_idx] * w[dt:, s_idx]) / np.nanmean(w[:, s_idx]**2)
            C_A_pos[s_rel].append(C_A_s)
            C_omega_pos[s_rel].append(C_w_s)

    if not C_A_all:
        print("⚠️ No valid data for autocorrelation")
        return lags, {}

    # convert to arrays and compute mean/SEM
    C_A_all = np.vstack(C_A_all)
    C_omega_all = np.vstack(C_omega_all)

    C_A_mean = np.nanmean(C_A_all, axis=0)
    C_omega_mean = np.nanmean(C_omega_all, axis=0)
    C_A_sem = np.nanstd(C_A_all, axis=0, ddof=1) / np.sqrt(C_A_all.shape[0])
    C_omega_sem = np.nanstd(C_omega_all, axis=0, ddof=1) / np.sqrt(C_omega_all.shape[0])

    results = dict(
        A_mean=C_A_mean,
        omega_mean=C_omega_mean,
        A_sem=C_A_sem,
        omega_sem=C_omega_sem,
        A_pos={},
        omega_pos={}
    )

    # aggregate position-specific means
    for s_rel in s_rel_positions:
        A_pos_arr = np.vstack(C_A_pos[s_rel])
        w_pos_arr = np.vstack(C_omega_pos[s_rel])
        results["A_pos"][s_rel] = np.nanmean(A_pos_arr, axis=0)
        results["omega_pos"][s_rel] = np.nanmean(w_pos_arr, axis=0)

    # --- Plot ---
    if plot:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        # global curves with error bars
        ax1.errorbar(lags, C_A_mean, yerr=C_A_sem, fmt="o-", lw=1.5,
                     color="tab:blue", capsize=3, label="⟨A(t)A(t+Δt)⟩ (global)")
        ax2.errorbar(lags, C_omega_mean, yerr=C_omega_sem, fmt="s--", lw=1.5,
                     color="tab:red", capsize=3, label="⟨ω(t)ω(t+Δt)⟩ (global)")

        # position-specific lines (no error bars, different markers)
        markers_A = ["^", "v", "D", "x"]
        markers_w = ["P", "*", "X", "1"]
        for i, s_rel in enumerate(s_rel_positions):
            label_suffix = f"s={s_rel:.1f}"
            ax1.plot(lags, results["A_pos"][s_rel],
                     marker=markers_A[i % len(markers_A)],
                     ls=":", lw=1.2, color="tab:blue",
                     label=f"A at {label_suffix}")
            ax2.plot(lags, results["omega_pos"][s_rel],
                     marker=markers_w[i % len(markers_w)],
                     ls=":", lw=1.2, color="tab:red",
                     label=f"ω at {label_suffix}")

        ax1.set_xlabel("Lag Δt (frames)")
        ax1.set_ylabel("Amplitude correlation", color="tab:blue")
        ax2.set_ylabel("Frequency correlation", color="tab:red")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Temporal autocorrelation (global ± SEM and selected s)")

        # combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show()

    return {
        "lags": lags,
        "C_A_all": C_A_all,
        "C_omega_all": C_omega_all,
        "results": results
    }

def local_aniso(local_omega_list, local_A_list, plot = True):
    A_all = np.concatenate(local_A_list, axis=0)
    omega_all = np.concatenate(local_omega_list, axis=0)

    good_n=np.std(omega_all,axis=0)<0.20
    A_all=A_all[:,good_n].flatten()
    omega_all=omega_all[:,good_n].flatten()
    
    if plot:
        bins = 150
        H, xedges, yedges = np.histogram2d(A_all, omega_all, bins=bins, density=True)
        xcent = 0.5 * (xedges[:-1] + xedges[1:])
        ycent = 0.5 * (yedges[:-1] + yedges[1:])
        Xc, Yc = np.meshgrid(xcent, ycent, indexing="ij")

        Z = np.log10(H + 1e-12)
        vmin, vmax = np.nanpercentile(Z[np.isfinite(Z)], [10, 99.9])
        levels = np.linspace(vmin, vmax, 12)
        fig, ax = plt.subplots(figsize=(5, 4))
        cs = ax.contour(Xc, Yc, Z, levels=levels, cmap="viridis", linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        ax.set_xlabel("loc Amplitude A")
        ax.set_ylabel("loc ω")
        ax.set_title("loc amplitude vs frequency (log density)")
        ax.grid(True, alpha=0.3)

        # fit (ω−1) = χ·(A−1)
        valid = np.isfinite(A_all) & np.isfinite(omega_all)
        A_fit, w_fit = A_all[valid], omega_all[valid]
        A_shift, w_shift = A_fit - 1.0, w_fit - 1.0
        chi_global = np.dot(A_shift, w_shift) / np.dot(A_shift, A_shift)
        xline = np.linspace(*ax.get_xlim(), 200)
        ax.plot(xline, 1 + chi_global * (xline - 1), "r", lw=1.5,
                label=fr"(ω−1)={chi_global:.3g}(A−1)")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return {"A_all": A_all, "omega_all": omega_all}

def local_aniso_phase(local_phi_list, local_omega_list, local_A_list,
                      plot=True, M=7, std_omega_thresh=0.20,
                      amp_clip=0.2, omega_clip=0.2, min_pts=50):
    """
    Phase-resolved local stiffness χ(φ, s) computed separately for each dataset and spatial position s.
    - Selects 'good' s where std_t(omega[:, s]) < std_omega_thresh
    - For each good s: regress (ω-<ω>) on (A-<A>) with Fourier features in φ, then
      rotate all harmonic coefficients by δ determined from the m=2 mode.
    - Plots each χ(φ, s) curve with a color gradient keyed to s, plus the grand mean.

    Parameters
    ----------
    local_phi_list, local_omega_list, local_A_list : list of np.ndarray
        Each entry has shape (T, n_s) for a dataset.
    plot : bool
        Show plot.
    M : int
        Max harmonic order for χ(φ).
    std_omega_thresh : float
        Threshold for selecting spatial positions by temporal std of ω.
    amp_clip, omega_clip : float
        Clip thresholds for making the regression mask: |A-<A>|<amp_clip and |ω-<ω>|<omega_clip.
    min_pts : int
        Minimum number of time points after masking to attempt a fit.

    Returns
    -------
    phi_grid : (Nphi,) np.ndarray
        Common phase grid (0..2π).
    chi_all : list of np.ndarray
        List of χ(φ) curves (aligned), one per (dataset, s) kept.
    kept_indices : list of dict
        Metadata for each curve: {'dataset': i, 's_idx': j, 'n_t': T_used}
    """


    phi_grid = np.linspace(0, 2 * np.pi, 400)
    chi_all = []
    kept_indices = []

    def idx_am(m): return 2 * m - 1
    def idx_bm(m): return 2 * m

    # Iterate over datasets
    for di, (phi, omega, amp) in enumerate(zip(local_phi_list, local_omega_list, local_A_list)):
        if phi.ndim != 2:
            phi = np.atleast_2d(phi)
        if omega.ndim != 2:
            omega = np.atleast_2d(omega)
        if amp.ndim != 2:
            amp = np.atleast_2d(amp)

        T, n_s = phi.shape
        if T < min_pts:
            continue

        # choose spatial positions by ω-stability
        good_s = np.nanstd(omega, axis=0) < std_omega_thresh
        if not np.any(good_s):
            continue

        # color gradient over s for plotting
        cmap = get_cmap("viridis")
        norm = Normalize(vmin=0, vmax=max(n_s - 1, 1))

        for sj in np.where(good_s)[0]:
            phi_s = np.mod(phi[:, sj], 2 * np.pi)
            A_s = amp[:, sj]
            w_s = omega[:, sj]

            # demean and clip to modest excursions (similar to global)
            da = A_s - np.nanmean(A_s)
            dw = w_s - np.nanmean(w_s)
            mask = np.isfinite(da) & np.isfinite(dw) & np.isfinite(phi_s)
            mask &= (np.abs(da) < amp_clip) & (np.abs(dw) < omega_clip)

            if np.count_nonzero(mask) < min_pts:
                continue

            # Build design matrix: [da, da*cos φ, da*sin φ, ..., up to M]
            cols = [da[mask]]
            for m in range(1, M + 1):
                c = np.cos(m * phi_s[mask])
                s = np.sin(m * phi_s[mask])
                cols.append(da[mask] * c)
                cols.append(da[mask] * s)
            X = np.column_stack(cols)

            # Solve least squares: dw ≈ X @ theta
            theta, *_ = np.linalg.lstsq(X, dw[mask], rcond=None)

            # Phase alignment: use m_align=2 like in global
            a0 = theta[0]
            a_m = theta[idx_am(2)]
            b_m = theta[idx_bm(2)]
            # δ chosen so that m=2 sine term is minimized (rotate to cosine)
            delta = 0.5 * np.arctan2(-b_m, a_m)

            # Rotate all harmonic coefficients by δ (per-s alignment)
            theta_rot = theta.copy()
            for m in range(1, M + 1):
                am = theta[idx_am(m)]
                bm = theta[idx_bm(m)]
                c = np.cos(m * delta)
                s = np.sin(m * delta)
                a_new = am * c - bm * s
                b_new = am * s + bm * c
                theta_rot[idx_am(m)] = a_new
                theta_rot[idx_bm(m)] = b_new
            theta_rot[0] = a0  # invariant

            # Evaluate χ(φ) on common grid
            chi_phi = theta_rot[0] * np.ones_like(phi_grid)
            for m in range(1, M + 1):
                chi_phi += theta_rot[idx_am(m)] * np.cos(m * phi_grid) \
                         + theta_rot[idx_bm(m)] * np.sin(m * phi_grid)

            chi_all.append(chi_phi)
            kept_indices.append({'dataset': di, 's_idx': sj, 'n_t': int(np.count_nonzero(mask))})

    if len(chi_all) == 0:
        if plot:
            print("⚠️ No valid local positions passed filters; saving dummy result.")
        # Return dummy arrays so downstream save_npz_safe still works
        return {
            "phi_grid": np.linspace(0, 2 * np.pi, 400),
            "chi_all": np.empty((0,), dtype=object),
            "kept_indices": np.empty((0,), dtype=object)
        }

    chi_stack = np.vstack(chi_all)
    chi_mean = np.nanmean(chi_stack, axis=0)

    # ---------- Plot ----------
    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Re-plot with color keyed to s (we need to iterate with metadata)
        # Build a map from (dataset -> n_s) to normalize consistently per dataset.
        # We approximate by using the maximum s index encountered per dataset.
        per_ds_max_s = {}
        for meta in kept_indices:
            di = meta['dataset']
            per_ds_max_s[di] = max(per_ds_max_s.get(di, -1), meta['s_idx'])
        cmap = plt.get_cmap("viridis")

        for chi_phi, meta in zip(chi_all, kept_indices):
            di = meta['dataset']
            sj = meta['s_idx']
            vmax = max(per_ds_max_s.get(di, sj), 1)
            color = cmap(sj / vmax)
            ax.plot(phi_grid, chi_phi, lw=1.0, alpha=0.55, color=color)

        # Mean curve
        ax.plot(phi_grid, chi_mean, "k", lw=2.0, label="mean χ(φ)")

        ax.set_xlabel(r"Phase $\phi$")
        ax.set_ylabel(r"$\chi(\phi)$")
        ax.set_title(r"Local phase-dependent stiffness $\chi(\phi, s)$ (aligned per $s$)")
        ax.grid(True, alpha=0.3)

        # Add a small colorbar keyed to s (0..1 relative index inside each dataset).
        # Since s-range differs per dataset, we indicate it's a relative index.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("relative s-index (per dataset)")

        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.show()

    return {
        "phi_grid": phi_grid,
        "chi_all": np.array(chi_all, dtype=object),
        "kept_indices": kept_indices
    }

def spatial_correlation_local_phase(local_phi_list, plot=True):
    """
    Compute spatial phase–phase correlation C(Δs) averaged across datasets.

    For each dataset (φ(t, s)):
        C_ds = mean_t,s |⟨e^{i(φ(t, s+Δs)−φ(t, s))}⟩_t|
    Then averaged over datasets, returning mean and SEM.

    Parameters
    ----------
    local_phi_list : list of np.ndarray
        Each array has shape (T, n_s): phase fields (radians).
    plot : bool
        If True, shows correlation curves.

    Returns
    -------
    dict with:
        ds_axis : array of spatial offsets
        angular_corr_mean : mean correlation
        angular_corr_sem : standard error of the mean
        angular_corr_all : list of per-dataset correlation curves
    """
    all_corrs = []
    ds_axis_common = None

    # loop over datasets
    for phi in tqdm(local_phi_list, desc="Spatial correlation datasets", leave=False):
        phi = np.asarray(phi)
        if phi.ndim != 2:
            raise ValueError(f"Expected 2D (time × space), got {phi.shape}")

        n_t, n_s = phi.shape
        if n_s < 3:
            continue

        max_ds = n_s - 2
        ds_axis = np.arange(1, max_ds)
        angular_corr = np.zeros(max_ds - 1)

        for ds in tqdm(range(1, max_ds), desc="Δs correlation", leave=False):
            # complex phase difference correlation
            diff = np.exp(1j * (phi[:, :-ds] - phi[:, ds:]))
            # average over time, take magnitude
            temp_mean = np.nanmean(diff, axis=0)
            angular_corr[ds - 1] = np.nanmean(np.abs(temp_mean))

        all_corrs.append(angular_corr)
        ds_axis_common = ds_axis

    if not all_corrs:
        print("⚠️ No valid datasets for spatial correlation")
        return {}

    # stack and compute mean + SEM
    corr_mat = np.vstack(all_corrs)
    angular_corr_mean = np.nanmean(corr_mat, axis=0)
    angular_corr_sem = np.nanstd(corr_mat, axis=0, ddof=1) / np.sqrt(corr_mat.shape[0])

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        for c in corr_mat:
            ax.semilogy(ds_axis_common, c, color="gray", alpha=0.3, lw=0.8)
        ax.errorbar(ds_axis_common, angular_corr_mean, yerr=angular_corr_sem,
                    fmt="o-", lw=1.5, color="tab:blue", capsize=3, label="mean ± SEM")
        ax.set_xlabel(r"Δs (spatial offset)")
        ax.set_ylabel(r"$C(Δs)$")
        ax.set_title("Spatial phase–phase correlation")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        "ds_axis": ds_axis_common,
        "angular_corr_mean": angular_corr_mean,
        "angular_corr_sem": angular_corr_sem,
        "angular_corr_all": np.array(all_corrs, dtype=object),
    }

def temporal_correlation_local_phase(local_phi_list, plot=True, n_lags=20):
    """
    Temporal phase statistics from local phases φ(t, s).
    - Treats each spatial position s separately.
    - Uses a common log-spaced lag grid up to 1% of the *shortest* dataset length.
    - Computes:
        1) MSD(τ) = ⟨(unwrap[φ(t+τ) - φ(t)])^2⟩_t  (per s)
        2) C(τ)   = ⟨cos(φ(t+τ) - φ(t))⟩_t         (per s, circular)
      and then averages over all s and datasets.
    - Plots the mean over all s, plus representative positions s=0, s=mid, s=1
      (each averaged over datasets at those relative indices).

    Parameters
    ----------
    local_phi_list : list of np.ndarray
        Each array has shape (T, n_s) for one dataset (time × space).
    plot : bool
        If True, make diagnostic plots.
    n_lags : int
        Number of points in the logarithmic lag grid (before uniqueness).

    Returns
    -------
    lags : (L,) int ndarray
        Common lag values (frames).
    results : dict
        {
          "msd_mean": (L,) float,          # mean across all s & datasets
          "circ_mean": (L,) float,         # same for circular correlation
          "msd_s0": (L,) float,            # s=0 averaged over datasets
          "msd_smid": (L,) float,          # s=mid averaged over datasets
          "msd_s1": (L,) float,            # s=1 averaged over datasets
          "circ_s0": (L,) float,
          "circ_smid": (L,) float,
          "circ_s1": (L,) float,
          "msd_all": list of (L, n_s) arrays per dataset,
          "circ_all": list of (L, n_s) arrays per dataset
        }
    """

    # -----------------------
    # Build a common lag grid
    # -----------------------
    if not local_phi_list:
        raise ValueError("local_phi_list is empty")

    T_list = [np.asarray(phi).shape[0] for phi in local_phi_list]
    min_T = int(np.min(T_list))
    if min_T < 10:
        raise ValueError(f"Datasets too short (min T={min_T}). Need at least ~10 frames.")

    max_lag = max(1, int(0.01 * min_T))  # 1% of shortest dataset
    # log-spaced, integer, unique, within [1, max_lag]
    lags = np.unique(np.clip(np.round(np.logspace(0, np.log10(max_lag), n_lags)).astype(int), 1, max_lag))

    msd_per_dataset = []   # each element: (L, n_s)
    circ_per_dataset = []  # each element: (L, n_s)

    # Representative positions across datasets (per relative index)
    msd_s0_list, msd_smid_list, msd_s1_list = [], [], []
    circ_s0_list, circ_smid_list, circ_s1_list = [], [], []

    # -----------------------
    # Process each dataset
    # -----------------------
    for phi in local_phi_list:
        phi = np.asarray(phi)
        if phi.ndim != 2:
            phi = np.atleast_2d(phi)
        T, n_s = phi.shape

        # unwrap along time for MSD
        phi_unw = np.unwrap(phi, axis=0)

        L = len(lags)
        msd = np.full((L, n_s), np.nan, dtype=float)
        circ = np.full((L, n_s), np.nan, dtype=float)

        for i, lag in enumerate(lags):
            if lag >= T:
                continue
            dphi_unw = phi_unw[lag:, :] - phi_unw[:-lag, :]
            dphi_circ = phi[lag:, :] - phi[:-lag, :]  # wrapped diffs

            # MSD per s
            msd[i, :] = np.nanmean(dphi_unw ** 2, axis=0)

            # Complex correlation magnitude |<e^{iΔφ}>|
            cval = np.nanmean(np.exp(1j * dphi_circ), axis=0)
            circ[i, :] = np.abs(cval)

        msd_per_dataset.append(msd)
        circ_per_dataset.append(circ)

        # Representative positions: 0, mid, 1 (relative per dataset)
        s0 = 0
        smid = n_s // 2
        s1 = n_s - 1

        msd_s0_list.append(msd[:, s0])
        msd_smid_list.append(msd[:, smid])
        msd_s1_list.append(msd[:, s1])

        circ_s0_list.append(circ[:, s0])
        circ_smid_list.append(circ[:, smid])
        circ_s1_list.append(circ[:, s1])

    # -----------------------
    # Aggregate across datasets & s
    # -----------------------
    # Mean over *all* s and datasets:
    msd_all_concat = np.concatenate(msd_per_dataset, axis=1)   # (L, sum n_s)
    circ_all_concat = np.concatenate(circ_per_dataset, axis=1) # (L, sum n_s)

    msd_mean = np.nanmean(msd_all_concat, axis=1)
    circ_mean = np.nanmean(circ_all_concat, axis=1)

    # Means for representative positions (averaged over datasets)
    def _mean_stack(arr_list):
        arr = np.stack(arr_list, axis=1)  # (L, n_datasets)
        return np.nanmean(arr, axis=1)

    msd_s0 = _mean_stack(msd_s0_list)
    msd_smid = _mean_stack(msd_smid_list)
    msd_s1 = _mean_stack(msd_s1_list)

    circ_s0 = _mean_stack(circ_s0_list)
    circ_smid = _mean_stack(circ_smid_list)
    circ_s1 = _mean_stack(circ_s1_list)

    results = dict(
        msd_mean=msd_mean,
        circ_mean=circ_mean,
        msd_s0=msd_s0,
        msd_smid=msd_smid,
        msd_s1=msd_s1,
        circ_s0=circ_s0,
        circ_smid=circ_smid,
        circ_s1=circ_s1,
        msd_all=msd_per_dataset,
        circ_all=circ_per_dataset,
    )

    # -----------------------
    # Plots
    # -----------------------
    if plot:
        # 1) MSD: diffusion ~ linear in τ (so log-log is informative)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.loglog(lags, msd_mean, "o-", lw=1.8, label="MSD mean over all s")
        ax1.loglog(lags, msd_s0, "s--", lw=1.2, label="MSD at s=0 (avg over sets)")
        ax1.loglog(lags, msd_smid, "d--", lw=1.2, label="MSD at s=mid (avg over sets)")
        ax1.loglog(lags, msd_s1, "x--", lw=1.2, label="MSD at s=1 (avg over sets)")
        ax1.set_xlabel(r"Lag $\Delta t$ (frames)")
        ax1.set_ylabel(r"MSD$(\Delta t)$")
        ax1.set_title("Temporal phase diffusion (MSD, unwrapped)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # 2) Circular correlation: typically decays with τ
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.semilogx(lags, circ_mean, "o-", lw=1.8, label=r"$|⟨e^{i\Delta\phi}⟩|$ mean over all s")
        ax2.semilogx(lags, circ_s0, "s--", lw=1.2, label=r"$|⟨e^{i\Delta\phi}⟩|$ at s=0 (avg)")
        ax2.semilogx(lags, circ_smid, "d--", lw=1.2, label=r"$|⟨e^{i\Delta\phi}⟩|$ at s=mid (avg)")
        ax2.semilogx(lags, circ_s1, "x--", lw=1.2, label=r"$|⟨e^{i\Delta\phi}⟩|$ at s=1 (avg)")
        ax2.set_xlabel(r"Lag $\Delta t$ (frames)")
        ax2.set_ylabel(r"$|⟨e^{i\Delta\phi}⟩|$")
        ax2.set_title("Temporal phase coherence magnitude")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return {
        "lags": lags,
        "msd_per_dataset": msd_per_dataset,
        "circ_per_dataset": circ_per_dataset,
        "results": results
    }

def phase_defects(local_phi_list, plot=True, cutoff=3):
    """
    Detect topological defects (asters / anti-asters) in space-time phase fields.

    Parameters
    ----------
    local_phi_list : list of np.ndarray
        Each array has shape (T, n_s), phase in radians.
    plot : bool
        If True, show phase kymographs with detected defects overlaid.
    cutoff : float
        Minimum distance (in pixels) below which opposite-charge pairs are removed.
        (for plotting)
    Returns
    -------
    all_results : list of dict
        Each element contains:
            {
                "charges": 2D array of integer charges per cell,
                "pos_coords": array of (i,j) +1 defect coordinates,
                "neg_coords": array of (i,j) -1 defect coordinates,
            }
    """

    all_results = []
    phi_aligned_list = []

    for idx, phi_field in enumerate(local_phi_list):
        # Ensure 2D (time × space)
        phi_field = np.asarray(phi_field)
        if phi_field.ndim != 2:
            raise ValueError(f"Expected 2D (time × space), got {phi_field.shape}")

        nx, ny = phi_field.shape  # nx = time, ny = space

        # --- Wrap into [-π, π)
        phi = np.angle(np.exp(1j * phi_field))

        # ============================================================
        # STEP 1: Spatial phase alignment (remove mean phase offsets)
        # ============================================================
        mid_idx = ny // 2
        phi_ref = phi[:, mid_idx]  # reference column

        # Compute average complex phase difference for each spatial position
        rel_phase = np.angle(np.exp(1j * (phi - phi_ref[:, None])))
        mean_diff = np.angle(np.nanmean(np.exp(1j * rel_phase), axis=0))

        # Subtract the mean offset for each column (align to mid position)
        phi_aligned = np.angle(np.exp(1j * (phi - mean_diff[None, :])))
        phi_aligned_list.append(phi_aligned)

        # ============================================================
        # STEP 2: Compute topological charge per (t,s) plaquette
        # ============================================================
        dphi_right = np.angle(np.exp(1j * (phi_aligned[1:, :-1] - phi_aligned[:-1, :-1])))
        dphi_up    = np.angle(np.exp(1j * (phi_aligned[1:, 1:]  - phi_aligned[1:, :-1])))
        dphi_left  = np.angle(np.exp(1j * (phi_aligned[:-1, 1:] - phi_aligned[1:, 1:])))
        dphi_down  = np.angle(np.exp(1j * (phi_aligned[:-1, :-1] - phi_aligned[:-1, 1:])))

        winding = dphi_right + dphi_up + dphi_left + dphi_down
        charges = np.rint(winding / (2 * np.pi)).astype(int)

        # ============================================================
        # STEP 3: Identify and clean up defects
        # ============================================================
        ii, jj = np.meshgrid(np.arange(charges.shape[0]),
                             np.arange(charges.shape[1]),
                             indexing="ij")
        pos_coords = np.column_stack([ii[charges == 1], jj[charges == 1]])
        neg_coords = np.column_stack([ii[charges == -1], jj[charges == -1]])

        

        # ============================================================
        # STEP 4: Store and visualize
        # ============================================================
        all_results.append(dict(
            charges=charges,
            pos_coords=pos_coords,
            neg_coords=neg_coords
        ))

        if plot:
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
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(phi_aligned,
                           cmap="hsv", aspect="auto",
                           origin="lower",
                           extent=[0, ny, 0, nx])
            ax.set_xlabel("Spatial position s")
            ax.set_ylabel("Time t (frames)")
            ax.set_title(f"Phase kymograph with topological defects (set {idx})")
            plt.colorbar(im, ax=ax, label="Phase [rad]")

            if len(pos_coords) > 0:
                ax.scatter(pos_coords[:, 1] + 0.5, pos_coords[:, 0] + 0.5,
                           marker="x", color="red", label="+1 defect (aster)")
            if len(neg_coords) > 0:
                ax.scatter(neg_coords[:, 1] + 0.5, neg_coords[:, 0] + 0.5,
                           marker="o", facecolors="none", edgecolors="blue",
                           label="−1 defect (anti-aster)")

            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.show()

    return {"all_results": all_results,
            "phi_aligned_list": phi_aligned_list,}

# ----------------------------------------------------------------------
# --- compute derived quantities for one group
# ----------------------------------------------------------------------

def compute_phase_quantities(group_folders, group_dir, plot=False):
    group_dir.mkdir(parents=True, exist_ok=True)

    all_A_global, all_omega_global = [], []
    all_phi_global = []
    local_phi_list, local_A_list, local_omega_list = [], [], []

    # --------------------------------------------------------------------------
    # Load data once
    # --------------------------------------------------------------------------
    for folder in group_folders:
        phi_file = folder / "phi_omega_a.npz"
        if not phi_file.exists():
            continue
        try:
            data = np.load(phi_file, allow_pickle=True)
            phi_g, omega_g, amp_g = data["global_phi_omega_a"]
            phi_l, omega_l, amp_l = data["local_phi_omega_a"]

            # collect
            all_phi_global.append(phi_g)
            all_omega_global.append(omega_g)
            all_A_global.append(amp_g)

            local_phi_list.append(phi_l)
            local_omega_list.append(omega_l)
            local_A_list.append(amp_l)

            del data  # free memory
        except Exception as e:
            print(f"⚠️ Skipping {folder.name}: {e}")



    # ---------- store each result ----------
    res_global = global_aniso(all_A_global, all_omega_global, plot=plot)
    save_npz_safe(group_dir / "global_aniso.npz", **res_global)

    res_gphase = global_aniso_phase(all_phi_global, all_omega_global, all_A_global, plot=plot)
    save_npz_safe(group_dir / "global_aniso_phase.npz", **res_gphase)

    res_corr_global = correlation_time_amp_omega(all_omega_global, all_A_global, plot=plot, max_dt=1000)
    save_npz_safe(group_dir / "correlation_time_global.npz", **res_corr_global)

    res_corr_local = correlation_time_amp_omega(local_omega_list, local_A_list, plot=plot)
    save_npz_safe(group_dir / "correlation_time_local.npz", **res_corr_local)

    res_local = local_aniso(local_omega_list, local_A_list, plot=plot)
    save_npz_safe(group_dir / "local_aniso.npz", **res_local)

    res_lphase = local_aniso_phase(local_phi_list, local_omega_list, local_A_list, plot=plot)
    save_npz_safe(group_dir / "local_aniso_phase.npz", **res_lphase)

    res_spat = spatial_correlation_local_phase(local_phi_list, plot=False)
    save_npz_safe(group_dir / "spatial_corr_phase.npz", **res_spat)

    res_temp = temporal_correlation_local_phase(local_phi_list, plot=plot)
    save_npz_safe(group_dir / "temporal_corr_phase.npz", **res_temp)

    res_def = phase_defects(local_phi_list, plot=plot)
    save_npz_safe(group_dir / "phase_defects.npz", **res_def)

# ----------------------------------------------------------------------
# --- load metadata only (no heavy arrays)
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# --- group metadata and process each group lazily
# ----------------------------------------------------------------------

def collect_grouped_results(summary_dir, output_root):
    meta_df = load_metadata(summary_dir)
    #print(meta_df)
    if meta_df.empty:
        print(f"No valid metadata found in {summary_dir}")
        return pd.DataFrame()

    expected_files = {
        "global_aniso.npz",
        "global_aniso_phase.npz",
        "correlation_time_global.npz",
        "correlation_time_local.npz",
        "local_aniso.npz",
        "local_aniso_phase.npz",
        "spatial_corr_phase.npz",
        "temporal_corr_phase.npz",
        "phase_defects.npz",
    }

    groups = list(meta_df.groupby(["Nmotor", "mu_a"]))
    for (Nmotor, mu_a), group_df in tqdm(groups, desc="Processing groups"):
        group_folders = [Path(p) for p in group_df["path"]]
        group_out = Path(output_root) / f"Nmotor_{Nmotor}_mu_{mu_a}"

        # --- Skip if all expected .npz files already exist ---
        if group_out.exists():
            existing = {f.name for f in group_out.glob("*.npz")}
            if expected_files.issubset(existing):
                print(f"✅ Skipping {group_out.name} (already complete)")
                #continue
                pass
        # Otherwise, compute and save
        # print(group_folders)
        compute_phase_quantities(group_folders, group_dir=group_out, plot=False)

def collect_experimental_results_grouped():
    """
    Group experimental segments by (genotype, ATP, KCL) using all_sharma_results.csv,
    then compute/save fluctuation results per group folder under OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT.
    """
    exp_root = Path(OUTPUT_DATA_SHARME_DIR)
    out_root = Path(OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = exp_root / "all_sharma_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Experimental CSV not found: {csv_path}")

    groups = _build_experiment_groups(exp_root, csv_path)
    if not groups:
        print("⚠️ No experimental groups to process.")
        return

    expected_files = {
        "global_aniso.npz",
        "global_aniso_phase.npz",
        "correlation_time_global.npz",
        "correlation_time_local.npz",
        "local_aniso.npz",
        "local_aniso_phase.npz",
        "spatial_corr_phase.npz",
        "temporal_corr_phase.npz",
        "phase_defects.npz",
    }

    for (genotype, ATP, KCL), group_folders in tqdm(groups.items(), desc="Processing experimental groups"):
        group_name = make_experiment_group_name(genotype, ATP, KCL)
        group_dir = out_root / group_name

        # Skip if complete
        if group_dir.exists():
            existing = {f.name for f in group_dir.glob("*.npz")}
            if expected_files.issubset(existing):
                # print(f"✅ Skipping {group_name} (already complete)")
                #continue
                pass
        # Compute using the same pipeline as simulations
        compute_phase_quantities(group_folders, group_dir=group_dir, plot=False)

def _compute_defect_rates(path: Path, dt: float, Ns: int, cutoff: float):
    """
    Load one phase_defects.npz, scale coords (t × dt × 1e3; s ÷ Ns × 1e1),
    annihilate close ± pairs (cutoff on scaled coords), and return
    mean ± SEM rates for + and − (per second), and number of valid runs.
    """
    if not path.exists():
        print(f"⚠️ Missing {path}")
        return dict(pos_mean=0.0, pos_sem=0.0, neg_mean=0.0, neg_sem=0.0, n_runs=0)

    res = np.load(path, allow_pickle=True)
    phi_list = list(res["phi_aligned_list"])
    all_results = list(res["all_results"])

    rates_pos_ds, rates_neg_ds = [], []

    for i, r in enumerate(all_results):
        # hard-fail if keys missing (avoid silent defaults)
        pos = np.array(r["pos_coords"], dtype=float)
        neg = np.array(r["neg_coords"], dtype=float)

        if pos.size == 0 and neg.size == 0:
            # no defects in this run; do not contribute
            continue

        phi_field = phi_list[i]  # (T, S)

        # --- Scale coordinates BEFORE culling pairs
        if pos.size:
            pos[:, 0] *= (dt * 1e3)           # 
            pos[:, 1] *= (1.0 / Ns) * 1e1     # space normalized
        if neg.size:
            neg[:, 0] *= (dt * 1e3)
            neg[:, 1] *= (1.0 / Ns) * 1e1

        # --- Remove close opposite-charge pairs (annihilations)
        if len(pos) > 0 and len(neg) > 0:
            pos_keep = np.ones(len(pos), dtype=bool)
            neg_keep = np.ones(len(neg), dtype=bool)
            dists = cdist(pos, neg)
            while True:
                i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
                dij = dists[i_min, j_min]
                if not np.isfinite(dij) or dij >= cutoff:
                    break
                pos_keep[i_min] = False
                neg_keep[j_min] = False
                dists[i_min, :] = np.inf
                dists[:, j_min] = np.inf
            pos = pos[pos_keep]
            neg = neg[neg_keep]

        # --- Filter defects by spatial window: keep only 1.5 ≤ s ≤ 7 ---
        if pos.size > 0:
            mask_pos = (pos[:, 1] >= 1.5) & (pos[:, 1] <= 7)
            pos = pos[mask_pos]

        if neg.size > 0:
            mask_neg = (neg[:, 1] >= 1.5) & (neg[:, 1] <= 7)
            neg = neg[mask_neg]


        # --- Total observed time (in seconds)
        T = float(phi_field.shape[0]) * dt 

        rates_pos_ds.append(len(pos) / T)
        rates_neg_ds.append(len(neg) / T)

    if len(rates_pos_ds) == 0:
        print(f"⚠️ No valid runs in {path} → using zero rate")
        return dict(pos_mean=0.0, pos_sem=0.0, neg_mean=0.0, neg_sem=0.0, n_runs=0)

    # mean ± SEM
    pos_mean = float(np.mean(rates_pos_ds))
    neg_mean = float(np.mean(rates_neg_ds))
    pos_sem = float(np.std(rates_pos_ds, ddof=1) / np.sqrt(len(rates_pos_ds))) if len(rates_pos_ds) > 1 else 0.0
    neg_sem = float(np.std(rates_neg_ds, ddof=1) / np.sqrt(len(rates_neg_ds))) if len(rates_neg_ds) > 1 else 0.0

    return dict(pos_mean=pos_mean, pos_sem=pos_sem,
                neg_mean=neg_mean, neg_sem=neg_sem,
                n_runs=len(rates_pos_ds))

def collect_phase_defect_rates_to_csv(
    output_csv: Path = PHASE_DEFECT_RATES_CSV,
    cutoff: float = 7.0,
):
    """
    Crawl extraction/new-parameter/experiment folders, compute +/− defect
    production rates per dataset, and save a tidy CSV used for fast plotting.

    Families included:
      - Extraction simulations near μ_a≈1570 at N0=20k and N0=100k (relN computed as (Nmotor*200)/N0)
      - New-parameter simulations near μ_a≈127 at N0=100 (relN computed as Nmotor/N0)
      - Experiment WT, ATP=750 μM (relN mapped from KCL)
    """
    rows = []

    # --- Helpers to enumerate dataset points (with relN) ---
    def parse_extraction_dirs(root, N0, mua0, label):
        for d in Path(root).iterdir():
            if not d.is_dir():
                continue
            m = re.match(r"Nmotor_(\d+(?:\.\d+)?)_mu_(\d+(?:\.\d+)?)$", d.name)
            if not m:
                continue
            n = float(m.group(1))
            mu = float(m.group(2))
            # follow your ratio-based family matching
            if abs(mu / n / 200 - mua0 / N0) > 1e-3:
                continue
            relN = (n * 200.0) / float(N0)
            if not (0.0 < relN <= 1.05):
                continue
            yield dict(
                scenario=label, relN=float(relN),
                path=d / "phase_defects.npz", dt=float(DT_SIMULATION), Ns=100
            )

    def parse_newparam_dirs(root, N0, color, label):
        for d in Path(root).iterdir():
            if not d.is_dir():
                continue
            m = re.match(r"Nmotor_(\d+(?:\.\d+)?)_mu_(\d+(?:\.\d+)?)$", d.name)
            if not m:
                continue
            n = float(m.group(1))*200
            relN = n  / float(N0)  
            if not (0.0 < relN <= 1.05):
                continue
            yield dict(
                scenario=label, color=color, relN=float(relN),
                path=d / "phase_defects.npz", dt=float(DT_SIMULATION/0.45), Ns=100
            )

    def parse_experiment_dirs(root, color, label):
        kcl_to_relN = {0: 1.00, 50: 0.96, 100: 0.93, 200: 0.87, 300: 0.80, 400: 0.74}
        for kcl, relN in kcl_to_relN.items():
            d = Path(root) / f"genotype_WT_ATP_750_KCL_{kcl}"
            yield dict(
                scenario=label, color=color, relN=float(relN),
                path=d / "phase_defects.npz", dt=0.001, Ns=24
            )

    # --- Build all dataset points ---
    # Extraction: μ_a≈1570, N0=17k and N0=100k
    for item in parse_extraction_dirs(OUTPUT_DATA_FLUCTUATIONS_EXTRACTION, N0=17000,  mua0=1570, label="Sim N₀=17k, μₐ≈1570"):
        rows.append(item)
    for item in parse_extraction_dirs(OUTPUT_DATA_FLUCTUATIONS_EXTRACTION, N0=100000, mua0=1570, label="Sim N₀=100k, μₐ≈1570"):
        rows.append(item)

    # New parameter: 
    for item in parse_newparam_dirs(OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER,  N0=85*200, color="tab:blue", label="Sim new parameter"):
        rows.append(item)

    # Experiment: WT, ATP=750 μM
    for item in parse_experiment_dirs(OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT, color="red", label="Exp (WT, ATP=750 μM)"):
        rows.append(item)

    # --- Compute rates for each point and assemble tidy table ---
    out_records = []
    for item in tqdm(rows, desc="Computing defect rates", leave=False):
        # if item["relN"]<0.999:
        #     continue
        stats = _compute_defect_rates(item["path"], item["dt"], item["Ns"], cutoff=cutoff)
        out_records.append({
            "scenario": item["scenario"],
            "relN": item["relN"],
            "pos_mean": stats["pos_mean"],
            "pos_sem":  stats["pos_sem"],
            "neg_mean": stats["neg_mean"],
            "neg_sem":  stats["neg_sem"],
            "n_runs":   stats["n_runs"],
            "dt":       item["dt"],
            "Ns":       item["Ns"],
            "cutoff":   float(cutoff),
            "source":   str(item["path"])
        })

    df_out = pd.DataFrame(out_records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"✓ Saved phase defect rates table to: {output_csv}")

def _fit_corr_length_from_npz(npz_path: Path, Ns: int, ds_norm_lo=0.2, ds_norm_hi=0.5):
    """
    Load spatial_corr_phase.npz and fit ln C = a + b * (Δs/L) separately
    for each realization (row in angular_corr_all).

    Correlation length ξ := -1/b. Returns mean ± SEM across realizations.

    Returns
    -------
    xi_mean : float
        Mean correlation length across realizations.
    xi_sem : float
        Standard error of the mean correlation length.
    n_used : int
        Number of realizations successfully fitted.
    """
    if not npz_path.exists():
        print(f"⚠️ Missing {npz_path}")
        return np.nan, np.nan, 0

    data = np.load(npz_path, allow_pickle=True)
    ds = np.asarray(data["ds_axis"], float)
    corr_raw = data["angular_corr_all"]

    # Convert to list of float arrays, skipping non-numeric entries
    corr_list = []
    for c in corr_raw:
        try:
            arr = np.asarray(c, dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                continue
            corr_list.append(arr)
        except Exception:
            continue

    if not corr_list:
        print(f"⚠️ No valid correlation arrays in {npz_path}")
        return np.nan, np.nan, 0

    # Pad to same length (ragged -> rectangular)
    max_len = max(len(c) for c in corr_list)
    corr_mat = np.full((len(corr_list), max_len), np.nan)
    for i, c in enumerate(corr_list):
        corr_mat[i, :len(c)] = c

    n_real, n_ds = corr_mat.shape
    ds_norm = ds[:n_ds] / float(Ns)
    win_mask = (ds_norm >= ds_norm_lo) & (ds_norm <= ds_norm_hi)

    xi_list = []

    for c in corr_mat:
        if not np.any(np.isfinite(c)):
            continue

        x = ds_norm[win_mask & np.isfinite(c)]
        y = c[win_mask & np.isfinite(c)]
        y = y[y > 0]
        if x.size < 2 or y.size != x.size:
            continue

        ly = np.log(y)
        try:
            b, a = np.polyfit(x, ly, 1)
        except Exception:
            continue

        if b >= 0:  # non-decaying
            continue

        xi_list.append(-1.0 / b)

    n_used = len(xi_list)
    if n_used == 0:
        return np.nan, np.nan, 0

    xi_arr = np.array(xi_list, float)
    xi_mean = float(np.nanmean(xi_arr))
    xi_sem = float(np.nanstd(xi_arr, ddof=1) / np.sqrt(n_used)) if n_used > 1 else 0.0

    return xi_mean, xi_sem, n_used

def collect_corr_length_to_csv(
    output_csv: Path,
    ds_norm_lo: float = 0.2,
    ds_norm_hi: float = 0.5,
):
    """
    Scan all relevant folders, compute correlation lengths, and save a tidy CSV:

    CSV header:
        scenario,relN,corr_len,corr_len_sem,n_points,Ns,source,extra

    Where:
      - scenario: descriptive label used by plotting
      - relN: relative motor extraction N/N0 (unitless fraction)
      - corr_len: fitted correlation length ξ from ln C vs Δs/L
      - corr_len_sem: SEM for ξ via error propagation
      - n_points: number of points used in the fit window
      - Ns: number of spatial bins (100 for sim, 24 for exp)
      - source: one of {"extraction","new_parameter","experiment"}
      - extra: auxiliary tag (e.g., folder name)

    Families included:
      - Extraction, μₐ≈1570 with N0={20k,100k} (Nmotor_*_mu_*)
      - New-parameter, μₐ≈127 with N0=100
      - Experiment WT, ATP=750 with KCL→relN mapping
    """
    rows = []

    # ---------- Helpers for parsing ----------
    def parse_extraction(root: Path, N0: int, mua0: float):
        # collect family along constant N/mu ratio (scaled by 200 for extraction)
        tol = 1e-3
        for d in root.iterdir():
            if not d.is_dir():
                continue
            m = re.match(r"Nmotor_([0-9]+(?:\.[0-9]+)?)_mu_([0-9]+(?:\.[0-9]+)?)$", d.name)
            if not m:
                continue
            n  = float(m.group(1))        # stored as "Nmotor_{n}"
            mu = float(m.group(2))
            # ratio test
            if abs(mu / (n * 200.0) - mua0 / float(N0)) > tol:
                continue
            relN = (n * 200.0) / float(N0)
            if not (0.0 < relN <= 1.05):
                continue
            yield d, relN

    def parse_newparam(root: Path, N0: int):
        for d in root.iterdir():
            if not d.is_dir():
                continue
            m = re.match(r"Nmotor_(\d+(?:\.\d+)?)_mu_(\d+(?:\.\d+)?)$", d.name)
            if not m:
                continue
            n = float(m.group(1)) * 200
            relN = n / float(N0)
            if not (0.0 < relN <= 1.05):
                continue
            yield d, relN

    def parse_experiment(root: Path):
        # WT, ATP=750 only; KCL→relN map
        kcl_to_relN = {0: 1.00, 50: 0.96, 100: 0.93, 200: 0.87, 300: 0.80, 400: 0.74}
        for kcl, relN in kcl_to_relN.items():
            d = root / f"genotype_WT_ATP_750_KCL_{kcl}"
            yield d, relN

    # ---------- Extraction families (μₐ≈1570) ----------
    for N0, color_tag in [(17000, "Sim N₀=17k, μₐ≈1570"), (100000, "Sim N₀=100k, μₐ≈1570")]:
        for d, relN in parse_extraction(OUTPUT_DATA_FLUCTUATIONS_EXTRACTION, N0=N0, mua0=1570.0):
            npz = d / "spatial_corr_phase.npz"
            if not npz.exists():
                print(f"⚠️ Missing {npz}")
                continue
            xi, xi_sem, npts = _fit_corr_length_from_npz(npz, Ns=100, ds_norm_lo=ds_norm_lo, ds_norm_hi=ds_norm_hi)
            rows.append({
                "scenario": color_tag,
                "relN": float(relN),
                "corr_len": xi,
                "corr_len_sem": xi_sem,
                "n_points": int(npts),
                "Ns": 100,
                "source": "extraction",
                "extra": d.name,
            })

    # ---------- New-parameter family  ----------
    for d, relN in parse_newparam(OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER, N0=85*200):
        npz = d / "spatial_corr_phase.npz"
        if not npz.exists():
            print(f"⚠️ Missing {npz}")
            continue
        xi, xi_sem, npts = _fit_corr_length_from_npz(npz, Ns=100, ds_norm_lo=ds_norm_lo, ds_norm_hi=ds_norm_hi)
        rows.append({
            "scenario": "Sim new parameter",
            "relN": float(relN),
            "corr_len": xi,
            "corr_len_sem": xi_sem,
            "n_points": int(npts),
            "Ns": 100,
            "source": "new_parameter",
            "extra": d.name,
        })

    # ---------- Experiment WT, ATP=750 ----------
    for d, relN in parse_experiment(OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT):
        npz = d / "spatial_corr_phase.npz"
        if not npz.exists():
            print(f"⚠️ Missing {npz}")
            continue
        xi, xi_sem, npts = _fit_corr_length_from_npz(npz, Ns=24, ds_norm_lo=ds_norm_lo, ds_norm_hi=ds_norm_hi)
        rows.append({
            "scenario": "Exp (WT, ATP=750 μM)",
            "relN": float(relN),
            "corr_len": xi,
            "corr_len_sem": xi_sem,
            "n_points": int(npts),
            "Ns": 24,
            "source": "experiment",
            "extra": d.name,
        })

    # ---------- Save CSV ----------
    df = pd.DataFrame(rows, columns=[
        "scenario","relN","corr_len","corr_len_sem","n_points","Ns","source","extra"
    ])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved correlation-length table to: {output_csv}")

# ----------------------------------------------------------------------
# --- main entry point
# ----------------------------------------------------------------------

def main():
    collect_grouped_results(OUTPUT_DATA_EXTRACTION_DIR, OUTPUT_DATA_FLUCTUATIONS_EXTRACTION)
    collect_grouped_results(OUTPUT_DATA_NEW_PARAMETER_DIR, OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER)
    collect_experimental_results_grouped()

    collect_corr_length_to_csv(SPATIAL_CORRELATION_CSV)
    collect_phase_defect_rates_to_csv()
    

if __name__ == "__main__":
    main()
