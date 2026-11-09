import numpy as np
from scipy.spatial import cKDTree

def _wrap_pm_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle differences to (-pi, pi] in a vectorized, stable way."""
    return np.angle(np.exp(1j * x))

def detect_phase_defects_2d(
    phase: np.ndarray,
    *,
    require_min_abs_winding: float = np.pi,
):
    """
    Detect phase defects via plaquette winding.

    Parameters
    ----------
    phase : (Nt, Ns) array
        Phase field (can be wrapped; we only use wrapped differences).
    require_min_abs_winding : float
        Optional gate to suppress numerical noise; pi is a good default.

    Returns
    -------
    defects : (N, 3) int array with columns [t_idx, s_idx, charge]
        t_idx,s_idx refer to plaquette lower-left corner (0<=t<Nt-1, 0<=s<Ns-1)
    w : (Nt-1, Ns-1) float array
        Plaquette winding sums (in radians).
    """
    phi = np.asarray(phase, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"phase must be 2D (Nt,Ns); got {phi.shape}")
    Nt, Ns = phi.shape
    if Nt < 2 or Ns < 2:
        return np.zeros((0, 3), dtype=int), np.zeros((max(Nt - 1, 0), max(Ns - 1, 0)), dtype=float)

    # Wrapped forward diffs
    d_t = _wrap_pm_pi(phi[1:, :] - phi[:-1, :])     # (Nt-1, Ns)
    d_s = _wrap_pm_pi(phi[:, 1:] - phi[:, :-1])     # (Nt,   Ns-1)

    # Plaquette winding (Nt-1, Ns-1)
    w = d_t[:, :-1] + d_s[1:, :] - d_t[:, 1:] - d_s[:-1, :]

    # Integer charge
    q = -np.rint(w / (2.0 * np.pi)).astype(int)

    # Keep only nonzero charges, plus an optional magnitude gate
    if require_min_abs_winding is not None:
        mask = (q != 0) & (np.abs(w) >= float(require_min_abs_winding))
    else:
        mask = (q != 0)

    t_idx, s_idx = np.nonzero(mask)
    charge = q[mask]

    defects = np.column_stack([t_idx.astype(int), s_idx.astype(int), charge.astype(int)])
    return defects, w


def merge_opposite_charges(
    defects: np.ndarray,
    *,
    theta_t: int,
    theta_s: int,
):
    """
    Annihilate opposite-sign pairs that are close in BOTH t and s:
        |dt| <= theta_t and |ds| <= theta_s

    Uses KD-tree (euclidean in scaled space) + exact axis-aligned filtering.
    Greedy nearest-neighbor matching from positives to negatives.

    Returns
    -------
    merged : (M,3) int array
    """
    D = np.asarray(defects, dtype=int)
    if D.size == 0:
        return D.reshape(0, 3)

    pos = D[D[:, 2] > 0]
    neg = D[D[:, 2] < 0]

    # keep any weird charges (|q|>1) untouched
    other = D[np.abs(D[:, 2]) != 1]

    if pos.size == 0 or neg.size == 0:
        return np.vstack([pos, neg, other]).reshape(-1, 3)

    theta_t = max(int(theta_t), 1)
    theta_s = max(int(theta_s), 1)

    # KDTree on negatives in scaled coords
    neg_xy = np.column_stack([neg[:, 0] / theta_t, neg[:, 1] / theta_s]).astype(float)
    tree = cKDTree(neg_xy)

    pos_xy = np.column_stack([pos[:, 0] / theta_t, pos[:, 1] / theta_s]).astype(float)

    # Euclidean radius that covers the axis-aligned box in scaled space
    r = np.sqrt(2.0)

    neg_available = np.ones(len(neg), dtype=bool)
    pos_kill = np.zeros(len(pos), dtype=bool)
    neg_kill = np.zeros(len(neg), dtype=bool)

    # Greedy: for each positive, match the closest still-available negative in the box
    for i in range(len(pos)):
        cand = tree.query_ball_point(pos_xy[i], r=r)
        if not cand:
            continue

        # keep only negatives not yet used
        cand = [j for j in cand if neg_available[j]]
        if not cand:
            continue

        dt = np.abs(neg[cand, 0] - pos[i, 0])
        ds = np.abs(neg[cand, 1] - pos[i, 1])

        ok = (dt <= theta_t) & (ds <= theta_s)
        if not np.any(ok):
            continue

        cand_ok = np.array(cand, dtype=int)[ok]

        # choose nearest in scaled euclidean distance (good tie-breaker)
        d2 = ((neg[cand_ok, 0] - pos[i, 0]) / theta_t) ** 2 + ((neg[cand_ok, 1] - pos[i, 1]) / theta_s) ** 2
        j = cand_ok[np.argmin(d2)]

        pos_kill[i] = True
        neg_kill[j] = True
        neg_available[j] = False

    pos_keep = pos[~pos_kill]
    neg_keep = neg[~neg_kill]

    merged = np.vstack([pos_keep, neg_keep, other]).reshape(-1, 3)
    return merged


def filter_defects_by_quality(
    defects: np.ndarray,
    *,
    msd_dphi_norm_s: np.ndarray,
    amp_p01_s: np.ndarray,
    msd_thr: float,
    amp_thr: float,
):
    D = np.asarray(defects, dtype=int)

    msd = np.asarray(msd_dphi_norm_s, dtype=float)
    amp = np.asarray(amp_p01_s, dtype=float)

    good_s_mask = (msd <= float(msd_thr)) & (amp >= float(amp_thr))

    good_indices = np.nonzero(good_s_mask)[0]
    if len(good_indices) == 0:
        return np.zeros((0, 3), dtype=int), (-1, -1), good_s_mask

    s_min = int(good_indices[0])
    s_max = int(good_indices[-1])

    if D.size == 0:
        return np.zeros((0, 3), dtype=int), (s_min, s_max), good_s_mask

    s_idx = D[:, 1]

    keep = (
        (s_idx >= s_min)
        & (s_idx <= s_max)
        & good_s_mask[s_idx]
    )

    return D[keep], (s_min, s_max), good_s_mask

def detect_and_clean_defects(
    phase: np.ndarray,
    aux: dict,
    *,
    theta_t: float = 0.5,
    theta_s: float = 0.1,
    msd_thr: float = 5.0,
    amp_thr: float = 0.05,
    require_min_abs_winding: float = np.pi,
    exclude_margin_s: int = 1,
    exclude_margin_t: int = 2,
    dt: float = 0.1,
):  
    if phase.ndim != 2:
        raise ValueError(f"phase must be 2D (Nt,Ns); got {phase.shape}")
    
    if np.isnan(phase).any():
        Nt, Ns = phase.shape

        empty_int = np.zeros((0, 3), dtype=np.int64)

        return {
            "raw": empty_int.copy(),
            "merged": empty_int.copy(),
            "clean": empty_int.copy(),
            "good_s_interval": np.array([0, 0], dtype=np.int64),
            "effective_Nt": 0,
            "effective_T": np.nan,  
        }
    
    raw, w = detect_phase_defects_2d(
        phase,
        require_min_abs_winding=require_min_abs_winding
    )

    Nt, Ns = phase.shape

     
    # Relative → absolute thresholds
    theta_s_abs = max(int(theta_s * Ns), 1)

    # --- DFT-based period estimation ---
    z = np.exp(1j * phase)
    Z = np.fft.fft(z, axis=0)
    PSD = np.abs(Z) ** 2
    PSD_sum = PSD.sum(axis=1)
    freqs = np.fft.fftfreq(Nt)
    pos = freqs > 0
    PSD_sum_pos = PSD_sum[pos]
    if len(PSD_sum_pos) == 0:
        period = Nt
    else:
        k_peak = np.argmax(PSD_sum_pos)
        k_peak = np.where(pos)[0][k_peak]
        freq = k_peak / Nt
        period = 1.0 / freq if freq > 0 else Nt

    theta_t_abs = max(int(theta_t * period), 1)

    merged = merge_opposite_charges(
        raw,
        theta_t=theta_t_abs,
        theta_s=theta_s_abs
    )

    # Exclude temporal + spatial margins
    if merged.size > 0:
        t_idx = merged[:, 0]
        s_idx = merged[:, 1]

        mask = (
            (s_idx >= exclude_margin_s)
            & (s_idx < (Ns - 1 - exclude_margin_s))
            & (t_idx >= exclude_margin_t)
            & (t_idx < (Nt - 1 - exclude_margin_t))
        )

        merged = merged[mask]

    # Effective kept intervals
    s_min_keep = exclude_margin_s
    s_max_keep = Ns - 1 - exclude_margin_s

    t_min_keep = exclude_margin_t
    t_max_keep = Nt - 1 - exclude_margin_t

    Nt_eff = max(t_max_keep - t_min_keep, 0)

     
    # 2) Quality filtering
     
    msd = aux.get("msd_dphi_norm_s", None)
    amp = aux.get("amp_p01_s", None)

    if msd is not None and amp is not None:
        cleaned, good_interval, _ = filter_defects_by_quality(
            merged,
            msd_dphi_norm_s=msd,
            amp_p01_s=amp,
            msd_thr=msd_thr,
            amp_thr=amp_thr,
        )

        if good_interval[0] < 0:
            good_s_interval = np.array([-1, -1], dtype=int)
        else:
            good_s_interval = np.array(good_interval, dtype=int)

    else:
        cleaned = merged
        good_s_interval = np.array([s_min_keep, s_max_keep], dtype=int)

    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(6, 4))

    # # kymograph
    # im = ax.imshow(
    #     phase.T,
    #     aspect="auto",
    #     origin="lower",
    #     cmap="twilight",
    #     interpolation="nearest",
    # )

    # plt.colorbar(im, ax=ax, label="phase")
    # # plot defects
    # if cleaned.size > 0:
    #     t = cleaned[:, 0]
    #     s = cleaned[:, 1]
    #     q = cleaned[:, 2]
    #     pos = q > 0
    #     neg = q < 0
    #     # positive → circle
    #     ax.scatter(
    #         t[pos],
    #         s[pos],
    #         marker="o",
    #         facecolors="none",
    #         edgecolors="white",
    #         linewidths=1.5,
    #         label="positive",
    #     )
    #     # negative → cross
    #     ax.scatter(
    #         t[neg],
    #         s[neg],
    #         marker="x",
    #         color="black",
    #         linewidths=1.5,
    #         label="negative",
    #     )
    # # good interval
    # if good_s_interval[0] >= 0:
    #     ax.axhline(good_s_interval[0], color="green", linestyle="--", alpha=0.7)
    #     ax.axhline(good_s_interval[1], color="green", linestyle="--", alpha=0.7)
    # ax.set_xlabel("time index")
    # ax.set_ylabel("s index")
    # ax.set_title("Phase kymograph with defects")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    return {
        "raw": raw,
        "merged": merged,
        "clean": cleaned,
        "good_s_interval": good_s_interval,
        "effective_Nt": Nt_eff,
        "effective_T": Nt_eff * dt,
    }

if __name__ == "__main__":
    import numpy as np

    # --- synthetic grid ---
    Nt, Ns = 80, 80
    t = np.arange(Nt)
    s = np.arange(Ns)
    T, S = np.meshgrid(t, s, indexing="ij")

     
    #  Isolated +1 defect
     
    t0p1, s0p1 = 20, 20
    phi_p1 = np.arctan2(T - t0p1, S - s0p1)

     
    #  Isolated -1 defect
     
    t0m1, s0m1 = 60, 60
    phi_m1 = -np.arctan2(T - t0m1, S - s0m1)

     
    #  Close opposite pair (should annihilate)
     
    t0p2, s0p2 = 40, 40
    t0m2, s0m2 = 42, 41   # close in both t and s

    phi_p2 = np.arctan2(T - t0p2, S - s0p2)
    phi_m2 = -np.arctan2(T - t0m2, S - s0m2)

     
    # Superpose and wrap
     
    phase = np.angle(np.exp(1j * (phi_p1 + phi_m1 + phi_p2 + phi_m2)))

     
    # Fake diagnostics (clean everywhere)
     
    aux = {
        "msd_dphi_norm_s": np.zeros(Ns),
        "amp_p01_s": np.ones(Ns),
    }

    print("Running defect detection...")

    result = detect_and_clean_defects(
        phase,
        aux,
        theta_t=3,   # large enough to merge close pair
        theta_s=3,
        msd_thr=5.0,
        amp_thr=0.1,
    )

    print("\nRAW defects:")
    print(result["raw"])

    print("\nMERGED defects (close pair should vanish):")
    print(result["merged"])

    print("\nCLEAN defects:")
    print(result["clean"])

    print("\nExpected:")
    print("One +1 near (20,20)")
    print("One -1 near (60,60)")
    print("Close pair near (40,40) should annihilate")