from pymatreader import read_mat
import numpy as np
import os
import hashlib
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

from cilia.datastructures.DataName import TANGENT_ANGLE_SEGMENTS
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datasets.DataItem import DataItem
from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SEGMENTS,
    ARCLENGTH,
    DT_FRAME,
    CONDITION_DESCRIPTION,
    A,F, Q
)
from cilia.datastructures.special_source_names import ORIGINAL, GIVEN

def reconstruct_curve_from_phi(phi_spline, s_grid, x0, y0):
    """Integrate tangent angle spline over s_grid to get curve (x(s),y(s))."""
    phi_vals = phi_spline(s_grid)
    ds_local = np.diff(s_grid)
    tx = np.cos(phi_vals)
    ty = np.sin(phi_vals)

    dx = tx[:-1] * ds_local
    dy = ty[:-1] * ds_local

    x = np.concatenate(([0.0], np.cumsum(dx))) + x0
    y = np.concatenate(([0.0], np.cumsum(dy))) + y0
    return x, y


def choose_best_orientation(x, y, psi_frame, ds, _arclength_param, show_plot=False):
    # --- base reconstruction from ψ ---
    dx = np.cos(psi_frame) * ds
    dy = np.sin(psi_frame) * ds

    x_rec_base = np.concatenate(([0], np.cumsum(dx)))
    y_rec_base = np.concatenate(([0], np.cumsum(dy)))

    cases = []

    for reverse_s in [False, True]:
        for flip in [False, True]:

            x_obs = x.copy()
            y_obs = y.copy()

            if flip:
                x_obs = -x_obs
                y_obs = -y_obs

            if reverse_s:
                x_obs = x_obs[::-1]
                y_obs = y_obs[::-1]

            # arc-length parameter
            s_obs = _arclength_param(x_obs, y_obs)
            u_obs = s_obs / s_obs[-1]

            # anchor reconstruction to first observed point
            x_rec = x_rec_base + x_obs[0]
            y_rec = y_rec_base + y_obs[0]

            s_rec = _arclength_param(x_rec, y_rec)
            u_rec = s_rec / s_rec[-1]

            interp_x = interp1d(u_rec, x_rec, bounds_error=False, fill_value="extrapolate") # type: ignore
            interp_y = interp1d(u_rec, y_rec, bounds_error=False, fill_value="extrapolate") # type: ignore

            x_interp = interp_x(u_obs)
            y_interp = interp_y(u_obs)

            residual = np.sqrt((x_obs - x_interp)**2 +
                               (y_obs - y_interp)**2)

            rms = np.sqrt(np.mean(residual**2))
            max_res = np.max(residual)

            cases.append({
                "reverse_s": reverse_s,
                "flip": flip,
                "x_obs": x_obs,
                "y_obs": y_obs,
                "residual": residual,
                "s_obs": s_obs,
                "rms": rms,
                "max_res": max_res
            })
            if show_plot:
                plt.scatter(x_obs,y_obs)
                plt.plot(x_interp,y_interp)
                plt.show()

    best_case = min(cases, key=lambda c: c["rms"])

    return (
        best_case["x_obs"],
        best_case["y_obs"],
        best_case["max_res"],
        best_case,
        cases
    )

def _valid_frame(frame):
    # Remove NaN rows
    mask = ~np.isnan(frame).any(axis=1)
    frame = frame[mask]
    if len(frame) == 0:
        return frame

    tol = 1e-12
    # Remove leading (0,0) points
    start = 0
    while start < len(frame) and np.all(np.abs(frame[start]) < tol):
        start += 1

    # Remove trailing (0,0) points
    end = len(frame)
    while end > start and np.all(np.abs(frame[end - 1]) < tol):
        end -= 1

    return frame[start:end]

def _arclength_param(x, y):
    ds_local = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate(([0], np.cumsum(ds_local)))

def _smooth_psi(x_obs, y_obs,psi_frame,ds):
    s_real = _arclength_param(x_obs, y_obs)
    L_real = s_real[-1]

    phi_use = np.unwrap(psi_frame.copy())

    # Build ψ arc-length grid from ds
    if np.isscalar(ds):
        s_psi = np.arange(len(phi_use)) * ds # type: ignore
    else:
        s_psi = np.concatenate(([0], np.cumsum(ds[:-1])))

    s_psi -= s_psi[0] # type: ignore
    s_psi = s_psi / s_psi[-1] * L_real

    # Fine integration grid
    Nfine = 1000
    s_fine = np.linspace(0.0, L_real, Nfine)

    Nctrl = 51
    s_ctrl = np.linspace(0.0, L_real, Nctrl)

    phi_init_ctrl = interp1d(
        s_psi,
        phi_use,
        kind="linear",
        fill_value="extrapolate" # type: ignore
    )(s_ctrl)

    lam1 = 1e-2
    lam2 = 1e-2

    # Precompute spline basis matrices

    Nfine = len(s_fine)
    Nctrl = len(s_ctrl)
    A  = np.zeros((Nfine, Nctrl))
    A1 = np.zeros((Nfine, Nctrl))
    A2 = np.zeros((Nfine, Nctrl))
    for j in range(Nctrl):
        basis = np.zeros(Nctrl)
        basis[j] = 1.0
        spl = CubicSpline(s_ctrl, basis, bc_type="natural")
        A[:, j]  = spl(s_fine)
        A1[:, j] = spl(s_fine, 1)
        A2[:, j] = spl(s_fine, 2)

    ds_fine = np.diff(s_fine)

    def objective_with_grad(phi_ctrl):
        # Evaluate spline on fine grid
        phi_f = A @ phi_ctrl
        cos_phi = np.cos(phi_f)
        sin_phi = np.sin(phi_f)
        # Reconstruct curve
        dx = cos_phi[:-1] * ds_fine
        dy = sin_phi[:-1] * ds_fine

        x_rec = np.concatenate(([x_obs[0]], x_obs[0] + np.cumsum(dx)))
        y_rec = np.concatenate(([y_obs[0]], y_obs[0] + np.cumsum(dy)))

        # KDTree once
        tree = cKDTree(np.c_[x_rec, y_rec])
        dists, idx = tree.query(np.c_[x_obs, y_obs], k=1)

        # DATA TERM VALUE
        data_term = np.trapz(dists**2, s_real)

        # PRECOMPUTE derivative of x_k, y_k wrt ALL controls
        Sx = -sin_phi[:-1][:, None] * ds_fine[:, None] * A[:-1]
        Sy =  cos_phi[:-1][:, None] * ds_fine[:, None] * A[:-1]

        dX = np.vstack([np.zeros((1, Nctrl)),
                        np.cumsum(Sx, axis=0)])

        dY = np.vstack([np.zeros((1, Nctrl)),
                        np.cumsum(Sy, axis=0)])

        grad = np.zeros(Nctrl)

        # DATA TERM GRADIENT
        for n in range(len(x_obs)):
            k = idx[n]

            ex = x_obs[n] - x_rec[k]
            ey = y_obs[n] - y_rec[k]

            tx = cos_phi[k]
            ty = sin_phi[k]

            dot = ex*tx + ey*ty

            rx = ex - dot*tx
            ry = ey - dot*ty

            dqx = dX[k]
            dqy = dY[k]

            dphi = A[k]
            dtx = -sin_phi[k] * dphi
            dty =  cos_phi[k] * dphi

            term1 = -2.0 * (rx * dqx + ry * dqy)
            term2 = -2.0 * dot * (ex*dtx + ey*dty)

            grad += term1 + term2

        # REGULARIZATION
        phi_p  = A1 @ phi_ctrl
        phi_pp = A2 @ phi_ctrl

        w = np.ones_like(s_fine)
        w[0] *= 0.5
        w[-1] *= 0.5
        w *= (s_fine[1] - s_fine[0])

        reg1 = np.sum(w * phi_p**2)
        reg2 = np.sum(w * phi_pp**2)

        grad += 2 * lam1 * (A1.T @ (w * phi_p))
        grad += 2 * lam2 * (A2.T @ (w * phi_pp))

        total_value = data_term + lam1*reg1 + lam2*reg2

        return total_value, grad


    res = minimize(
        objective_with_grad,
        phi_init_ctrl,
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=200, ftol=1e-5)
    )

    return res

def process(entry,i,j, file):
    xy = entry["xy_in_micron"]
    psi = entry["tangent_angle_psi_in_rad"]
    ds = entry["ds_in_micron"]

    print("\n--- PROCESS DEBUG ---")
    print("xy shape:", xy.shape)

    n_frames = xy.shape[0]
    segments = []
    current_segment = []

    # Iterate over frames
    pbar = tqdm(total=n_frames, desc=f"Processing {file} [{i},{j}]", leave=False)

    f = 0
    while f < n_frames:
        frame = _valid_frame(xy[f])

        if len(frame) == 0:
            f += 1
            pbar.update(1)
            continue

        # Duplicate detection
        if f < n_frames - 1:
            frame_next = _valid_frame(xy[f + 1])
            if frame.shape == frame_next.shape and \
            np.allclose(frame, frame_next, atol=1e-12):
                print(f"\n⚠ Duplicate detected at f={f} and f={f+1}")
                # store current segment
                if len(current_segment) > 0:
                    segment_array = np.vstack(current_segment)  # shape (n_frames_segment, 51)
                    segments.append(segment_array)
                    print(f"Segment stored with shape {segment_array.shape}")
                current_segment = []
                # skip all consecutive duplicates safely
                while f < n_frames - 1:
                    frame_curr = _valid_frame(xy[f])
                    frame_next = _valid_frame(xy[f + 1])
                    if frame_curr.shape != frame_next.shape:
                        break
                    if not np.allclose(frame_curr, frame_next, atol=1e-12):
                        break
                    f += 1
                    pbar.update(1)
                f += 1
                pbar.update(1)
                continue

        frame = _valid_frame(xy[f])
        if len(frame) == 0:
            f += 1
            pbar.update(1)
            continue
        x = frame[:, 0]
        y = frame[:, 1]
        psi_frame = psi[f]

        s_real = _arclength_param(x, y)
        L_real = s_real[-1]
        x_obs, y_obs, max_residual, best_case, cases = choose_best_orientation(
            x, y, psi_frame, L_real/psi_frame.shape[0], _arclength_param
        )

        if max_residual > 2.5:
            print("❌ Residual exceeds 2.5 µm — terminating segment")
            print(f,i,j, file)
            if len(current_segment) > 0:
                segment_array = np.vstack(current_segment)  # shape (n_frames_segment, 51)
                segments.append(segment_array)
                print(f"Segment stored with shape {segment_array.shape}")
            current_segment = []
            f += 1
            pbar.update(1)
            continue


        res = _smooth_psi(x_obs,y_obs,psi_frame,ds)
    
        if res.fun > 0.1:
            print("❌ Smoothed fit max residual > 0.1 µm — terminating segment")
            if len(current_segment) > 0:
                segment_array = np.vstack(current_segment)  # shape (n_frames_segment, 51)
                segments.append(segment_array)
                print(f"Segment stored with shape {segment_array.shape}")
            current_segment = []

        phi_opt = res.x
        current_segment.append(phi_opt.copy())

        # ------------------------------------------------------------
        # Debug plot
        # ------------------------------------------------------------

        # fig, axes = plt.subplots(1, 2, figsize=(14,6))

        # phi_linear = interp1d(
        #     s_psi,
        #     phi_use,
        #     kind="linear",
        #     fill_value="extrapolate"
        # )

        # x_init, y_init = reconstruct_curve_from_phi(
        #     phi_linear,
        #     s_fine,
        #     x_obs[0],
        #     y_obs[0]
        # )

        # # Geometry
        # axes[0].plot(x_obs, y_obs, label="Observed (oriented)", linewidth=3)
        # axes[0].scatter(x_obs, y_obs, s=10)

        # axes[0].plot(x_init, y_init, color='gray', linewidth=2,
        #              label="Initial φ (linear)")

        # axes[0].plot(x_smooth, y_smooth, color='black', linewidth=3,
        #              label="Smoothed result")

        # axes[0].set_aspect("equal")
        # axes[0].set_title(f"Frame {f}")
        # axes[0].legend()

        # # Residual
        # axes[1].plot(s_real, d_opt, color='black', linewidth=2)
        # axes[1].axhline(0.5, linestyle='--')
        # axes[1].set_xlabel("Arc-length (µm)")
        # axes[1].set_ylabel("Residual (µm)")
        # axes[1].set_title("Residual along filament")

        # plt.tight_layout()
        # plt.show()

        f += 1
        pbar.update(1)

    # -------------------------------------------------------
    # Store last segment
    # -------------------------------------------------------
    if len(current_segment) > 0:
        segment_array = np.vstack(current_segment)  # shape (n_frames_segment, 51)
        segments.append(segment_array)
        print(f"Segment stored with shape {segment_array.shape}")


    print(f"\nTotal segments found: {len(segments)}")

    return segments

class SharmaDataset(DataSet):

    def __init__(self, path: str):
        super().__init__(path)

    def _initialize_realizations(self):
        """
        Automatically load existing realizations
        from disk if dataset already exists.
        """
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

from local_config import CILIA_FOLDER

def main():
    dataset_path = os.path.join(CILIA_FOLDER,"structured/sharma")
    original_path = os.path.join(CILIA_FOLDER,"original/sharma")

    dataset = SharmaDataset(dataset_path)
    existing_ids = {r.experiment_id for r in dataset.realizations}

    file_list = [f for f in os.listdir(original_path)
                 if f.endswith("_master.mat")]

    for file in file_list:
        data = read_mat(os.path.join(original_path, file))
        master = data["Master"]

        for i in range(len(master)):
            for j in range(len(master[i])):

                entry = master[i][j]
                if not isinstance(entry, dict):
                    continue

                xy = entry.get("xy_in_micron", None)
                if not isinstance(xy, np.ndarray):
                    continue

                # Build checksum → experiment_id
                xy_contig = np.ascontiguousarray(xy)
                checksum = hashlib.sha256(
                    xy_contig.tobytes()
                ).hexdigest()

                experiment_id = checksum[:16]

                # Skip if already processed
                if experiment_id in existing_ids:
                    print(f"{experiment_id} already exists — skipping.")
                    continue

                # Create realization
                realization = Realization(experiment_id)
                dataset.realizations.append(realization)
                print(f"Processing new realization {experiment_id}")

                # Run smoothing
                segments = process(entry, i, j, file)

                if len(segments) == 0:
                    print("No valid segments — skipping.")
                    continue

                # Save DataItems
                # Segments
                item_segments = DataItem(
                    data_name=TANGENT_ANGLE_SEGMENTS,
                    data=np.array(segments, dtype=object), 
                    dependencies=[ORIGINAL.key],
                    algorithm="psi_smoothing_v1",
                )

                dataset.add_data_to_realization(
                    realization,
                    item_segments,
                )

                # Arclength (51 control points)
                arc = np.linspace(
                    0,
                    entry["ds_in_micron"] * entry['tangent_angle_psi_in_rad'].shape[1],
                    51
                )

                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=ARCLENGTH,
                        data=arc,
                        dependencies=[ORIGINAL.key],
                        algorithm="psi_smoothing_v1",
                    ),
                )

                # dt
                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=DT_FRAME,
                        data=float(entry["dt_in_second"]),
                        dependencies=[ORIGINAL.key],
                        algorithm="psi_smoothing_v1",
                    ),
                )

                # condition description
                cond_dict = {
                    "ATP": entry["ATP_in_uM"],
                    "KCl": entry["KCl_in_mM"],
                    "sexp": entry["sexp"],
                    "file": entry["File"],
                }

                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=CONDITION_DESCRIPTION,
                        data=cond_dict,
                        dependencies=[ORIGINAL.key],
                        algorithm="psi_smoothing_v1",
                    ),
                )

                # A given
                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=A,
                        data=float(entry["A"]),
                        dependencies=[GIVEN.key],
                        algorithm="given by geyer et al.",
                    ),
                )

                # F given
                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=F,
                        data=float(entry["f0_in_Hz"]),
                        dependencies=[GIVEN.key],
                        algorithm="given by geyer et al.",
                    ),
                )

                # Q given
                dataset.add_data_to_realization(
                    realization,
                    DataItem(
                        data_name=Q,
                        data=float(entry["Q"]),
                        dependencies=[GIVEN.key],
                        algorithm="given by geyer et al.",
                    ),
                )

                existing_ids.add(experiment_id)

    print("\nDone.")


if __name__ == "__main__":
    main()