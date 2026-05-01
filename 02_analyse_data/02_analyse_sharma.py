import numpy as np
import os
from collections import defaultdict
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from cilia.datastructures.DataName import TANGENT_ANGLE_SEGMENTS
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.transformers.gauge_transformer import base_gauge_sym_segments_t
from cilia.datastructures.DataName import CONDITION_DESCRIPTION, SEGMENT_CORR_FLUC
from cilia.transformers.Transformer import run_viewer_on_dataset
from cilia.transformers.local_phase_transformer import (
    segment_local_phase_t, segment_defect_rate_t, segment_correlation_length_t,# segment_non_iso_global_t,#segment_non_iso_local_t
)
from cilia.transformers.Transformer import (
    run_transformer_on_dataset,
)
from cilia.transformers.local_phase_transformer import segment_correlations_and_fluctuations_t
from cilia.transformers.hydro_transformer import segment_hydrodynamics_t
from cilia.transformers.a_transformer import (
    segment_amplitude_psd_t,
)

from cilia.transformers.f_transformer import (
    segment_f_from_tangent_angle_power_t,
)

from cilia.transformers.q_transformer import (
    segment_Q_msd_fixed_tau_t,
)

from cilia.transformers.geyer_transformer import (
    get_geyer_fit_cycle_t,
)

from cilia.transformers.phase_transformer import (
    protophase_from_spatial_modes_segment_t,
    segment_phase_from_protophase_t,
)

from cilia.transformers.periodic_avg_transformer import (
    periodic_avg_segment_t
)
from cilia.datastructures.special_source_names import ORIGINAL

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

def collect_conditions(dataset: DataSet) -> pd.DataFrame:
    rows = []

    def on_combo(realization, combo, combo_rows):
        item = combo[0]
        row = combo_rows[0]

        value = row["value"]

        if not isinstance(value, dict):
            return

        rows.append({
            "experiment_id": realization.experiment_id,
            "ATP_uM": value.get("ATP"),
            "KCl_mM": value.get("KCl"),
            "sexp": value.get("sexp"),
            "file": value.get("file"),
        })

    run_viewer_on_dataset(
        dataset,
        inputs=(CONDITION_DESCRIPTION,),
        on_combo=on_combo,
    )

    df = pd.DataFrame(rows)

    # Remove duplicates (shouldn't exist, but safe)
    df = df.drop_duplicates(subset=["experiment_id"])
    df["experiment_id"] = df["experiment_id"].astype(str)
    return df

def _run_transformers(dataset: DataSet, transformers, title=None):
    if title:
        print(f"\n=== {title} ===")

    for tr in transformers:
        run_transformer_on_dataset(
            dataset,
            tr,
            skip_existing=True,
        )

def run_pipeline_with_gauge(dataset: DataSet) -> None:
    original_segment_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == "tangent_angle_segments"
        and ORIGINAL.key in item.dependencies
    ]
    run_transformer_on_dataset(
        dataset,
        base_gauge_sym_segments_t,
        allowed_ids={"tangent_angle_segments": original_segment_ids},
        skip_existing=True,
    )
    
    gauged_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == "tangent_angle_segments"
        and item.algorithm == base_gauge_sym_segments_t.algorithm
    ]

    tangent_angle_transformers = [
        segment_amplitude_psd_t,
        segment_f_from_tangent_angle_power_t,
    ]
    
    for tr in tangent_angle_transformers:
        run_transformer_on_dataset(
            dataset,
            tr,
            allowed_ids={"tangent_angle_segments": gauged_ids},
            skip_existing=True,
        )
    
    # Phase construction
    run_transformer_on_dataset(
        dataset,
        protophase_from_spatial_modes_segment_t,
        allowed_ids={"tangent_angle_segments": gauged_ids},
        skip_existing=True,
    )

    
    run_transformer_on_dataset(
        dataset,
        segment_phase_from_protophase_t,
        allowed_algorithms=[
            protophase_from_spatial_modes_segment_t.algorithm,
        ],
        skip_existing=True,
    )

    corrected_phase_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == "phase_segments"
        and item.algorithm == segment_phase_from_protophase_t.algorithm
    ]

    # Phase-based estimators
    run_transformer_on_dataset(
        dataset,
        segment_Q_msd_fixed_tau_t,
        allowed_ids={
            "phase_segments": corrected_phase_ids,
        },
        skip_existing=True,
    )
    
    # Cycle averages
    run_transformer_on_dataset(
        dataset,
        periodic_avg_segment_t,
        allowed_ids={
            "tangent_angle_segments": gauged_ids,
            "phase_segments": corrected_phase_ids,
        },
        skip_existing=True,
    )

    # downstream cycle estimators (safe)
    _run_transformers(dataset, [
        # get_a_psd_cycle_t,
        # get_a_ms_cycle_t,
        #get_lambda_cycle_t,
        get_geyer_fit_cycle_t,
    ])
    
    run_transformer_on_dataset(
        dataset,
        segment_local_phase_t,
        allowed_ids={
            "tangent_angle_segments": gauged_ids,
            "phase_segments": corrected_phase_ids,
        },
        skip_existing=True,
    )
    
    run_transformer_on_dataset(
        dataset,
        segment_defect_rate_t,
        skip_existing=True,
    )
    
    run_transformer_on_dataset(
        dataset,
        segment_correlation_length_t,
        skip_existing=True,
    )


    
    run_transformer_on_dataset(
        dataset,
        segment_correlations_and_fluctuations_t,
        skip_existing=True,
    )

    # run_transformer_on_dataset(
    #     dataset,
    #     avg_torque_power_t,
    #     skip_existing=True,
    # )


    run_transformer_on_dataset(
        dataset,
        segment_hydrodynamics_t,
        allowed_ids={"tangent_angle_segments": original_segment_ids},
        skip_existing=True,
    )
    


    print("\nPipeline done.")

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


from cilia.datastructures.DataName import F, CORR_LENGTH, GEYER, A, Q, LAM
from cilia.datastructures.DataName import PHASE_DEFECTS_SEGMENTS
from cilia.datastructures.special_source_names import GIVEN

def collect_scalar_dense(dataset, conditions_df, out_csv):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        out = {"experiment_id": exp_id}

        # -------------------------------------------------
        # Collect ALL items once
        # -------------------------------------------------
        items = realization.data_items.values()

        def get_items(key):
            return [it for it in items if it.data_name.key == key]

        # Source-aware scalar extraction
        def extract_scalar_with_sources(items, getter, key_name):
            grouped = {}

            for it in items:
                val = it.resolve(dataset, os.path.join(dataset.path, exp_id))

                try:
                    parsed = getter(val)
                except Exception:
                    continue

                deps = set(it.dependencies)

                if GIVEN.key in deps:
                    source = "given"
                else:
                    source = ''

                grouped.setdefault(source, []).append(parsed)

            out_local = {}

            for source, vals in grouped.items():
                if len(vals) > 1:
                    print(
                        f"⚠️ WARNING [{key_name}]: multiple values for source '{source}' "
                        f"in experiment {exp_id} → taking mean"
                    )

                out_local[f"{key_name}_{source}"] = float(np.nanmean(vals))

            return out_local

        # --- F ---
        out.update(
            extract_scalar_with_sources(
                get_items(F.key),
                lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
                "f",
            )
        )

        # --- Q ---
        out.update(
            extract_scalar_with_sources(
                get_items(Q.key),
                lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
                "Q",
            )
        )

        # --- amplitude ---
        out.update(
            extract_scalar_with_sources(
                get_items(A.key),
                lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
                "amplitude",
            )
        )

        # --- lambda (Geyer) + R2 ---
        def geyer_extract(v):
            if not isinstance(v, dict):
                return {}

            out_local = {}

            # --- lambda ---
            try:
                lam = float(v.get("lambda", np.nan))
                L = float(v.get("L", np.nan))
                if np.isfinite(lam) and np.isfinite(L) and L != 0:
                    out_local["lambda"] = lam / L
            except Exception:
                pass

            # --- R2 full ---
            if "r2_full_profiles" in v:
                try:
                    out_local["r2_full_profiles"] = float(v["r2_full_profiles"])
                except Exception:
                    out_local["r2_full_profiles"] = np.nan

            # --- R2 matrix ---
            if "r2_matrix" in v:
                try:
                    r2_mat = np.asarray(v["r2_matrix"], dtype=float)

                    # flatten into columns: r2_ij
                    for i in range(r2_mat.shape[0]):
                        for j in range(r2_mat.shape[1]):
                            out_local[f"r2_{i}_{j}"] = float(r2_mat[i, j])
                except Exception:
                    pass

            return out_local


        def extract_geyer_with_sources(items):
            grouped = {}

            for it in items:
                v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
                extracted = geyer_extract(v)

                if not extracted:
                    continue

                deps = set(it.dependencies)

                if GIVEN.key in deps:
                    source = "given"
                else:
                    source = ""

                grouped.setdefault(source, []).append(extracted)

            out_local = {}

            for source, dicts in grouped.items():
                keys = dicts[0].keys()

                for k in keys:
                    vals = [d.get(k, np.nan) for d in dicts]
                    if len(vals) > 1:
                        print(
                            f"⚠️ WARNING [GEYER:{k}]: multiple values for source '{source}' "
                            f"in experiment {exp_id} → taking mean"
                        )

                    out_local[f"{k}_{source}"] = float(np.nanmean(vals))

            return out_local


        # apply it
        out.update(
            extract_geyer_with_sources(
                get_items(GEYER.key)
            )
        )

        # Correlation length 
        corr_items = get_items(CORR_LENGTH.key)
        vals = []
        for it in corr_items:
            try:
                v = float(it.resolve(dataset, os.path.join(dataset.path, exp_id)))
                if v < 1e4:
                    vals.append(v)
            except Exception:
                continue
        out["corr_length"] = np.nanmean(vals) if vals else np.nan

        # Hydrodynamic contributions
        def hydro_get(v, key):
            if isinstance(v, dict):
                return float(v.get(key, np.nan))
            return np.nan

        hydro_items = get_items(HYDRO_CONTRIBUTION.key)

        def extract_hydro(key):
            vals = []
            for it in hydro_items:
                try:
                    v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
                    vals.append(hydro_get(v, key))
                except Exception:
                    continue
            return np.nanmean(vals) if vals else np.nan

        out["hydro_free"] = extract_hydro("mean_R_free")
        out["hydro_fixed"] = extract_hydro("mean_R_fixed")
        out["hydro_chlamy"] = extract_hydro("mean_R_chlamy")

        # --- power (pN µm / s) ---
        out["power_free"]   = extract_hydro("mean_power_free")
        out["power_fixed"]  = extract_hydro("mean_power_fixed")
        out["power_chlamy"] = extract_hydro("mean_power_chlamy")

        def to_fW(x):
            return x * 1e-3 if np.isfinite(x) else np.nan

        out["power_free_fW"]   = to_fW(out["power_free"])
        out["power_fixed_fW"]  = to_fW(out["power_fixed"])
        out["power_chlamy_fW"] = to_fW(out["power_chlamy"])

        # DEFECT RATES
        defect_items = get_items(PHASE_DEFECTS_SEGMENTS.key)

        total_pos = 0
        total_neg = 0
        total_time = 0.0

        for it in defect_items:
            segs = it.resolve(dataset, os.path.join(dataset.path, exp_id))

            ta_items = get_items(TANGENT_ANGLE_SEGMENTS.key)
            if not ta_items:
                continue

            ta_data = ta_items[0].resolve(dataset, os.path.join(dataset.path, exp_id))
            tangent = np.asarray(ta_data[0])
            Ns = tangent.shape[1]

            for seg in segs:
                clean = seg.get("clean")
                good_interval = seg.get("good_s_interval")
                T_eff = seg.get("effective_T")

                if clean is None or good_interval is None or T_eff is None:
                    continue
                if T_eff <= 0:
                    continue
                if good_interval[0] < 0:
                    continue

                g0, g1 = good_interval

                s_lo = int(0.15 * Ns)
                s_hi = int(0.7 * Ns)

                if g0 > s_lo or g1 < s_hi:
                    continue

                if clean.size == 0:
                    n_pos = 0
                    n_neg = 0
                else:
                    s = clean[:, 1]
                    q = clean[:, 2]
                    mask = (s >= s_lo) & (s <= s_hi)
                    n_pos = np.sum(q[mask] > 0)
                    n_neg = np.sum(q[mask] < 0)

                total_pos += n_pos
                total_neg += n_neg
                total_time += T_eff

        if total_time > 0:
            out["defect_rate_pos"] = total_pos / total_time
            out["defect_rate_neg"] = total_neg / total_time
        else:
            out["defect_rate_pos"] = np.nan
            out["defect_rate_neg"] = np.nan


        # -------------------------------------------------
        # λ(t) and a(t) statistics (segments)
        # -------------------------------------------------
        corr_items = get_items(SEGMENT_CORR_FLUC.key)

        lambda_all = []
        amp_all = []

        for it in corr_items:
            print(it)
            
            v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
           
            print('resolved')
            lam = v.get("lambda_t", None)
            amp = v.get("mean_a_t", None)

            if lam is not None:
                lambda_all.append(np.asarray(lam, float))

            if amp is not None:
                amp_all.append(np.asarray(amp, float))

        if len(lambda_all) > 0:
            lambda_all = np.concatenate(lambda_all)
        else:
            lambda_all = np.array([])

        if len(amp_all) > 0:
            amp_all = np.concatenate(amp_all)
        else:
            amp_all = np.array([])


        def _mean_std_filtered(x):
            x = x[np.isfinite(x)]
            if len(x) < 10:
                return np.nan, np.nan

            m = np.mean(x)
            s = np.std(x)

            mask = np.abs(x - m) < 3 * s
            x = x[mask]

            if len(x) < 5:
                return np.nan, np.nan

            return np.mean(x), np.std(x)


        lam_mean, lam_std = _mean_std_filtered(lambda_all)
        amp_mean, amp_std = _mean_std_filtered(amp_all)

        out["lambda_t_mean"] = lam_mean
        out["lambda_t_std"] = lam_std

        out["amplitude_t_mean"] = amp_mean
        out["amplitude_t_std"] = amp_std

        rows.append(out)

    # -------------------------------------------------
    # DataFrame
    # -------------------------------------------------
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("⚠️ No valid scalar data found")
        return df

    # -------------------------------------------------
    # Merge conditions
    # -------------------------------------------------
    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    # -------------------------------------------------
    # Column ordering (dynamic!)
    # -------------------------------------------------
    base_cols = [
        "experiment_id", "ATP_uM", "KCl_mM", "sexp", "file",
        "corr_length",
        "hydro_free", "hydro_fixed", "hydro_chlamy",
        "power_free_fW", "power_fixed_fW", "power_chlamy_fW",
        "defect_rate_pos", "defect_rate_neg",
    ]

    # dynamically include all source-split columns
    dynamic_cols = sorted([
        c for c in df.columns
        if c not in base_cols and c != "experiment_id"
    ])

    cols = ["experiment_id"] + [c for c in base_cols if c in df.columns] + dynamic_cols
    cols = list(dict.fromkeys(cols))  # remove duplicates

    df = df[cols]

    df.to_csv(out_csv, index=False)
    print(f"Saved dense scalar table → {out_csv}")

    return df

from cilia.datastructures.DataName import GEYER, PHASE_DEFECTS_SEGMENTS, CORR_LENGTH, HYDRO_CONTRIBUTION, PHASE_SEGMENTS

def main():
    dataset_path = os.path.join(CILIA_FOLDER,"structured/sharma")
    dataset = SharmaDataset(dataset_path)
    # dataset.clean()    
    dataset.clean(delete_only=[HYDRO_CONTRIBUTION.key])
    # dataset.clean(delete_only_exp_ids=["f34b4da146115fec"])#,"6279a8dd84538cbe"5fcc8a13641dc628
    
    df = collect_conditions(dataset)

    print("\n=== CONDITIONS OVERVIEW ===\n")
    print(df.head())

    print("\n=== UNIQUE CONDITIONS ===\n")
    print(
        df.groupby(["ATP_uM", "KCl_mM", "sexp"])
          .size()
          .reset_index(name="count")
          .sort_values(["ATP_uM", "KCl_mM"])
    )
    

    run_pipeline_with_gauge(dataset)
    out_csv = "./scalar_observables.csv"

    collect_scalar_dense(
        dataset,
        df,
        out_csv
    )

if __name__ == "__main__":
    main()