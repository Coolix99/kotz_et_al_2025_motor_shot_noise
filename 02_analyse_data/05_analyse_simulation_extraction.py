import numpy as np
import os
import pandas as pd

from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization

from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
    CONDITION_DESCRIPTION,
    A, F, GEYER, Q, PHASE_SERIES, PERIODIC_AVG, CORR_FLUC, DT_FRAME
)
from cilia.datastructures.DataName import PHASE_DEFECTS_SERIES

from cilia.transformers.Transformer import (
    run_transformer_on_dataset,
    run_viewer_on_dataset,
    run_transformer_on_realization
)

# transformers (reduced set)
from cilia.transformers.a_transformer import estimate_amplitude_psd_t 
from cilia.transformers.f_transformer import estimate_f_from_tangent_angle_power_t
from cilia.transformers.geyer_transformer import get_geyer_fit_cycle_t
from cilia.transformers.phase_transformer import (
    protophase_from_spatial_modes_t,
    estimate_phase_from_protophase_t,
)
# from cilia.transformers.global_non_iso_transformer import non_iso_global_t
from cilia.transformers.periodic_avg_transformer import (
    periodic_avg_t
)
from cilia.transformers.q_transformer import (
    estimate_Q_msd_fixed_tau_t,
)
from cilia.transformers.local_phase_transformer import local_phase_t
from cilia.transformers.hydro_transformer import hydrodynamics_t
from cilia.transformers.local_phase_transformer import (correlation_length_t, defect_rate_t, correlations_and_fluctuations_t)
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


# ============================================================
# CONDITIONS
# ============================================================
def collect_conditions(dataset: DataSet) -> pd.DataFrame:
    rows = []

    def on_combo(realization, combo, combo_rows):
        row = combo_rows[0]
        value = row["value"]

        if not isinstance(value, dict):
            return

        rows.append({
            "experiment_id": realization.experiment_id,
            "mu_a": value.get("mu_a"),
            "mu": value.get("mu"),
            "eta": value.get("eta"),
            "zeta": value.get("zeta"),
            "beta": value.get("beta"),
            "Nmotor": value.get("Nmotor"),
            "mode": value.get("mode"),
            "file": value.get("file"),
        })

    run_viewer_on_dataset(
        dataset,
        inputs=(CONDITION_DESCRIPTION,),
        on_combo=on_combo,
    )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["experiment_id"])
    df["experiment_id"] = df["experiment_id"].astype(str)

    return df


# ============================================================
# PIPELINE (SIMPLIFIED)
# ============================================================
def run_pipeline(dataset: DataSet):

    print("\n=== Running simulation pipeline ===")

    tangent_angle_transformers = [
        estimate_amplitude_psd_t,
        estimate_f_from_tangent_angle_power_t,
    ]
    
    for tr in tangent_angle_transformers:
        run_transformer_on_dataset(
            dataset,
            tr,
            skip_existing=True,
        )
    
    # Phase construction
    run_transformer_on_dataset(
        dataset,
        protophase_from_spatial_modes_t,
        skip_existing=True,
    )

    
    run_transformer_on_dataset(
        dataset,
        estimate_phase_from_protophase_t,
        allowed_algorithms=[
            protophase_from_spatial_modes_t.algorithm,
        ],
        skip_existing=True,
    )

    corrected_phase_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key ==  PHASE_SERIES.key
        and item.algorithm == estimate_phase_from_protophase_t.algorithm
    ]

    # Phase-based estimators
    run_transformer_on_dataset(
        dataset,
        estimate_Q_msd_fixed_tau_t,
        allowed_ids={
            PHASE_SERIES.key: corrected_phase_ids,
        },
        skip_existing=True,
    )
    
    # Cycle averages
    run_transformer_on_dataset(
        dataset,
        periodic_avg_t,
        allowed_ids={
            PHASE_SERIES.key: corrected_phase_ids,
        },
        skip_existing=True,
    )

    # downstream cycle estimators 
    run_transformer_on_dataset(
        dataset,
        get_geyer_fit_cycle_t,
        skip_existing=True,
    )
    
    # run_transformer_on_dataset(
    #     dataset,
    #     local_phase_t,
    #     skip_existing=True,
    # )
    
    # run_transformer_on_dataset(
    #     dataset,
    #     defect_rate_t,
    #     skip_existing=True,
    # )
    
    # run_transformer_on_dataset(
    #     dataset,
    #     correlation_length_t,
    #     skip_existing=True,
    # )

    # run_transformer_on_dataset(
    #     dataset,
    #     correlations_and_fluctuations_t,
    #     skip_existing=True,
    # )
    #-----until here

    # run_transformer_on_dataset(
    #     dataset,
    #     segment_non_iso_local_t,
    #     skip_existing=True,
    # )

    # run_transformer_on_dataset(
    #     dataset,
    #     non_iso_global_t,
    #     skip_existing=True,
    # )

    # run_transformer_on_dataset(
    #     dataset,
    #     hydrodynamics_t,
    #     skip_existing=True,
    # )
    
    print("\nPipeline done.")

# ============================================================
# SCALAR COLLECTION (SIMPLIFIED)
# ============================================================
from cilia.datastructures.DataName import CORR_LENGTH
def collect_scalar_dense(dataset, conditions_df, out_csv):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        out = {"experiment_id": exp_id}

        items = realization.data_items.values()

        def get_items(key):
            return [it for it in items if it.data_name.key == key]

        def extract_scalar(items, getter):
            vals = []
            for it in items:
                try:
                    v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
                    vals.append(getter(v))
                except Exception:
                    continue
            return np.nanmean(vals) if vals else np.nan

        # -------------------------------------------------
        # Scalars
        # -------------------------------------------------
        out["f"] = extract_scalar(
            get_items(F.key),
            lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
        )

        out["amplitude"] = extract_scalar(
            get_items(A.key),
            lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
        )

        def geyer_get(v):
            if isinstance(v, dict) and "lambda" in v:
                return float(v["lambda"]) / float(v["L"])
            return np.nan

        out["lambda"] = extract_scalar(
            get_items(GEYER.key),
            geyer_get,
        )

        out["Q"] = extract_scalar(
            get_items(Q.key),
            lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
        )
        out["corr_length"] = extract_scalar(
            get_items(CORR_LENGTH.key),
            lambda v: float(v.get("metadata", v) if isinstance(v, dict) else v),
        )

        phase_items = [
            it for it in items
            if it.data_name.key == "phase_series"
            and it.algorithm == estimate_phase_from_protophase_t.algorithm
        ]

        def get_percentile(v):
            if isinstance(v, dict):
                return float(v.get("percentile_1", np.nan))
            return np.nan

        def get_variance(v):
            if isinstance(v, dict):
                return float(v.get("variance_explained", np.nan))
            return np.nan

        out["percentile_1"] = extract_scalar(
            phase_items,
            get_percentile,
        )

        out["variance_explained"] = extract_scalar(
            phase_items,
            get_variance,
        )

        # -------------------------------------------------
        # DEFECT RATES (series-based)
        # -------------------------------------------------
        defect_items = get_items(PHASE_DEFECTS_SERIES.key)

        total_pos = 0
        total_neg = 0
        total_time = 0.0

        for it in defect_items:
            try:
                defects = it.resolve(dataset, os.path.join(dataset.path, exp_id))
            except Exception:
                continue

            clean = defects.get("clean", None)
            T_eff = defects.get("effective_T", None)

            if clean is None or T_eff is None:
                continue
            if T_eff <= 0:
                continue

            clean = np.asarray(clean)

            if clean.size == 0:
                n_pos = 0
                n_neg = 0
            else:
                q = clean[:, 2]
                n_pos = np.sum(q > 0)
                n_neg = np.sum(q < 0)

            total_pos += n_pos
            total_neg += n_neg
            total_time += T_eff

        if total_time > 0:
            out["defect_rate_pos"] = total_pos / total_time
            out["defect_rate_neg"] = total_neg / total_time
            out["defect_rate_total"] = (
                out["defect_rate_pos"] + out["defect_rate_neg"]
            )
        else:
            out["defect_rate_pos"] = np.nan
            out["defect_rate_neg"] = np.nan
            out["defect_rate_total"] = np.nan


        # -------------------------------------------------
        # λ(t) and a(t) statistics (with 3σ filtering)
        # -------------------------------------------------
        corr_items = get_items(CORR_FLUC.key)

        lambda_all = []
        amp_all = []

        for it in corr_items:
            try:
                v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
            except Exception:
                continue

            lam = v.get("lambda_t", None)
            amp = v.get("mean_a_t", None)

            if lam is not None:
                lam = np.asarray(lam, float)
                lambda_all.append(lam)

            if amp is not None:
                amp = np.asarray(amp, float)
                amp_all.append(amp)

        # concatenate across realizations
        if len(lambda_all) > 0:
            lambda_all = np.concatenate(lambda_all)
        else:
            lambda_all = np.array([])

        if len(amp_all) > 0:
            amp_all = np.concatenate(amp_all)
        else:
            amp_all = np.array([])


        def _mean_std_filtered(x):
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]

            if len(x) < 10:
                return np.nan, np.nan

            m = np.mean(x)
            s = np.std(x)

            if not np.isfinite(s) or s <= 0:
                return m, 0.0

            mask = np.abs(x - m) < 3 * s
            x_f = x[mask]

            if len(x_f) < 5:
                return np.nan, np.nan

            return np.mean(x_f), np.std(x_f)


        lam_mean, lam_std = _mean_std_filtered(lambda_all)
        amp_mean, amp_std = _mean_std_filtered(amp_all)

        out["lambda_t_mean"] = lam_mean
        out["lambda_t_std"] = lam_std

        out["amplitude_t_mean"] = amp_mean
        out["amplitude_t_std"] = amp_std

        rows.append(out)
        
        # Unload cached data to prevent memory leak
        realization.unload_data()

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("⚠️ No data found")
        return df

    # -------------------------------------------------
    # Merge conditions
    # -------------------------------------------------
    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    # column order
    cols = [
        "experiment_id",
        "mu_a", "mu", "eta", "zeta", "beta", "Nmotor", "mode",
        "f",
        "amplitude",
        "lambda",
        "Q",
        "corr_length", 
        "percentile_1",
        "variance_explained",
        "defect_rate_pos",
        "defect_rate_neg",
        "defect_rate_total",
        "lambda_t_mean",
        "lambda_t_std",
        "amplitude_t_mean",
        "amplitude_t_std",
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    return df


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
    import importlib.util
    cfg_path = PROJECT_ROOT / "local_config.py"
    spec = importlib.util.spec_from_file_location("local_config", str(cfg_path))
    local_config = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(local_config)  # type: ignore
    CILIA_FOLDER = local_config.CILIA_FOLDER


def main():
    dataset_path = os.path.join(CILIA_FOLDER, "structured", 'cass_3d_extraction')
    out_csv = f"./scalar_observables_cass_3d_extraction.csv"


    dataset = SimulationDataset(dataset_path)
    # dataset.clean()
    dataset.clean(delete_only=[PHASE_SERIES.key, ])
    #dataset.clean(delete_only=[DT_FRAME.key, F.key, PHASE_DEFECTS_SERIES.key, CORR_FLUC.key])
    #return
    df = collect_conditions(dataset)
    run_pipeline(dataset)
    collect_scalar_dense(dataset, df, out_csv)


if __name__ == "__main__":
    main()