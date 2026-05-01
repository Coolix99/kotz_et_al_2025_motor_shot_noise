import numpy as np
import os
import pandas as pd

from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization

from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
    CONDITION_DESCRIPTION,
    A, F, GEYER, Q,
    ARCLENGTH,
    PERIODIC_AVG,
    HYDRO_CONTRIBUTION,
)

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
    protophase_from_first_mode_hilbert_t,
    estimate_phase_from_protophase_t,
)

from cilia.transformers.periodic_avg_transformer import (
    periodic_avg_t
)
from cilia.transformers.q_transformer import (
    estimate_Q_msd_fixed_tau_t,
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

    # get original tangent angle IDs
    segment_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == TANGENT_ANGLE_SERIES.key
    ]

    # amplitude + frequency
    for tr in [
        estimate_amplitude_psd_t,
        estimate_f_from_tangent_angle_power_t,
    ]:
        run_transformer_on_dataset(
            dataset,
            tr,
            allowed_ids={"tangent_angle_series": segment_ids},
            skip_existing=True,
        )
    
    run_transformer_on_dataset(
        dataset,
        protophase_from_first_mode_hilbert_t, 
        allowed_ids={"tangent_angle_series": segment_ids},
        skip_existing=True,
    )
 
    run_transformer_on_dataset(
        dataset,
        estimate_phase_from_protophase_t,
        allowed_algorithms=[
            protophase_from_first_mode_hilbert_t.algorithm,
        ],
        skip_existing=True,
    )

    phase_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == "phase_series"
        and item.algorithm == estimate_phase_from_protophase_t.algorithm
    ]

    run_transformer_on_dataset(
        dataset,
        estimate_Q_msd_fixed_tau_t,
        allowed_ids={
            "phase_series": phase_ids,
        },
        skip_existing=True,
    )

    run_transformer_on_dataset(
        dataset,
        periodic_avg_t,
        allowed_ids={
            "tangent_angle_series": segment_ids,
            "phase_series": phase_ids,
        },
        skip_existing=True,
    )

    run_transformer_on_dataset(
        dataset,
        get_geyer_fit_cycle_t,
        skip_existing=True,
    )

    print("\nPipeline done.")
from cilia.transformers.hydro_transformer import avg_hydrodynamics_t


def _ancestor_ids_with_key(realization: Realization, item, target_key: str):
    """Return IDs of ancestors of `item` whose DataName.key matches `target_key`."""
    matches = set()
    stack = list(item.dependencies)
    visited = set()

    while stack:
        dep_id = stack.pop()
        if dep_id in visited:
            continue
        visited.add(dep_id)

        dep = realization.data_items.get(dep_id)
        if dep is None:
            continue

        if dep.data_name.key == target_key:
            matches.add(dep.id)
            continue

        stack.extend(dep.dependencies)

    return matches

def run_hydro_for_cass(dataset: DataSet):
    """
    Run extra transformers only for:
        Nmotor == -1 AND mu_a == 1570
    """

    for realization in dataset:
        matched = False
        for item in realization.data_items.values():
            if item.data_name.key != CONDITION_DESCRIPTION.key:
                continue

            try:
                val = item.resolve(dataset, os.path.join(dataset.path, realization.experiment_id))
            except Exception:
                continue

            if not isinstance(val, dict):
                continue

            meta = val.get("metadata", val)

            mu_a = meta.get("mu_a")
            Nmotor = meta.get("Nmotor")
            
            if mu_a == 1570.0 and Nmotor == -1.0:
                print(mu_a, Nmotor)
                matched = True
                break

        if not matched:
            continue  
        print(realization.experiment_id)

        tangent_angle_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == TANGENT_ANGLE_SERIES.key
        ]

        arclength_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == ARCLENGTH.key
        ]

        periodic_items = [
            item for item in realization.data_items.values()
            if item.data_name.key == PERIODIC_AVG.key
        ]

        f_items = [
            item for item in realization.data_items.values()
            if item.data_name.key == F.key
        ]

        if not arclength_ids:
            print(f"{realization.experiment_id}: missing arclength, skipping")
            continue

        for tangent_id in tangent_angle_ids:
            periodic_ids = [
                it.id for it in periodic_items
                if tangent_id in _ancestor_ids_with_key(realization, it, TANGENT_ANGLE_SERIES.key)
            ]

            f_ids = [
                it.id for it in f_items
                if tangent_id in _ancestor_ids_with_key(realization, it, TANGENT_ANGLE_SERIES.key)
            ]

            if not periodic_ids or not f_ids:
                continue

            run_transformer_on_realization(
                dataset,
                realization,
                avg_hydrodynamics_t,
                allowed_ids={
                    PERIODIC_AVG.key: periodic_ids,
                    F.key: f_ids,
                    ARCLENGTH.key: arclength_ids,
                },
                skip_existing=True,
            )


def collect_hydro_results(dataset: DataSet, conditions_df: pd.DataFrame, out_csv: str):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        items = [
            it for it in realization.data_items.values()
            if it.data_name.key == HYDRO_CONTRIBUTION.key
        ]

        for it in items:
            try:
                val = it.resolve(dataset, os.path.join(dataset.path, exp_id))
            except Exception:
                continue

            rows.append({
                "experiment_id": exp_id,
                "algorithm": it.algorithm,
                "mean_R_free": float(val.get("mean_R_free", np.nan)),
                "mean_R_fixed": float(val.get("mean_R_fixed", np.nan)),
                "mean_R_chlamy": float(val.get("mean_R_chlamy", np.nan)),
            })

        realization.unload_data()

    if len(rows) == 0:
        print("⚠️ No hydrodynamics results found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    conditions_df = conditions_df.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    cols = [
        "experiment_id",
        "algorithm",
        "mu_a", "mu", "eta", "zeta", "beta", "Nmotor", "mode",
        "mean_R_free", "mean_R_fixed", "mean_R_chlamy",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    return df


# ============================================================
# SCALAR COLLECTION (SIMPLIFIED)
# ============================================================
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
        "percentile_1",
        "variance_explained",
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
    datasets = [
        "cass_2d_cwn_phasespace",
        "cass_2d_wn_phasespace",
        "cass_2d_phasespace",
    ]

    for name in datasets:

        dataset_path = os.path.join(CILIA_FOLDER, "structured", name)
        out_csv = f"./scalar_observables_{name.replace('cass_2d_', '').replace('_phasespace','')}.csv"
        hydro_csv = "./hydro_cass_2d.csv"

        print(f"\n=== {name} ===")

        dataset = SimulationDataset(dataset_path)
        #dataset.clean(delete_only=[HYDRO_CONTRIBUTION.key])
        # df = collect_conditions(dataset)
        # run_pipeline(dataset)
        # collect_scalar_dense(dataset, df, out_csv)

        df_cond = collect_conditions(dataset)
        run_hydro_for_cass(dataset)
        collect_hydro_results(dataset, df_cond, hydro_csv)

if __name__ == "__main__":
    main()
