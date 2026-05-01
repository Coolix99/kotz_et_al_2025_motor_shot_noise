import os
import hashlib
import numpy as np

from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datasets.DataItem import DataItem


from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
    CONDITION_DESCRIPTION,
    DT_FRAME,ARCLENGTH
)
from cilia.datastructures.special_source_names import ORIGINAL
from open_res import read_spde


# CONFIG
SIM_SPDE_ALGO = "spde_loader_v1"


# DATASET CLASS
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


# HELPERS
def build_experiment_id(file_path):
    with open(file_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:16]

def extract_conditions(params):
    return {
        "mu_a": params.get("mu_a"),
        "mu": params.get("mu"),
        "eta": params.get("eta"),
        "zeta": params.get("zeta"),
        "beta": params.get("beta"),
        "fstar": params.get("fstar"),
        "Nmotor": params.get("Nmotor"),
        "mode": params.get("mode"),
        "dt": params.get("dt"),
        "T": params.get("T"),
    }


# MAIN BUILD FUNCTION
def build_dataset(original_path, dataset_path):

    dataset = SimulationDataset(dataset_path)
    existing_ids = {r.experiment_id for r in dataset.realizations}

    files = [f for f in os.listdir(original_path) if f.endswith(".gz")]

    print(f"Found {len(files)} simulation files")

    for file in files:
        fpath = os.path.join(original_path, file)

        experiment_id = build_experiment_id(fpath)

        # if experiment_id in existing_ids:
        #     print(f"{experiment_id} exists → skip")
        #     continue

        print(f"Processing {file} → {experiment_id}")

        # read params only
        try:
            data = read_spde(fpath)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        params = data["params"]
        gamma = data["gamma_mat"]
        Ns = gamma.shape[1]
        s = np.linspace(0.0, 1.0, Ns)
        realization = Realization(experiment_id)
        dataset.realizations.append(realization)

        exp_dir = os.path.join(dataset_path, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        # ----------------------------------------------------
        # Symlink
        # ----------------------------------------------------
        target_file = os.path.join(exp_dir, file)

        if not os.path.exists(target_file):
            rel_src = os.path.relpath(fpath, exp_dir)
            os.symlink(rel_src, target_file)

        # ----------------------------------------------------
        # TANGENT ANGLE SERIES (correct!)
        # ----------------------------------------------------
        dataset.add_data_to_realization(
            realization,
            DataItem(
                data_name=TANGENT_ANGLE_SERIES,
                data={"filename": file},
                dependencies=[ORIGINAL.key],
                algorithm=SIM_SPDE_ALGO,
            ),
        )

        # ----------------------------------------------------
        # CONDITIONS
        # ----------------------------------------------------
        cond = extract_conditions(params)
        cond["file"] = file

        dataset.add_data_to_realization(
            realization,
            DataItem(
                data_name=CONDITION_DESCRIPTION,
                data=cond,
                dependencies=[ORIGINAL.key],
                algorithm="spde_metadata",
            ),
        )

        # ----------------------------------------------------
        # DT
        # ----------------------------------------------------
        dt = params.get("dt", np.nan)
        t_sub = params.get("t_sub", np.nan)

        if np.isfinite(dt) and np.isfinite(t_sub) and dt > 0 and t_sub > 0:
            dt_frame = 1.0 / t_sub
        else:
            dt_frame = np.nan

        dataset.add_data_to_realization(
            realization,
            DataItem(
                data_name=DT_FRAME,
                data=float(dt_frame),
                dependencies=[ORIGINAL.key],
                algorithm="spde_metadata",
            ),
        )
        dataset.add_data_to_realization(
            realization,
            DataItem(
                data_name=ARCLENGTH,
                data=s,
                dependencies=[ORIGINAL.key],
                algorithm="spde_metadata",
            ),
        )

        existing_ids.add(experiment_id)

    print("\n✅ Simulation dataset built.")


# MAIN
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

    dataset_pairs = [
        (
            "original/simulated/reaction_diffusion_data/res_cwn_phasespace",
            "structured/cass_2d_cwn_phasespace",
        ),
        (
            "original/simulated/reaction_diffusion_data/res_wn_phasespace",
            "structured/cass_2d_wn_phasespace",
        ),
        (
            "original/simulated/reaction_diffusion_data/c_simdata_phasespace",
            "structured/cass_2d_phasespace",
        ),
        (
            "original/simulated/reaction_diffusion_data/c_simdata_2d_extraction",
            "structured/cass_2d_extraction",
        ),
        (
            "original/simulated/reaction_diffusion_data/c_simdata_3d_extraction",
            "structured/cass_3d_extraction",
        ),
        (
            "original/simulated/reaction_diffusion_data/c_simdata_2d_extraction_cass",
            "structured/c_simdata_2d_extraction_cass",
        ),
        (
            "original/simulated/reaction_diffusion_data/c_simdata_2d_extraction_cass_5fold",
            "structured/c_simdata_2d_extraction_cass_5fold",
        ),
    ]
    for original_rel, structured_rel in dataset_pairs:

        print("\n" + "=" * 60)
        print(f"Building dataset:\n{original_rel} → {structured_rel}")
        print("=" * 60)

        original_path = os.path.join(CILIA_FOLDER, original_rel)
        dataset_path = os.path.join(CILIA_FOLDER, structured_rel)

        build_dataset(original_path, dataset_path)

if __name__ == "__main__":
    main()
