import os
import sys
import time
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

from pipeline_utils import (
    SimulationDataset,
    count_realizations,
    collect_conditions,
    run_pipeline_s1,
    collect_scalar_dense_s1,
    run_pipeline_s2,
    collect_scalar_dense_s2,
    run_pipeline_s3,
    collect_scalar_dense_s3
)

SLEEP_SECONDS = 1

DATASETS = [
    {
        "name": "S1_2d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S1_2d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s1_2d.csv"),
        "run_pipeline": run_pipeline_s1,
        "collect_scalar": collect_scalar_dense_s1,
        "clean_on_start": False,
    },
    {
        "name": "S2_2d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S2_2d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s2_2d.csv"),
        "run_pipeline": run_pipeline_s2,
        "collect_scalar": collect_scalar_dense_s2,
        "clean_on_start": False,
    },
    {
        "name": "S3_2d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S3_2d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s3_2d.csv"),
        "run_pipeline": run_pipeline_s3,
        "collect_scalar": collect_scalar_dense_s3,  
        "clean_on_start": False,
    },
        {
        "name": "S1_3d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S1_3d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s1_3d.csv"),
        "run_pipeline": run_pipeline_s1,
        "collect_scalar": collect_scalar_dense_s1,
        "clean_on_start": False,
    },
    {
        "name": "S2_3d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S2_3d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s2_3d.csv"),
        "run_pipeline": run_pipeline_s2,
        "collect_scalar": collect_scalar_dense_s2,
        "clean_on_start": False,
    },
    {
        "name": "S3_3d",
        "dataset_path": os.path.join(CILIA_FOLDER, "structured/SBI/SBI_S3_3d"),
        "out_csv": os.path.join(PROJECT_ROOT, "scalar_observables_sbi_s3_3d.csv"),
        "run_pipeline": run_pipeline_s3,
        "collect_scalar": collect_scalar_dense_s3,  
        "clean_on_start": False,
    },
]


def update_dataset(cfg):
    dataset = SimulationDataset(cfg["dataset_path"])
    df_conditions = collect_conditions(dataset)
    cfg["run_pipeline"](dataset)
    df = cfg["collect_scalar"](dataset, df_conditions, cfg["out_csv"], dimension=cfg["name"].split("_")[1])
    print(f"[{cfg['name']}] Updated CSV with {len(df)} rows")


def main():
    last_counts = {}

    for cfg in DATASETS:
        if cfg.get("clean_on_start", False):
            dataset = SimulationDataset(cfg["dataset_path"])
            dataset.clean()
        last_counts[cfg["name"]] = -1

    while True:
        any_update = False

        for cfg in DATASETS:
            name = cfg["name"]
            dataset_path = cfg["dataset_path"]

            current_count = count_realizations(dataset_path)

            if current_count == 0:
                continue

            if current_count == last_counts[name]:
                continue

            print(f"[{name}] Update detected: {last_counts[name]} -> {current_count}")
            update_dataset(cfg)
            last_counts[name] = current_count
            any_update = True

        if not any_update:
            if all(count_realizations(cfg["dataset_path"]) == 0 for cfg in DATASETS):
                print("No realizations yet")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()