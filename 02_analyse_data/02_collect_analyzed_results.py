import numpy as np
import pandas as pd
from pathlib import Path

from utils_rdf.config import OUTPUT_DATA_PHASESPACE_DIR, OUTPUT_DATA_EXTRACTION_DIR, \
OUTPUT_DATA_NEW_PARAMETER_DIR, OUTPUT_DATA_PARAMSCAN_DIR, OUTPUT_DATA_WN_PHASESPACE_DIR, OUTPUT_DATA_CWN_PHASESPACE_DIR

def collect_results(summary_dir):
    rows = []

    for folder in sorted(Path(summary_dir).iterdir()):
        if not folder.is_dir():
            continue

        params_path = folder / "parameters.npz"
        analysis_path = folder / "analysis_data.npz"

        if not (params_path.exists() and analysis_path.exists()):
            continue

        # Load all data
        try:
            params = dict(np.load(params_path, allow_pickle=True))
            analysis = dict(np.load(analysis_path, allow_pickle=True))
        except Exception as e:
            print(f"Skipping {folder.name} due to error: {e}")
            continue

        # --- build result row ---
        row = {'sample': folder.name}

        # --- collect all scalar entries from parameters ---
        for key, val in params.items():
            # Convert 0D arrays to scalars
            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val)
            # Skip arrays, lists, dicts, etc.
            if np.isscalar(val):
                row[key] = val

        # --- handle special cases ---
        if "Nmotor" in row:
            if isinstance(row["Nmotor"], (int, float)) and row["Nmotor"] < 0:
                row["Nmotor"] = np.inf  # deterministic limit

        # --- collect all scalar entries from analysis ---
        for key, val in analysis.items():
            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val)
            if np.isscalar(val):
                row[key] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def collect_and_save(summary_directory, filename):
    """Collect results from a directory and save them as CSV."""
    result_df = collect_results(summary_directory)
    output_csv = Path(summary_directory) / filename
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved results to {output_csv}")

def main():
    collect_and_save(OUTPUT_DATA_PHASESPACE_DIR, "collected_results_phasespace.csv")
    collect_and_save(OUTPUT_DATA_EXTRACTION_DIR, "collected_results_extraction.csv")
    collect_and_save(OUTPUT_DATA_NEW_PARAMETER_DIR, "collected_results_new_parameter.csv")
    collect_and_save(OUTPUT_DATA_PARAMSCAN_DIR, "collected_results_paramscan.csv")
    collect_and_save(OUTPUT_DATA_WN_PHASESPACE_DIR, "collected_results_wn_phasespace.csv")
    collect_and_save(OUTPUT_DATA_CWN_PHASESPACE_DIR, "collected_results_cwn_phasespace.csv")


if __name__ == "__main__":
    main()




