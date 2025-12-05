import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils_rdf.config import (
    OUTPUT_DATA_PHASESPACE_DIR,
    OUTPUT_DATA_EXTRACTION_DIR,
    OUTPUT_DATA_NEW_PARAMETER_DIR,
    INITIAL_TRANSIENT_TIME_SIMULATION,
    DT_SIMULATION as DT,
    N_HARMONICS_GLOBAL_PHASE_SIMULATION,
    REMOVE_HILBERT_SIMULATION,
    SPDE_DATA_DIR_PHASESPACE,SPDE_DATA_DIR_EXTRACTION,
    SPDE_DATA_DIR_NEW_PARAMETER,
    SPDE_DATA_DIR_PARAMSCAN, OUTPUT_DATA_PARAMSCAN_DIR,
)
from utils_rdf.flagella_algorythms import (
    center_data,
    get_quantities,
    get_global_phase_amplitude,
    get_local_phase_amplitude,
)

from utils_rdf.open_res import read_spde

# === PROCESSING FUNCTION ===
def process_spde_file(filepath, output_dir, do_phi_omega_a=False):
    """Process one .spde.gz file: read, extract parameters, compute analysis, save."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_folder = Path(output_dir) / base_name

    expected_files = ["parameters.npz", "analysis_data.npz"]
    if out_folder.exists():
        all_exist = all((out_folder / f).exists() and (out_folder / f).stat().st_size > 0 for f in expected_files)
        if all_exist:
            # pass
            return False  # Skip already processed

    out_folder.mkdir(parents=True, exist_ok=True)

    try:
        data = read_spde(filepath)

        params = data["params"]
        n_coarse = data["n_coarse"]
        n_nodes = data["n_nodes"]
        count_large = data["count_large_zeta_gamma"]
        count_regular = data["count_regular_poisson"]
        runtime = data["runtime_s"]
        t_coarse = data["t_coarse"]
        gamma_mat = data["gamma_mat"]
        nplus_mat = data["nplus_mat"]
        nminus_mat = data["nminus_mat"]

        del nplus_mat, nminus_mat, t_coarse

    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {type(e).__name__}: {e}")
        raise
        return False

    # --- Save scalar parameters and metadata ---
    scalars = dict(params)
    scalars.update({
        "runtime": runtime,
        "count_large_zeta_gamma": count_large,
        "count_regular_poisson": count_regular,
        "n_coarse": n_coarse,
        "n_nodes": n_nodes,
    })
    np.savez(out_folder / "parameters.npz", **scalars)

    # --- Process gamma_mat ---
    if isinstance(gamma_mat, np.ndarray) and gamma_mat.size > 0:
        gamma_mat = gamma_mat.reshape((n_coarse, n_nodes))
        gamma_mat = gamma_mat[INITIAL_TRANSIENT_TIME_SIMULATION:, :]

        mean_shape, centered = center_data(gamma_mat)

        res = get_quantities(
            centered,
            DT,
            N_HARMONICS_GLOBAL_PHASE_SIMULATION,
            REMOVE_HILBERT_SIMULATION,
            Q_adaptive=False,
        )

        np.savez(out_folder / "analysis_data.npz", mean_shape=mean_shape, **res)

        # --- Optional phi/omega/amplitude ---
        if do_phi_omega_a:
            phi_global, omega_global, amp_global = get_global_phase_amplitude(
                centered - centered[:, [0]],
                nharm=N_HARMONICS_GLOBAL_PHASE_SIMULATION,
            )
            global_phi_omega_a = np.vstack((phi_global, omega_global, amp_global))

            phi_local, omega_local, amp_local = get_local_phase_amplitude(
                centered - centered[:, [0]],
                nharm=N_HARMONICS_GLOBAL_PHASE_SIMULATION,
                dt=15, ds=4
            )
            local_phi_omega_a = np.stack((phi_local, omega_local, amp_local), axis=0)

            np.savez(
                out_folder / "phi_omega_a.npz",
                global_phi_omega_a=global_phi_omega_a,
                local_phi_omega_a=local_phi_omega_a,
            )
    else:
        print(f"⚠️ gamma_mat missing or empty in {filepath}")

    return True


# === GENERIC PROCESS LOOP ===
def process_all_spde_files(input_dir, output_dir, process_func, do_phi_omega_a=False):
    files = [f for f in os.listdir(input_dir) if f.endswith(".spde.gz") or f.endswith(".gz")]
    np.random.shuffle(files)
    processed, skipped = 0, 0

    for fname in tqdm(files, desc=f"Processing SPDE files from {input_dir}"):
        filepath = os.path.join(input_dir, fname)
        if process_func(filepath, output_dir, do_phi_omega_a):
            processed += 1
        else:
            skipped += 1

    print(f"\n✅ Done ({input_dir}): {processed} processed, {skipped} skipped.")


# === MAIN ===
def main():
    process_all_spde_files(SPDE_DATA_DIR_PHASESPACE, OUTPUT_DATA_PHASESPACE_DIR, process_spde_file, do_phi_omega_a=False)
    process_all_spde_files(SPDE_DATA_DIR_EXTRACTION, OUTPUT_DATA_EXTRACTION_DIR, process_spde_file, do_phi_omega_a=True)
    process_all_spde_files(SPDE_DATA_DIR_NEW_PARAMETER, OUTPUT_DATA_NEW_PARAMETER_DIR, process_spde_file, do_phi_omega_a=True)
    process_all_spde_files(SPDE_DATA_DIR_PARAMSCAN, OUTPUT_DATA_PARAMSCAN_DIR, process_spde_file, do_phi_omega_a=False)

if __name__ == "__main__":
    main()
    