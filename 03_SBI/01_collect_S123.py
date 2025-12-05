import os
import time
import glob
import logging
import numpy as np
import pandas as pd

from utils_rdf.config import (
    OUTPUT_DATA_SBI_S1_DIR,
    OUTPUT_DATA_SBI_S2_DIR,
    OUTPUT_DATA_SBI_S3_DIR,  
    N_HARMONICS_GLOBAL_PHASE_SIMULATION,
)
import config_sbi

from utils_rdf.flagella_algorythms import (
    center_data,
    leadingModes,
    compute_psd,
    get_peak_frequency,
    variance_based_amplitude,
    periodic_average_tangent_angle_series,
    get_phase_by_hilbert,
    fit_shapeGeyer, getDQ_adaptive
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SUMMARY_S1 = os.path.join(OUTPUT_DATA_SBI_S1_DIR, "summary_s1.csv")
SUMMARY_S2 = os.path.join(OUTPUT_DATA_SBI_S2_DIR, "summary_s2.csv")
SUMMARY_S3 = os.path.join(OUTPUT_DATA_SBI_S3_DIR, "summary_s3.csv")

SLEEP_TIME = 10
REMOVE_INITIAL_TRANSIENT_TIME_SIMULATION = 100
REMOVE_HILBERT_SIMULATION = 50
EPS = 1e-12

# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def load_npz(filepath: str):
    data = np.load(filepath, allow_pickle=True)
    content = {k: data[k] for k in data.files}
    return content

def compute_single_likelihood(gamma_mat: np.ndarray) -> dict:
    """
    Compute basic oscillation quality and derived quantities.
    Invalid or weak signals now return NaN instead of raising exceptions.
    """
    stats = {"peak_freq": np.nan, "variance_amplitude": np.nan, "wavelength": np.nan}

    if gamma_mat is None or gamma_mat.size == 0:
        logging.warning("compute_single_likelihood: Empty or None gamma_mat.")
        return stats

    
    gamma_mat = gamma_mat[REMOVE_INITIAL_TRANSIENT_TIME_SIMULATION:, :]
    signal = gamma_mat[:, gamma_mat.shape[1] // 2]
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    power = np.abs(fft_vals) ** 2
    peak_idx = np.argmax(power[1:]) + 1

    # invalid frequency range
    if peak_idx < 3 or peak_idx > gamma_mat.shape[0]/3:
        logging.warning(f"compute_single_likelihood: Peak frequency out of expected range ({peak_idx})")
        return stats , f"compute_single_likelihood: Peak frequency out of expected range ({peak_idx})"

    mean_shape, centered = center_data(gamma_mat)
    leading_modes_result = leadingModes(centered, n_modes=3)
    freqs, psd = compute_psd(centered, 1.0)
    peak_freq = get_peak_frequency(freqs, psd, min_freq=1e-4)

    variance_amp = variance_based_amplitude(centered)
    if variance_amp < 1e-5:
        logging.warning("compute_single_likelihood: Amplitude too small.")
        return stats, "compute_single_likelihood: Amplitude too small."

    phase, *_ = get_phase_by_hilbert(
        leading_modes_result["projected"][:, 0],
        nharm=N_HARMONICS_GLOBAL_PHASE_SIMULATION,
        REMOVE_HILBERT=REMOVE_HILBERT_SIMULATION,
    )

    periodic_avg = periodic_average_tangent_angle_series( gamma_mat[REMOVE_HILBERT_SIMULATION:-REMOVE_HILBERT_SIMULATION, :],phase    )
    wavelength = fit_shapeGeyer(periodic_avg)["wavelength"]

    stats["peak_freq"] = peak_freq
    stats["variance_amplitude"] = variance_amp
    stats["wavelength"] = wavelength



    return stats, ''

def gaussian_loglike(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2

# --- S3:  Q computation -----------------------------------------
def compute_quality_Q(gamma_mat: np.ndarray) -> float:
    mean_shape, centered = center_data(gamma_mat)
    leading_modes_result = leadingModes(centered, n_modes=3)

    phase,p1, p10, p50=get_phase_by_hilbert(leading_modes_result["projected"][:,0],nharm=N_HARMONICS_GLOBAL_PHASE_SIMULATION, REMOVE_HILBERT=REMOVE_HILBERT_SIMULATION*10)

    _,Q=getDQ_adaptive(phase)
    return Q
# -------------------------------------------------------------------------------

# -------------------------------------------------------
# STAGE 1
# -------------------------------------------------------
def collect_S1_once(processed: set) -> list[dict]:
    """Collect new S1 results into summary."""
    results = []
    filepaths = sorted(glob.glob(os.path.join(OUTPUT_DATA_SBI_S1_DIR, "sim*.npz")))
    for fp in filepaths:
        fid = os.path.basename(fp)
        if fid in processed:
            continue
        try:
            data = load_npz(fp)
        except Exception as e:
            logging.error(f"Error loading {fid}: {e}")
            continue
        gamma = data.get("gamma_mat", None)
        meta = data.get("params", None)
        meta = dict(meta.item()) if hasattr(meta, "item") else dict(meta) if meta is not None else {}
        sampler = data.get("sampler", None)
        sampler = dict(sampler.item()) if hasattr(sampler, "item") else dict(sampler) if sampler is not None else {}
        error = data.get("error", '')
        row = {**meta, **sampler, "filename": fid}

        stats, error_single_like = compute_single_likelihood(gamma)
   
        error = (error or "") + (error_single_like or "")

        # if invalid (NaN), mark as low likelihood
        if np.isnan(stats["wavelength"]) or np.isnan(stats["variance_amplitude"]) or np.isnan(stats["peak_freq"]):
            row.update(stats)
            row["likelihood"] = 0.0
            row["error"] =  str(error)
        else:
            wavelength = stats["wavelength"]
            expected = config_sbi.expected_wavelength_pert
            sigma = config_sbi.sigma_wavelength
            log_like = gaussian_loglike(wavelength, expected, sigma)
            likelihood = float(np.exp(log_like))
            row.update(stats)
            row["likelihood"] = likelihood
            row["error"] = str(error)
        

        results.append(row)
        processed.add(fid)
        logging.info(f"S1 processed: {fid}")

    return results


# -------------------------------------------------------
# STAGE 2
# -------------------------------------------------------
def collect_S2_once(processed: set) -> list[dict]:
    """Collect new paired (S2) results with explicit error reporting for norm/pert."""
    results = []
    filepaths = sorted(glob.glob(os.path.join(OUTPUT_DATA_SBI_S2_DIR, "S2_*.npz")))
    for fp in filepaths:
        fid = os.path.basename(fp)
        if fid in processed:
            continue

        try:
            data = load_npz(fp)
        except Exception as e:
            logging.error(f"Error loading {fid}: {e}")
            continue
        sampler = data.get("sampler", None)
        sampler = dict(sampler.item()) if hasattr(sampler, "item") else dict(sampler) if sampler is not None else {}

        gamma_norm = data.get("gamma_norm", None)
        gamma_pert = data.get("gamma_pert", None)
        params_norm = dict(data["params_norm"].item()) if "params_norm" in data else {}

        error = ""

        # --- analyze norm ---
        stats_norm, err_norm = compute_single_likelihood(gamma_norm)
        if err_norm:
            error += f"[norm] {err_norm} "

        # --- analyze pert ---
        stats_pert, err_pert = compute_single_likelihood(gamma_pert)
        if err_pert:
            error += f"[pert] {err_pert} "

        # if any invalid value: mark low likelihood
        if (
            np.isnan(stats_norm["wavelength"])
            or np.isnan(stats_norm["variance_amplitude"])
            or np.isnan(stats_norm["peak_freq"])
            or np.isnan(stats_pert["wavelength"])
            or np.isnan(stats_pert["variance_amplitude"])
            or np.isnan(stats_pert["peak_freq"])
        ):
            row = {
                "filename": fid,
                "likelihood": 0.0,
                "error": error.strip(),
                "wavelength_norm": stats_norm["wavelength"],
                "wavelength_pert": stats_pert["wavelength"],
                "variance_amp_norm": stats_norm["variance_amplitude"],
                "variance_amp_pert": stats_pert["variance_amplitude"],
                "freq_ratio": np.nan,
                **{k: params_norm.get(k, np.nan) for k in ["beta", "mu", "eta", "zeta", "mu_a", "fstar"]},
                **sampler,
            }
            results.append(row)
            processed.add(fid)
            logging.info(f"S2 processed (invalid): {fid}")
            continue

        # --- expected values from config ---
        mu_wave_norm = config_sbi.expected_wavelength_norm
        mu_wave_pert = config_sbi.expected_wavelength_pert
        sigma_wave = config_sbi.sigma_wavelength

        mu_amp_ratio = config_sbi.expected_amplitude_ratio
        sigma_amp_ratio = config_sbi.sigma_amplitude_ratio

        mu_freq_ratio = config_sbi.expected_freq_ratio
        sigma_freq_ratio = config_sbi.sigma_freq_ratio

        # --- observed ---
        obs_wave_norm = stats_norm["wavelength"]
        obs_wave_pert = stats_pert["wavelength"]
        obs_amp_ratio = stats_pert["variance_amplitude"] / (stats_norm["variance_amplitude"] + EPS)
        obs_freq_ratio = stats_pert["peak_freq"] / (stats_norm["peak_freq"] + EPS)

        # --- log-likelihood ---
        ll = 0.0
        ll += gaussian_loglike(obs_wave_norm, mu_wave_norm, sigma_wave)
        ll += gaussian_loglike(obs_wave_pert, mu_wave_pert, sigma_wave)
        ll += gaussian_loglike(obs_amp_ratio, mu_amp_ratio, sigma_amp_ratio)
        ll += gaussian_loglike(obs_freq_ratio, mu_freq_ratio, sigma_freq_ratio)

        likelihood = float(np.exp(ll))

        row = {
            "filename": fid,
            "likelihood": likelihood,
            "wavelength_norm": obs_wave_norm,
            "wavelength_pert": obs_wave_pert,
            "variance_amp_norm": stats_norm["variance_amplitude"],
            "variance_amp_pert": stats_pert["variance_amplitude"],
            "freq_ratio": obs_freq_ratio,
            "error": error.strip(),
            **{k: params_norm.get(k, np.nan) for k in ["beta", "mu", "eta", "zeta", "mu_a", "fstar"]},
            **sampler,
        }

        results.append(row)
        processed.add(fid)
        logging.info(f"S2 processed: {fid}")

    return results


# -------------------------------------------------------
# STAGE 3
# -------------------------------------------------------

def collect_S3_once(processed: set) -> list[dict]:
    """
    Collect new paired (S3) results.
    Same structure as S2, but also computes Q_norm and Q_pert
    and applies the "floor" likelihood for Q:
        If Q >= Q_min → contribution 1 (loglike=0)
        Else          → Gaussian penalty with sigma_Q
    Includes robust error handling and per-simulation diagnostics.
    """
    results = []
    filepaths = sorted(glob.glob(os.path.join(OUTPUT_DATA_SBI_S3_DIR, "S3_*.npz")))
    for fp in filepaths:
        fid = os.path.basename(fp)
        if fid in processed:
            continue

        try:
            data = load_npz(fp)
        except Exception as e:
            logging.error(f"Error loading {fid}: {e}")
            continue
        sampler = data.get("sampler", None)
        sampler = dict(sampler.item()) if hasattr(sampler, "item") else dict(sampler) if sampler is not None else {}

        error = ""

        try:
            gamma_norm = data.get("gamma_norm", None)
            gamma_pert = data.get("gamma_pert", None)
            params_norm = dict(data["params_norm"].item()) if "params_norm" in data else {}

            # --- analyze both (with detailed errors) ---
            stats_norm, err_norm = compute_single_likelihood(gamma_norm)
            stats_pert, err_pert = compute_single_likelihood(gamma_pert)
            if err_norm:
                error += f"[norm] {err_norm} "
            if err_pert:
                error += f"[pert] {err_pert} "

            # invalid check
            invalid = (
                np.isnan(stats_norm["wavelength"])
                or np.isnan(stats_norm["variance_amplitude"])
                or np.isnan(stats_norm["peak_freq"])
                or np.isnan(stats_pert["wavelength"])
                or np.isnan(stats_pert["variance_amplitude"])
                or np.isnan(stats_pert["peak_freq"])
            )

            if invalid:
                row = {
                    "filename": fid,
                    "likelihood": 0.0,
                    "error": error.strip(),
                    "Q_norm": np.nan,
                    "Q_pert": np.nan,
                    "wavelength_norm": stats_norm["wavelength"],
                    "wavelength_pert": stats_pert["wavelength"],
                    "variance_amp_norm": stats_norm["variance_amplitude"],
                    "variance_amp_pert": stats_pert["variance_amplitude"],
                    "freq_ratio": np.nan,
                    **{k: params_norm.get(k, np.nan) for k in ["beta", "mu", "eta", "zeta", "mu_a", "fstar"]},
                    **sampler,
                }
                results.append(row)
                processed.add(fid)
                logging.info(f"S3 processed (invalid): {fid}")
                continue

            # --- Compute quality metrics ---
            
            q_norm = np.log10(compute_quality_Q(gamma_norm))
            q_pert = np.log10(compute_quality_Q(gamma_pert))
            

            # --- expected values (from config) ---
            mu_wave_norm = config_sbi.expected_wavelength_norm
            mu_wave_pert = config_sbi.expected_wavelength_pert
            sigma_wave = config_sbi.sigma_wavelength

            mu_amp_ratio = config_sbi.expected_amplitude_ratio
            sigma_amp_ratio = config_sbi.sigma_amplitude_ratio

            mu_freq_ratio = config_sbi.expected_freq_ratio
            sigma_freq_ratio = config_sbi.sigma_freq_ratio

            Q_min_norm = config_sbi.logQ_min_norm
            Q_min_pert = config_sbi.logQ_min_pert
            sigma_Q = config_sbi.sigma_logQ

            # --- observed ---
            obs_wave_norm = stats_norm["wavelength"]
            obs_wave_pert = stats_pert["wavelength"]
            obs_amp_ratio = stats_pert["variance_amplitude"] / (stats_norm["variance_amplitude"] + EPS)
            obs_freq_ratio = stats_pert["peak_freq"] / (stats_norm["peak_freq"] + EPS)

            # --- joint log-likelihood ---
            ll = 0.0
            ll += gaussian_loglike(obs_wave_norm, mu_wave_norm, sigma_wave)
            ll += gaussian_loglike(obs_wave_pert, mu_wave_pert, sigma_wave)
            ll += gaussian_loglike(obs_amp_ratio, mu_amp_ratio, sigma_amp_ratio)
            ll += gaussian_loglike(obs_freq_ratio, mu_freq_ratio, sigma_freq_ratio)
            ll += gaussian_loglike(q_norm, Q_min_norm, sigma_Q)
            ll += gaussian_loglike(q_pert, Q_min_pert, sigma_Q)

            likelihood = float(np.exp(ll))

            row = {
                "filename": fid,
                "likelihood": likelihood,
                "wavelength_norm": obs_wave_norm,
                "wavelength_pert": obs_wave_pert,
                "variance_amp_norm": stats_norm["variance_amplitude"],
                "variance_amp_pert": stats_pert["variance_amplitude"],
                "freq_ratio": obs_freq_ratio,
                "Q_norm": q_norm,
                "Q_pert": q_pert,
                "error": error.strip() or None,
                **{k: params_norm.get(k, np.nan) for k in ["beta", "mu", "eta", "zeta", "mu_a", "fstar"]},
                **sampler,
            }

        except Exception as e:
            # major failure fallback
            params_norm = {}
            try:
                if "params_norm" in data:
                    params_norm = dict(data["params_norm"].item())
            except Exception:
                pass
            row = {
                "filename": fid,
                "likelihood": 0.0,
                "error": str(e),
                "Q_norm": np.nan,
                "Q_pert": np.nan,
                "wavelength_norm": np.nan,
                "wavelength_pert": np.nan,
                "variance_amp_norm": np.nan,
                "variance_amp_pert": np.nan,
                "freq_ratio": np.nan,
                **{k: params_norm.get(k, np.nan) for k in ["beta", "mu", "eta", "zeta", "mu_a", "fstar"]},
                **sampler,
            }

        results.append(row)
        processed.add(fid)
        logging.info(f"S3 processed: {fid}")

    return results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    setup_logging()
    logging.info("Collector for S1, S2 & S3 started...")
    processed_s1, processed_s2, processed_s3 = set(), set(), set()

    # preload existing
    if os.path.exists(SUMMARY_S1):
        df1 = pd.read_csv(SUMMARY_S1)
        if "filename" in df1.columns:
            processed_s1.update(df1["filename"].astype(str))
    if os.path.exists(SUMMARY_S2):
        df2 = pd.read_csv(SUMMARY_S2)
        if "filename" in df2.columns:
            processed_s2.update(df2["filename"].astype(str))
    if os.path.exists(SUMMARY_S3):
        df3 = pd.read_csv(SUMMARY_S3)
        if "filename" in df3.columns:
            processed_s3.update(df3["filename"].astype(str))

    while True:
        # --- S1 ---
        new1 = collect_S1_once(processed_s1)
        if new1:
            df_new = pd.DataFrame(new1)
            if os.path.exists(SUMMARY_S1):
                df_all = pd.concat([pd.read_csv(SUMMARY_S1), df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(SUMMARY_S1, index=False)
            logging.info(f"Updated {SUMMARY_S1} ({len(df_all)} rows)")

        # --- S2 ---
        new2 = collect_S2_once(processed_s2)
        if new2:
            df_new = pd.DataFrame(new2)
            if os.path.exists(SUMMARY_S2):
                df_all = pd.concat([pd.read_csv(SUMMARY_S2), df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(SUMMARY_S2, index=False)
            logging.info(f"Updated {SUMMARY_S2} ({len(df_all)} rows)")

        # --- S3 ---
        new3 = collect_S3_once(processed_s3)
        if new3:
            df_new = pd.DataFrame(new3)
            if os.path.exists(SUMMARY_S3):
                df_all = pd.concat([pd.read_csv(SUMMARY_S3), df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(SUMMARY_S3, index=False)
            logging.info(f"Updated {SUMMARY_S3} ({len(df_all)} rows)")

        if not new1 and not new2 and not new3:
            logging.info("No new results. Waiting...")
        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    main()
