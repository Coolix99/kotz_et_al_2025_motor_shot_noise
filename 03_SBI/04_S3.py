import argparse
import logging
import sys
import os
from os.path import join
import time
import numpy as np
import signal

from utils_rdf.simulate_spde_capi import simulate_episode
import config_sbi

from utils_rdf.config import (
    OUTPUT_DATA_SBI_S2_DIR,
    OUTPUT_DATA_SBI_S3_DIR,
)
from utils_sbi import (
    sample_from_prior,
    prior_settings_hash,
    load_s2_data,        
    load_s3_data,
    fit_gp_scalar,
    fit_gp_safe,
    fit_classifier,
    acquisition_scores,
)

STOP_REQUESTED = False
def handle_sigint(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("SIGINT received — will stop after current simulation.")
signal.signal(signal.SIGINT, handle_sigint)

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
def setup_logging(logfile=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


# -----------------------------------------------------------
# Propose parameters (PRIOR or GP with LIKE/ALL & EI/UCB/MES)
# -----------------------------------------------------------
def _propose_params(args, rng: np.random.Generator):
    """
    - args.sampler in {"prior","gp"}
    - args.stage in {"BEFORE","CURRENT"} decides which summary to learn from:
        BEFORE  -> S2
        CURRENT -> S3
    - args.gpfit in {"LIK","ALL"} selects the GP target composition
    - args.acq in {"EI","UCB","MES"} for acquisition scoring
    """
    strat = args.sampler

    # --- PRIOR ---
    if strat == "prior":
        s, k = sample_from_prior(rng, 1)
        config_sampler = {
            "type": f"prior_{args.stage}",  # tag stage for traceability
            "hash": prior_settings_hash()
        }
        return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler

    # --- GP (BEFORE -> S2 data, CURRENT -> S3 data) ---
    if strat == "gp":
        if args.stage == "BEFORE":
            csv = join(OUTPUT_DATA_SBI_S2_DIR, "summary_s2.csv")
            X, y_lik, y_bool, y_aux, feature_keys = load_s2_data(csv)
        elif args.stage == "CURRENT":
            csv = join(OUTPUT_DATA_SBI_S3_DIR, "summary_s3.csv")
            X, y_lik, y_bool, y_aux, feature_keys = load_s3_data(csv)
        else:
            raise ValueError(f"Invalid stage: {args.stage}")

        if X is None:
            logging.warning("No training data found — fallback to prior.")
            s, k = sample_from_prior(rng, 1)
            config_sampler = {"type": f"prior_{args.stage}", "hash": prior_settings_hash()}
            return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler

        y_best = float(np.max(y_lik))

        # Candidate pool from prior
        Xcand, keys = sample_from_prior(rng, args.gp_candidates)
        Xcand_log = np.log10(Xcand + 1e-50)
        Xcand_log[:, 1] = Xcand[:, 1]  # keep eta linear

        # Choose GP “fit” flavor — same as S1/S2
        if args.gpfit == "LIK":
            gp_lik = fit_gp_scalar(X, y_lik)
            mu_lik, sigma_lik = gp_lik.predict(Xcand_log, return_std=True)

        elif args.gpfit == "ALL":
            # ------------------------------
            # Classifier for oscillation
            # ------------------------------
            clf = fit_classifier(X, y_bool)
            P_osc_cand = clf.predict_proba(Xcand_log)[:, 1]
            logP_osc   = np.log(P_osc_cand + 1e-50)

            # ------------------------------
            # Expected values & sigmas (from config)
            # ------------------------------
            mu_wave_norm_exp = config_sbi.expected_wavelength_norm
            mu_wave_pert_exp = config_sbi.expected_wavelength_pert
            sigma_wave       = config_sbi.sigma_wavelength

            mu_amp_ratio_exp = config_sbi.expected_amplitude_ratio
            sigma_amp        = config_sbi.sigma_amplitude_ratio

            mu_freq_ratio_exp = config_sbi.expected_freq_ratio
            sigma_freq        = config_sbi.sigma_freq_ratio

            # ------------------------------
            # Fit GPs on PHYSICAL VALUES
            # ------------------------------

            # wavelength_norm (log)
            y_wavenorm = np.log(np.clip(y_aux["wavelength_norm"], 1e-50, None))
            gp_wave_norm = fit_gp_safe(X, y_wavenorm)
            mu_log, sig_log = gp_wave_norm.predict(Xcand_log, return_std=True)
            mu_wave_norm = np.exp(mu_log)
            sigma_wave_norm = mu_wave_norm * sig_log

            # wavelength_pert (log)
            y_wavepert = np.log(np.clip(y_aux["wavelength_pert"], 1e-50, None))
            gp_wave_pert = fit_gp_safe(X, y_wavepert)
            mu_log, sig_log = gp_wave_pert.predict(Xcand_log, return_std=True)
            mu_wave_pert = np.exp(mu_log)
            sigma_wave_pert = mu_wave_pert * sig_log

            # amplitude ratio
            amp_ratio = np.asarray(y_aux["variance_amp_pert"], float) / (
                np.asarray(y_aux["variance_amp_norm"], float) + 1e-50
            )
            gp_amp_ratio = fit_gp_safe(X, amp_ratio)
            mu_amp_ratio, sigma_amp_ratio = gp_amp_ratio.predict(Xcand_log, return_std=True)

            # frequency ratio
            gp_freq_ratio = fit_gp_safe(X, y_aux["freq_ratio"])
            mu_freq_ratio, sigma_freq_ratio = gp_freq_ratio.predict(Xcand_log, return_std=True)

            ll_Q_norm = ll_Q_pert = np.zeros_like(Xcand_log[:, 0])
            sigma_Q_pert = sigma_Q_norm = np.zeros_like(Xcand_log[:, 0])
            
            # ------------------------------
            #  Compute log-likelihoods AFTER prediction
            # ------------------------------
            def ll_gauss(x, mu, sigma):
                return -0.5 * ((x - mu) / (sigma + 1e-50)) ** 2

            def ll_qfloor(q, q_min, sigma_q):
                if q is None:
                    return np.zeros_like(q)
                q = np.asarray(q, float)
                ll = np.zeros_like(q)
                mask = (q < q_min) & np.isfinite(q)
                ll[mask] = -0.5 * ((q[mask] - q_min) / (sigma_q + 1e-50)) ** 2
                return ll
            
            # optional S3 Q metrics
            if args.stage == "CURRENT":
                Q_min_norm = config_sbi.logQ_min_norm
                Q_min_pert = config_sbi.logQ_min_pert
                sigma_logQ = config_sbi.sigma_logQ

                gp_Q_norm = fit_gp_safe(X, y_aux["Q_norm"])
                mu_Q_norm, sigma_Q_norm = gp_Q_norm.predict(Xcand_log, return_std=True)
                gp_Q_pert = fit_gp_safe(X, y_aux["Q_pert"])
                mu_Q_pert, sigma_Q_pert = gp_Q_pert.predict(Xcand_log, return_std=True)

                ll_Q_norm = ll_qfloor(mu_Q_norm, Q_min_norm, sigma_logQ)
                ll_Q_pert = ll_qfloor(mu_Q_pert, Q_min_pert, sigma_logQ)
            

            ll_wave_norm = ll_gauss(mu_wave_norm, mu_wave_norm_exp, sigma_wave)
            ll_wave_pert = ll_gauss(mu_wave_pert, mu_wave_pert_exp, sigma_wave)
            ll_amp_ratio = ll_gauss(mu_amp_ratio, mu_amp_ratio_exp, sigma_amp)
            ll_freq_ratio = ll_gauss(mu_freq_ratio, mu_freq_ratio_exp, sigma_freq)
            
            # ------------------------------
            #  Combine total μ and σ
            # ------------------------------
            mu_lik = (
                ll_wave_norm
                + ll_wave_pert
                + ll_amp_ratio
                + ll_freq_ratio
                + ll_Q_norm
                + ll_Q_pert
                + logP_osc
            )

            sigma_lik = np.sqrt(
                sigma_wave_norm**2
                + sigma_wave_pert**2
                + sigma_amp_ratio**2
                + sigma_freq_ratio**2
                + sigma_Q_norm**2
                + sigma_Q_pert**2
            )

        else:
            raise ValueError(f"Unknown gpfit: {args.gpfit}")

        # Clamp sigma to avoid exploding UCB/MES
        sigma_lik = np.clip(sigma_lik, 0.0, config_sbi.UPPER_SIGMA_BOUND)

        scores = acquisition_scores(args.acq, mu_lik, sigma_lik, y_best, args.ucb_beta)
        i_best = int(np.argmax(scores))

        theta = {k: float(Xcand[i_best, j]) for j, k in enumerate(keys)}
        config_sampler = {
            "type": f"gp_{args.gpfit}_{args.acq}_{args.stage}",
            "hash": prior_settings_hash() + f"_{args.ucb_beta}",
        }
        return theta, config_sampler

    raise ValueError(f"Unknown sampler: {strat}")

# -----------------------------------------------------------
# Main routine
# -----------------------------------------------------------
def run_stage3(args):
    all_strategies = [
        # --- Baseline ---
        {"sampler": "prior"},  # simple prior draw

        # --- GP-based (Stage 2 summary: BEFORE) ---
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "EI"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "UCB"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "MES"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "EI"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "UCB"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "MES"},

        # --- GP-based (Stage 3 summary: CURRENT) ---
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "EI"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "UCB"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "MES"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "EI"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "UCB"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "MES"},
    ]
    setup_logging(args.log)
    rng = np.random.default_rng(args.seed)

    os.makedirs(OUTPUT_DATA_SBI_S3_DIR, exist_ok=True)
    logging.info("=== Stage 3: paired simulations (longer) ===")
    logging.info(f"Strategy: {args.sampler} | n_samples: {args.n_samples} | stage={args.stage}")

    base_params = dict(config_sbi.fixed_params)

    base_params["T"] = config_sbi.T_S3

    for i in range(args.n_samples):
        if STOP_REQUESTED:
            logging.info("Graceful stop requested. Exiting Stage 3 loop.")
            break

        # Select sampler configuration (rotate or fixed)
        if args.sampler == "rotate":
            strat_cfg = all_strategies[i % len(all_strategies)]
            current_sampler = strat_cfg["sampler"]
            args_local = argparse.Namespace(**{**vars(args), **strat_cfg})
            logging.info(f"[Rotation] sampler={current_sampler} (gpfit={strat_cfg.get('gpfit','')}, acq={strat_cfg.get('acq','')})")
        else:
            args_local = args
            current_sampler = args.sampler

        # Propose θ in the same parameter space as S1/S2
        theta, sampler_config = _propose_params(
            argparse.Namespace(**{**vars(args_local), "sampler": current_sampler}), rng
        )

        # Construct normal params
        params_norm = dict(base_params)
        params_norm.update({k: float(theta[k]) for k in theta.keys()})
        # reconstruct mu_a, zeta from mu_a_times_zeta and mu_a_div_zeta in config/fixed
        params_norm["mu_a"] = np.sqrt(params_norm["mu_a_times_zeta"] * params_norm["mu_a_div_zeta"])
        params_norm["zeta"] = params_norm["mu_a"] / params_norm["mu_a_div_zeta"]

        # Perturbed scaling (same as S2)
        params_pert = dict(params_norm)
        nm0 = float(params_norm.get("Nmotor", config_sbi.fixed_params.get("Nmotor", 100)))
        params_pert["Nmotor"] = max(1, int(round(config_sbi.rel_motor_extraction * nm0)))
        params_pert["mu_a"]   = config_sbi.rel_motor_extraction * params_norm["mu_a"]

        seed_norm = int(rng.integers(1, 2**31 - 1))
        seed_pert = int(rng.integers(1, 2**31 - 1))

        logging.info(
            f"[{i+1}/{args.n_samples}] θ: mu_a={params_norm['mu_a']:.3g}, mu={params_norm['mu']:.3g}, "
            f"eta={params_norm['eta']:.3g}, zeta={params_norm['zeta']:.3g} | "
            f"Nmotor(norm)={params_norm.get('Nmotor', base_params.get('Nmotor', 100))} → "
            f"Nmotor(pert)={params_pert['Nmotor']} | mu_a(pert)={params_pert['mu_a']:.3g} | T={base_params['T']}"
        )

        # Run the two simulations (longer)
        try:
            t1, g1, nplus1, nminus1 = simulate_episode(params_norm, seed_norm)
        except Exception as e:
            logging.error(f"S3 simulation (normal) failed: {e}")
            tag = f"S3_{i:04d}_FAILED_norm"
            np.savez_compressed(join(OUTPUT_DATA_SBI_S3_DIR, f"{tag}.npz"),
                                params_norm=params_norm, seed_norm=seed_norm, error=str(e),
                                sampler=sampler_config, theta=theta)
            continue

        try:
            t2, g2, nplus2, nminus2 = simulate_episode(params_pert, seed_pert)
        except Exception as e:
            logging.error(f"S3 simulation (perturbed) failed: {e}")
            tag = f"S3_{i:04d}_FAILED_pert"
            np.savez_compressed(join(OUTPUT_DATA_SBI_S3_DIR, f"{tag}.npz"),
                                params_pert=params_pert, seed_pert=seed_pert, error=str(e),
                                sampler=sampler_config, theta=theta)
            continue

        # Save the pair — collector will compute likelihood & Q_* later
        tag = f"S3_{i:04d}_seedN{seed_norm}_seedP{seed_pert}"
        rawpath = join(OUTPUT_DATA_SBI_S3_DIR, f"{tag}.npz")
        np.savez_compressed(
            rawpath,
            # normal
            t_norm=t1, gamma_norm=g1, nplus_norm=nplus1, nminus_norm=nminus1,
            params_norm=params_norm, seed_norm=seed_norm,
            # perturbed
            t_pert=t2, gamma_pert=g2, nplus_pert=nplus2, nminus_pert=nminus2,
            params_pert=params_pert, seed_pert=seed_pert,
            # meta
            theta=theta,
            sampler=sampler_config,
            error=None
        )
        logging.info(f"Saved S3 pair: {rawpath}")

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 SBI: paired simulations (longer) with Q via collector")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of parameter proposals to simulate")
    parser.add_argument("--log", type=str, default=None, help="Optional log file path")

    # Sampling strategy
    parser.add_argument("--sampler", type=str, default="rotate",
                        choices=["prior", "gp", "rotate"],
                        help="Parameter proposal strategy")

    # BEFORE/CURRENT choice of training data (S2 vs S3)
    parser.add_argument("--stage", type=str, default="BEFORE",
                        choices=["BEFORE", "CURRENT"],
                        help="GP training data source: BEFORE=S2, CURRENT=S3")

    # GP controls
    parser.add_argument("--gpfit", type=str, default="ALL",
                        choices=["LIK", "ALL"], help="GP fit target composition")
    parser.add_argument("--acq", type=str, default="UCB",
                        choices=["EI", "UCB", "MES"], help="Acquisition")
    parser.add_argument("--ucb-beta", type=float, default=2.0, help="UCB beta")
    parser.add_argument("--gp-candidates", type=int, default=10000,
                        help="Number of candidate points from prior for acquisition")

    # RNG
    parser.add_argument("--seed", type=int, default=int(time.time()*100),
                        help="RNG seed")

    args = parser.parse_args()
    run_stage3(args)
