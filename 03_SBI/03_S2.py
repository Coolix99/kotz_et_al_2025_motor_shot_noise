import argparse
import logging
import sys
import os
import numpy as np
from os.path import join
import time


from utils_rdf.simulate_spde_capi import simulate_episode
import config_sbi
from utils_rdf.config import OUTPUT_DATA_SBI_S1_DIR, OUTPUT_DATA_SBI_S2_DIR
from utils_sbi import sample_from_prior, prior_settings_hash, load_s1_data, fit_gp_scalar, fit_gp_safe, fit_classifier, acquisition_scores, load_s2_data
import sys

import signal

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
# propose parameters
# -----------------------------------------------------------
def _propose_params(args, rng: np.random.Generator):
    """
    Propose parameters for S2 (paired simulations).
    Reuses GP-based proposal logic from Stage 1 (BEFORE) or Stage 2 (CURRENT) summary.
    """
    strat = args.sampler

    # --- PRIOR ---
    if strat == "prior":
        s, k = sample_from_prior(rng, 1)
        config_sampler = {"type": strat, "hash": prior_settings_hash()}
        return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler

    # --- GP ---
    if strat == "gp":
        # Choose source of training data
        if args.stage == "BEFORE":
            X, y_lik, y_bool, y_wave, feature_keys = load_s1_data(
                join(OUTPUT_DATA_SBI_S1_DIR, "summary_s1.csv")
            )
        elif args.stage == "CURRENT":
            X, y_lik, y_bool, y_aux, feature_keys = load_s2_data(
                join(OUTPUT_DATA_SBI_S2_DIR, "summary_s2.csv")
            )
        else:
            raise ValueError(f"Invalid stage: {args.stage}")

        # Fallback to prior if no data
        if X is None:
            logging.warning("No training data found — fallback to prior.")
            s, k = sample_from_prior(rng, 1)
            config_sampler = {"type": "prior", "hash": prior_settings_hash()}
            return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler

        y_best = np.max(y_lik)

        # --- Candidate pool ---
        n_cand = args.gp_candidates
        Xcand, keys = sample_from_prior(rng, n_cand)
        Xcand_log = np.log10(Xcand + 1e-50)
        Xcand_log[:, 1] = Xcand[:, 1]  # keep η linear

        # --- choose GP mode ---
        if args.gpfit == "LIK":
            gp_lik = fit_gp_scalar(X, y_lik)
            mu_lik, sigma_lik = gp_lik.predict(Xcand_log, return_std=True)

        elif args.gpfit == "ALL":
            if args.stage == "BEFORE":
                # --- use wavelength GP and oscillation classifier ---
                gp_wave = fit_gp_safe(X, np.log(y_wave))
                clf = fit_classifier(X, y_bool)
                mu_wave_log, sigma_wave_log = gp_wave.predict(Xcand_log, return_std=True)
                mu_wave = np.exp(mu_wave_log)
                sigma_wave =  mu_wave * sigma_wave_log
                P_osc = clf.predict_proba(Xcand_log)[:, 1]
                logP = np.log(P_osc + 1e-50)

                expected = config_sbi.expected_wavelength_pert
                sigma_target = config_sbi.sigma_wavelength
                logL_w = -0.5 * ((mu_wave - expected) / sigma_target) ** 2

                mu_lik = logP + logL_w
                sigma_lik = sigma_wave

            elif args.stage == "CURRENT":
                # --- classifier for oscillatory vs non-oscillatory ---
                clf = fit_classifier(X, y_bool)
                P_osc = clf.predict_proba(Xcand_log)[:, 1]
                logP = np.log(P_osc + 1e-50)

                # --- GPs for direct observables ---
                # wavelength (norm + pert)
                gp_wave_norm = fit_gp_safe(X, np.log(y_aux["wavelength_norm"]))
                mu_wave_norm_log, sigma_wave_norm_log = gp_wave_norm.predict(Xcand_log, return_std=True)
                mu_wave_norm = np.exp(mu_wave_norm_log)
                sigma_wave_norm =  mu_wave_norm * sigma_wave_norm_log

                gp_wave_pert = fit_gp_safe(X, np.log(y_aux["wavelength_pert"]))
                mu_wave_pert_log, sigma_wave_pert_log = gp_wave_pert.predict(Xcand_log, return_std=True)
                mu_wave_pert = np.exp(mu_wave_pert_log)
                sigma_wave_pert =  mu_wave_pert * sigma_wave_pert_log


                # amplitude ratio
                amp_ratio = np.asarray(y_aux["variance_amp_pert"]) / (
                    np.asarray(y_aux["variance_amp_norm"]) + 1e-50
                )
                gp_amp_ratio = fit_gp_safe(X, amp_ratio)
                mu_amp_ratio, sigma_amp_ratio = gp_amp_ratio.predict(Xcand_log, return_std=True)

                # frequency ratio
                gp_freq_ratio = fit_gp_safe(X, y_aux["freq_ratio"])
                mu_freq_ratio, sigma_freq_ratio = gp_freq_ratio.predict(Xcand_log, return_std=True)

                # --- expected values & sigmas (from config_sbi) ---
                mu_wave_norm = config_sbi.expected_wavelength_norm
                mu_wave_pert = config_sbi.expected_wavelength_pert
                sigma_wave   = config_sbi.sigma_wavelength

                mu_amp_ratio = config_sbi.expected_amplitude_ratio
                sigma_amp    = config_sbi.sigma_amplitude_ratio

                mu_freq_ratio = config_sbi.expected_freq_ratio
                sigma_freq    = config_sbi.sigma_freq_ratio

                # --- pseudo-loglikelihood contributions ---
                logL_wave_norm = -0.5 * ((mu_wave_norm - mu_wave_norm) / sigma_wave) ** 2
                logL_wave_pert = -0.5 * ((mu_wave_pert - mu_wave_pert) / sigma_wave) ** 2
                logL_amp_ratio = -0.5 * ((mu_amp_ratio - mu_amp_ratio) / sigma_amp) ** 2
                logL_freq_ratio = -0.5 * ((mu_freq_ratio - mu_freq_ratio) / sigma_freq) ** 2

                # combine (sum) + oscillation probability
                mu_lik = logP + logL_wave_norm + logL_wave_pert + logL_amp_ratio + logL_freq_ratio
                sigma_lik = np.sqrt(
                    sigma_wave_norm**2 + sigma_wave_pert**2 + sigma_amp_ratio**2 + sigma_freq_ratio**2
                )

        else:
            raise ValueError(f"Unknown gpfit mode: {args.gpfit}")

        # --- acquisition step ---
        sigma_lik = np.clip(sigma_lik, 0, config_sbi.UPPER_SIGMA_BOUND)
        scores = acquisition_scores(args.acq, mu_lik, sigma_lik, y_best, args.ucb_beta)
        i_best = int(np.argmax(scores))

        theta = {k: float(Xcand[i_best, j]) for j, k in enumerate(keys)}
        config_sampler = {
            "type": f"{strat}_{args.gpfit}_{args.acq}_{args.stage}",
            "hash": prior_settings_hash() + f"_{args.ucb_beta}",
        }
        return theta, config_sampler

    raise ValueError(f"Unknown sampler: {strat}")


# -----------------------------------------------------------
# main routine
# -----------------------------------------------------------

def run_stage2(args):
    all_strategies = [
        # --- Baseline ---
        {"sampler": "prior"},  # simple prior draw

        # --- GP-based (Stage 1 summary: BEFORE) ---
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "EI"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "UCB"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "LIK", "acq": "MES"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "EI"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "UCB"},
        {"sampler": "gp", "stage": "BEFORE", "gpfit": "ALL", "acq": "MES"},

        # --- GP-based (Stage 2 summary: CURRENT) ---
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "EI"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "UCB"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "LIK", "acq": "MES"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "EI"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "UCB"},
        {"sampler": "gp", "stage": "CURRENT", "gpfit": "ALL", "acq": "MES"},
    ]


    setup_logging(args.log)
    seed = args.seed
    rng = np.random.default_rng(seed)

    os.makedirs(OUTPUT_DATA_SBI_S2_DIR, exist_ok=True)
    logging.info("=== Stage 2: pai simulations ===")
    logging.info(f"Strategy: {args.sampler} | n_samples: {args.n_samples}")

    base_params = dict(config_sbi.fixed_params)
    base_params["T"] = config_sbi.T_S2

    for i in range(args.n_samples):
        if STOP_REQUESTED:
            logging.info("Graceful stop requested. Exiting Stage 2 loop.")
            break

        # Select sampler configuration
        if args.sampler == "rotate":
            strat_cfg = all_strategies[i % len(all_strategies)]
            current_sampler = strat_cfg["sampler"]
            args_local = argparse.Namespace(**{**vars(args), **strat_cfg})
            logging.info(f"[Rotation mode] Using sampler: {current_sampler} "
                         f"({strat_cfg.get('gpfit', '')}, {strat_cfg.get('acq', '')})")
        else:
            args_local = args
            current_sampler = args.sampler

        # propose θ
        theta, sampler_config = _propose_params(argparse.Namespace(**{**vars(args_local), "sampler": current_sampler}), rng)
        params_norm = dict(base_params)
        params_norm.update({k: float(theta[k]) for k in theta.keys()})
        params_norm["mu_a"] = np.sqrt(params_norm["mu_a_times_zeta"] * params_norm["mu_a_div_zeta"])
        params_norm["zeta"] = params_norm["mu_a"] / params_norm["mu_a_div_zeta"]
        
        # perturbed: Nmotor & mu_a scaled by rel_motor_extraction
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
            f"Nmotor(pert)={params_pert['Nmotor']} | mu_a(pert)={params_pert['mu_a']:.3g}"
        )

        # run two simulations
        try:
            t1, g1, nplus1, nminus1 = simulate_episode(params_norm, seed_norm)
        except Exception as e:
            logging.error(f"Simulation (normal) failed: {e}")
            tag = f"S2_{i:04d}_FAILED_norm"
            np.savez_compressed(join(OUTPUT_DATA_SBI_S2_DIR, f"{tag}.npz"),
                                params_norm=params_norm, seed_norm=seed_norm, error=str(e))
            continue

        try:
            t2, g2, nplus2, nminus2 = simulate_episode(params_pert, seed_pert)
        except Exception as e:
            logging.error(f"Simulation (perturbed) failed: {e}")
            tag = f"S2_{i:04d}_FAILED_pert"
            np.savez_compressed(join(OUTPUT_DATA_SBI_S2_DIR, f"{tag}.npz"),
                                params_pert=params_pert, seed_pert=seed_pert, error=str(e))
            continue

        # save pair together
        tag = f"S2_{i:04d}_seedN{seed_norm}_seedP{seed_pert}"
        rawpath = join(OUTPUT_DATA_SBI_S2_DIR, f"{tag}.npz")

        

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
        logging.info(f"Saved S2 pair")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 SBI: pair simulations")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of simulations to run")
    parser.add_argument("--log", type=str, default=None, help="Optional log file path")

    # CLI args
    parser.add_argument("--sampler", type=str, default="rotate",
                        choices=["prior", "gp", "rotate"],
                        help="Sampling strategy")
    # GP controls
    parser.add_argument("--stage", type=str, default="BEFORE",
                        choices=["BEFORE", "CURRENT"],
                        help="Current or stage before")
    parser.add_argument("--gpfit", type=str, default="LIK",
                        choices=["LIK", "ALL"],
                        help="Likelihood or separated")
    parser.add_argument("--acq", type=str, default="UCB",
                        choices=["EI", "UCB", "MES"],
                        help="Acquisition for GP sampler")
    parser.add_argument("--ucb-beta", type=float, default=2.0,
                        help="UCB beta (only for --acq UCB)")
    parser.add_argument("--gp-candidates", type=int, default=10000,
                        help="Number of random candidates for acquisition maximization")

    # seed 
    parser.add_argument("--seed", type=int, default=int(time.time()*100),
                        help="RNG seed for the sampler")

    args = parser.parse_args()
    run_stage2(args)

