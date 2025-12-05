import argparse
import logging
import sys
import os
import numpy as np
from os.path import join
import time
import signal

from utils_rdf.simulate_spde_capi import simulate_episode
import config_sbi
from utils_rdf.config import OUTPUT_DATA_SBI_S1_DIR
from utils_sbi import sample_from_prior, prior_settings_hash, load_s1_data, fit_gp_scalar, fit_gp_safe, fit_classifier, acquisition_scores


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
    strat = args.sampler

    #  PRIOR
    if strat == "prior":
        s, k = sample_from_prior(rng, 1)
        config_sampler = {"type": strat, "hash": prior_settings_hash()}
        return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler

    #  GP
    if strat == "gp":
        csv = join(OUTPUT_DATA_SBI_S1_DIR, "summary_s1.csv")
        n_cand = args.gp_candidates
        X, y_lik, y_bool, y_wave, feature_keys = load_s1_data(csv)
        y_best = np.max(y_lik)

        if X is None:
            logging.warning("No data found — fallback to prior.")
            s, k = sample_from_prior(rng, 1)
            config_sampler = {"type": "prior", "hash": prior_settings_hash()}
            return {key: float(s[0, i]) for i, key in enumerate(k)}, config_sampler
        
        # Draw candidate points from prior
        Xcand, keys = sample_from_prior(rng, n_cand)
        Xcand_log = np.log10(Xcand + 1e-50)
        Xcand_log[:, 1] = Xcand[:, 1]  # keep eta linear

        # Fit GP(s)
        if args.gpfit == "LIK":
            gp_lik = fit_gp_scalar(X, y_lik)
            mu_lik, sigma_lik = gp_lik.predict(Xcand_log, return_std=True)

        elif args.gpfit == "ALL":
            gp_wave = fit_gp_safe(X, np.log(y_wave))
            clf = fit_classifier(X, y_bool)
            # Predict wavelength GP
            mu_wave_log, sigma_wave_log = gp_wave.predict(Xcand_log, return_std=True)
            mu_wave = np.exp(mu_wave_log)
            sigma_wave =  mu_wave * sigma_wave_log

            # Classifier probability for oscillatory behavior
            P_osc = clf.predict_proba(Xcand_log)[:, 1]
            logP = np.log(P_osc + 1e-50)

            # Convert wavelength to "likelihood-like" term
            expected = config_sbi.expected_wavelength_pert
            sigma_target = config_sbi.sigma_wavelength

            # log-likelihood based on expected wavelength
            logL_w = -0.5 * ((mu_wave - expected) / sigma_target) ** 2
            # Combined "log-likelihood" mean and uncertainty proxy
            mu_lik = logP + logL_w
            sigma_lik = sigma_wave  
        else:
            raise ValueError(f"Unknown gpfit mode: {args.gpfit}")
        
        sigma_lik=np.clip(sigma_lik,0,config_sbi.UPPER_SIGMA_BOUND)
        
        scores = acquisition_scores(args.acq, mu_lik, sigma_lik, y_best, args.ucb_beta)

        i_best = int(np.argmax(scores))
        theta = {k: float(Xcand[i_best, j]) for j, k in enumerate(keys)}
       
        config_sampler = {
            "type": strat+args.gpfit+args.acq,
            "hash": prior_settings_hash()+str(args.ucb_beta),
        }

        return theta, config_sampler

# -----------------------------------------------------------
# main routine
# -----------------------------------------------------------

def run_stage1(args):
    all_strategies = [
        {"sampler": "prior"},  # simple prior draw
        {"sampler": "gp", "gpfit": "LIK", "acq": "EI"},
        {"sampler": "gp", "gpfit": "LIK", "acq": "UCB"},
        {"sampler": "gp", "gpfit": "LIK", "acq": "MES"},
        {"sampler": "gp", "gpfit": "ALL", "acq": "EI"},
        {"sampler": "gp", "gpfit": "ALL", "acq": "UCB"},
        {"sampler": "gp", "gpfit": "ALL", "acq": "MES"},
    ]

    setup_logging(args.log)
    seed = args.seed
    rng = np.random.default_rng(seed)

    os.makedirs(OUTPUT_DATA_SBI_S1_DIR, exist_ok=True)
    logging.info("=== Stage 1: short simulations ===")
    logging.info(f"Strategy: {args.sampler} | n_samples: {args.n_samples}")

    base_params = dict(config_sbi.fixed_params)
    base_params["T"] = config_sbi.T_S1

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
        params = dict(base_params)
        params.update({k: float(theta[k]) for k in theta.keys()})
        
        params["mu_a"] = np.sqrt(params["mu_a_times_zeta"] * params["mu_a_div_zeta"])
        params["zeta"] = params["mu_a"] / params["mu_a_div_zeta"]

        params_original = params.copy()

        # Apply reduction only for simulation
        params["mu_a"] = config_sbi.rel_motor_extraction * params_original["mu_a"]
        params["Nmotor"] = config_sbi.rel_motor_extraction * params_original["Nmotor"]

        logging.info(
            f"[{i+1}/{args.n_samples}] Sim mu_a(pert)={params['mu_a']:.3g}, "
            f"mu={params['mu']:.3g}, eta={params['eta']:.3g}, "
            f"zeta={params['zeta']:.3g}, seed={seed}"
        )

        try:
            t, gamma_mat, nplus_mat, nminus_mat = simulate_episode(params, seed%10000)

            # Save both original and perturbed parameters
            tag = f"sim{i:04d}_seed{seed}"
            rawpath = os.path.join(OUTPUT_DATA_SBI_S1_DIR, f"{tag}.npz")
            np.savez_compressed(
                rawpath,
                t=t,
                gamma_mat=gamma_mat,
                nplus_mat=nplus_mat,
                nminus_mat=nminus_mat,
                params=params_original,       
                params_pert=params,            
                seed=seed,
                error=None,
                sampler=sampler_config
            )

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Exiting gracefully...")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Simulation failed: {e}")
            # still save parameters + error for debugging
            tag = f"sim{i:04d}_seed{seed}_FAILED"
            rawpath = os.path.join(OUTPUT_DATA_SBI_S1_DIR, f"{tag}.npz")
            np.savez_compressed(
                rawpath,
                params=params,
                seed=seed,
                error=str(e)
            )
            continue


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 SBI: short simulations with prior sampling")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of simulations to run")
    parser.add_argument("--log", type=str, default=None, help="Optional log file path")

    # CLI args
    parser.add_argument("--sampler", type=str, default="rotate",
                        choices=["prior", "gp", "rotate"],
                        help="Sampling strategy")
    # GP controls
    parser.add_argument("--gpfit", type=str, default="ALL",
                        choices=["LIK", "ALL"],
                        help="Likelihood or separated")
    parser.add_argument("--acq", type=str, default="UCB",
                        choices=["EI", "UCB", "MES"],
                        help="Acquisition for GP sampler")
    parser.add_argument("--ucb-beta", type=float, default=2.0,
                        help="UCB beta (only for --acq UCB)")
    parser.add_argument("--gp-candidates", type=int, default=4000,
                        help="Number of random candidates for acquisition maximization")

    # seed 
    parser.add_argument("--seed", type=int, default=int(time.time()*100),
                        help="RNG seed for the sampler")

    args = parser.parse_args()
    run_stage1(args)

