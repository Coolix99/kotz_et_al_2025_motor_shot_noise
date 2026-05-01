#!/usr/bin/env python3

import ctypes
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from local_config import CILIA_FOLDER
except ModuleNotFoundError:
    cfg_path = PROJECT_ROOT / "local_config.py"
    if not cfg_path.exists():
        raise
    import importlib.util
    spec = importlib.util.spec_from_file_location("local_config", str(cfg_path))
    local_config = importlib.util.module_from_spec(spec) # type: ignore
    spec.loader.exec_module(local_config) # type: ignore
    CILIA_FOLDER = local_config.CILIA_FOLDER

from config_sbi import fixed_params, prior_params, T_S2, REL_MOTOR_EXTRACTION, fixed_params_d_tilde

# -----------------------------------------------------------------------------
# IMPORT COMMON
# -----------------------------------------------------------------------------
from sbi_common import (
    SimulationMode,
    SpdeParams,
    load_c_api,
    sample_prior, # type: ignore
    build_spde_params,
    run_simulation, # type: ignore
    SimulationDataset,
)


lib = load_c_api(PROJECT_ROOT)


# -----------------------------
# DATASET IMPORTS
# -----------------------------
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization
from cilia.datasets.DataItem import DataItem
from cilia.datastructures.DataName import (
    ARCLENGTH,
    DT_FRAME,
    TANGENT_ANGLE_SERIES,
    CONDITION_DESCRIPTION,
)
from cilia.datastructures.special_source_names import ORIGINAL

from ml_features import (
    train_svm_from_dataframe,
    svm_accept_sample,
    train_gp_from_dataframe,
    gp_ucb_score,train_svm_from_s2
)

# -----------------------------
# STRATEGIES
# -----------------------------
def sample_svm_strategy(rng, svm, tau=0.1):
    while True:
        s = sample_prior(rng, prior_params, fixed_params)
        if svm_accept_sample(svm, s, tau=tau):
            return s



def transform_logL1(df):
    df = df.copy()

    # valid only where logL1 is finite and negative
    mask = np.isfinite(df["logL1"]) & (df["logL1"] < 0)

    df = df[mask].copy()

    df["target_gp"] = -np.log10(-df["logL1"])

    return df

def train_gp_from_s2(df_s2):
    df = transform_logL1(df_s2)

    # only oscillatory_full
    df = df[
        (df["oscillatory_full"] == 1) &
        (df["oscillatory_reduced"] == 1)
    ]

    if len(df) < 20:
        print("WARNING: too little S2 data for GP")
        return lambda x: (0.0, 1.0)

    return train_gp_from_dataframe(df, target_col="target_gp")


def sample_svm_gp_s2_strategy(
    rng, svm, gp, tau=0.1, n_candidates=100, alpha=1.0
):
    candidates = []
    tries = 0

    while len(candidates) < n_candidates and tries < 10 * n_candidates:
        s = sample_prior(rng, prior_params, fixed_params)
        tries += 1

        if svm_accept_sample(svm, s, tau=tau):
            candidates.append(s)

    if len(candidates) == 0:
        return sample_prior(rng, prior_params, fixed_params)  # fallback

    scores = [gp_ucb_score(gp, s, alpha) for s in candidates]
    return candidates[int(np.argmax(scores))]

# -----------------------------
# DATASET
# -----------------------------
def save_realization(dataset, sample, gamma_red, gamma_full, strategy, seed):
    exp_id = hashlib.sha256(
        json.dumps(dict(sample=sample, seed=seed, strategy=strategy), sort_keys=True).encode()
    ).hexdigest()[:16]

    realization = Realization(exp_id)

    condition = {
        **sample,
        "strategy": strategy,
    }

    # --- reduced ---
    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=gamma_red,
            dependencies=[ORIGINAL.key],
            algorithm="reduced",
        ),
    )

    # --- full ---
    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=gamma_full,
            dependencies=[ORIGINAL.key],
            algorithm="full",
        ),
    )

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=CONDITION_DESCRIPTION,
            data=condition,
            dependencies=[ORIGINAL.key],
            algorithm="meta",
        ),
    )

    s = np.linspace(0.0, 1.0, fixed_params['n']+1)
    dt_frame = 1.0 / fixed_params["t_sub"]
    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=ARCLENGTH,
            data=s,
            dependencies=[ORIGINAL.key],
            algorithm="simulation_metadata",
        ),
    )

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=DT_FRAME,
            data=float(dt_frame),
            dependencies=[ORIGINAL.key],
            algorithm="simulation_metadata",
        ),
    )

    dataset.realizations.append(realization)

def run_simulation(params: SpdeParams):
    n_coarse = int(params.T * params.t_sub) + 1
    N = params.n + 1

    t = np.zeros(n_coarse, dtype=np.float64)
    gamma = np.zeros((n_coarse, N), dtype=np.float64)

    nplus_flat = np.zeros(n_coarse * 4 * N, dtype=np.float64)
    nminus_flat = np.zeros(n_coarse * 4 * N, dtype=np.float64)

    ret = lib.spde_simulate(
        ctypes.byref(params),
        t,
        gamma,
        nplus_flat,
        nminus_flat,
    )

    if ret < 0:
        return None, None, None, None

    nplus = nplus_flat.reshape(n_coarse, 4, N)
    nminus = nminus_flat.reshape(n_coarse, 4, N)

    return t, gamma, nplus, nminus

# -----------------------------
# MAIN LOOP
# -----------------------------

def build_dataset(dataset_root, n_realizations, strategy, seed, dimension):
    rng = np.random.default_rng(seed)
    dataset = SimulationDataset(dataset_root)

    # --- load S1 classifier ---
    df_s1 = pd.read_csv(f"scalar_observables_sbi_s1_{dimension}.csv")
    svm_s1 = train_svm_from_dataframe(df_s1, class_weight={0: 1, 1: 5})

    created = 0

    while created < n_realizations:
        seed_i = int(rng.integers(0, 2**31))

        if strategy == "prior":
            sample = sample_prior(rng, prior_params, fixed_params)

        elif strategy == "svm_guided_S1":
            sample = sample_svm_strategy(rng, svm_s1)

        elif strategy == "svm_and_gp":
            # --- load S2 models ---
            svm_s2 = None
            gp = None

            if os.path.exists(f"scalar_observables_sbi_s2_{dimension}.csv"):
                df_s2 = pd.read_csv(f"scalar_observables_sbi_s2_{dimension}.csv")

                svm_s2, tau_s2 = train_svm_from_s2(df_s2)
                gp = train_gp_from_s2(df_s2)
               
            else:
                print("WARNING: no S2 data → fallback")
                raise
            sample = sample_svm_gp_s2_strategy(rng, svm_s2, gp, tau=tau_s2)

        else:
            raise ValueError(strategy)

        # --- simulate both systems ---
        # --- build params (reduced) ---
        params_red = build_spde_params(sample, seed_i, fixed_params, fixed_params_d_tilde, dimension, T_S2, fixed_params["Nmotor"] * REL_MOTOR_EXTRACTION, sample["mu_a"] * REL_MOTOR_EXTRACTION, "Poisson")
        _, gamma_red, _, _ = run_simulation(params_red)
        params_full = build_spde_params(sample, seed_i, fixed_params, fixed_params_d_tilde, dimension, T_S2, fixed_params["Nmotor"], sample["mu_a"], "Poisson")
        _, gamma_full, _, _ = run_simulation(params_full)


        save_realization(dataset, sample, gamma_red, gamma_full, strategy, seed_i)

        created += 1
        print(f"{created}/{n_realizations} done ({strategy})")

    print(f"\n✅ S2 dataset built: {dataset_root}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build S2 dataset (reduced + full simulations)."
    )

    parser.add_argument(
        "--dimension",
        choices=["2d", "3d"],
        default="3d",
        help="Simulation dimension (2d or 3d)",
    )

    parser.add_argument(
        "--dataset-root",
        help="Target dataset directory",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--strategy",
        choices=["prior", "svm_guided_S1", "svm_and_gp"],
        default="svm_and_gp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.join(CILIA_FOLDER, f"structured/SBI/SBI_S2_{args.dimension}")

    return args


def main():
    args = parse_args()

    os.makedirs(args.dataset_root, exist_ok=True)

    build_dataset(
        dataset_root=args.dataset_root,
        n_realizations=args.n_realizations,
        strategy=args.strategy,
        seed=args.seed,
        dimension=args.dimension,
    )


if __name__ == "__main__":
    main()
