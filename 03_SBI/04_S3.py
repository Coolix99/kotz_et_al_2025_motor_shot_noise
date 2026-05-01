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
    import importlib.util
    spec = importlib.util.spec_from_file_location("local_config", str(cfg_path))
    local_config = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(local_config)  # type: ignore
    CILIA_FOLDER = local_config.CILIA_FOLDER

from config_sbi import N_MOTOR_2D, N_Motor_3D, fixed_params, prior_params, T_S3, REL_MOTOR_EXTRACTION, fixed_params_d_tilde

# -----------------------------------------------------------------------------
# IMPORT COMMON
# -----------------------------------------------------------------------------
from sbi_common import (
    SimulationMode,
    SpdeParams,
    load_c_api,
    sample_prior,
    build_spde_params,
    run_simulation, # type: ignore
    SimulationDataset,
)

lib = load_c_api(PROJECT_ROOT)


# -----------------------------------------------------------------------------
# DATASET IMPORTS (same as S2)
# -----------------------------------------------------------------------------
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
    gp_ucb_score,
    train_svm_from_s2,
)

def sample_svm_strategy(rng, svm, tau=0.1):
    while True:
        s = sample_prior(rng, prior_params, fixed_params)
        if svm_accept_sample(svm, s, tau=tau):
            return s


def train_gp_from_s23(df_s2):
    df = df_s2[
        (df_s2["oscillatory_full"] == 1) &
        (df_s2["oscillatory_reduced"] == 1)
    ].copy()

    required_cols = [
        "logL1",
        "mu", "eta", "zeta", "mu_a", "fstar", "beta",
    ]

    for col in required_cols:
        df = df[np.isfinite(df[col])]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #print(f"GP training data: {len(df)} samples after filtering")

    if len(df) < 20:
        print("WARNING: too little data for GP")
        return lambda x: (0.0, 1.0)

    return train_gp_from_dataframe(df, target_col="logL1")


def sample_svm_gp_strategy(
    rng,
    svm,
    gp,
    tau=0.1,
    n_candidates=100,
    alpha=1.0,
):
    candidates = []
    tries = 0

    while len(candidates) < n_candidates and tries < 10 * n_candidates:
        s = sample_prior(rng, prior_params, fixed_params)
        tries += 1

        if svm_accept_sample(svm, s, tau=tau):
            candidates.append(s)

    if len(candidates) == 0:
        return sample_prior(rng, prior_params, fixed_params)

    scores = [gp_ucb_score(gp, s, alpha) for s in candidates]
    return candidates[int(np.argmax(scores))]

def sample_best_from_s2(rng, df_s2, top_k=50):
    df = df_s2.copy()

    # valid logL1
    df = df[np.isfinite(df["logL1"])]

    if len(df) == 0:
        raise RuntimeError("No valid S2 entries")

    # sort: best first (logL1 closer to 0 is better)
    df = df.sort_values("logL1", ascending=False)

    # take top-k
    df_top = df.head(top_k)

    # random pick (important: avoids duplicates)
    row = df_top.sample(n=1, random_state=rng.integers(0, 2**31)).iloc[0]

    return dict(
        mu=float(row["mu"]),
        eta=float(row["eta"]),
        zeta=float(row["zeta"]),
        mu_a=float(row["mu_a"]),
        fstar=float(row["fstar"]),
        beta=float(row["beta"]),
    )

def jitter_sample(sample, rng, scale=0.03):
    return {
        k: float(v * np.exp(scale * rng.normal()))
        for k, v in sample.items()
    }


# SAVE REALIZATION 
def save_realization(dataset, sample, gamma_red, gamma_full, strategy, seed):
    exp_id = hashlib.sha256(
        json.dumps(dict(sample=sample, seed=seed, strategy=strategy), sort_keys=True).encode()
    ).hexdigest()[:16]

    realization = Realization(exp_id)

    condition = {
        **sample,
        "Nmotor": sample["Nmotor"],
        "strategy": strategy,
        "stage": "S3",
    }

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=gamma_red,
            dependencies=[ORIGINAL.key],
            algorithm="reduced",
        ),
    )

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

    Ns = fixed_params['n']+1
    s = np.linspace(0.0, 1.0, Ns)

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
            data=float(1.0 / fixed_params["t_sub"]),
            dependencies=[ORIGINAL.key],
            algorithm="simulation_metadata",
        ),
    )

    dataset.realizations.append(realization)
# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------
def run_simulation(params):
    n_coarse = int(params.T * params.t_sub) + 1
    N = params.n + 1
    t = np.zeros(n_coarse)
    gamma = np.zeros((n_coarse, N))

    nplus = np.zeros(n_coarse * 4 * N)
    nminus = np.zeros(n_coarse * 4 * N)

    ret = lib.spde_simulate(
        ctypes.byref(params),
        t,
        gamma,
        nplus,
        nminus,
    )

    if ret < 0:
        return None,None, None, None

    return gamma


# -----------------------------------------------------------------------------
# PARAM BUILDING
# -----------------------------------------------------------------------------


def sample_best_from_s3(rng, df_s3, top_k=50):
    df = df_s3.copy()

    # valid posterior
    df = df[np.isfinite(df["logL1"])]

    if len(df) == 0:
        raise RuntimeError("No valid S3 entries")

    # sort: best first
    df = df.sort_values("logL1", ascending=False)

    # take top-k
    df_top = df.head(top_k)

    # random pick (avoid duplicates)
    row = df_top.sample(n=1, random_state=rng.integers(0, 2**31)).iloc[0]

    return dict(
        mu=float(row["mu"]),
        eta=float(row["eta"]),
        zeta=float(row["zeta"]),
        mu_a=float(row["mu_a"]),
        fstar=float(row["fstar"]),
        beta=float(row["beta"]),
    )

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def build_dataset(dataset_root, n_realizations, strategy, seed, dimension):
    rng = np.random.default_rng(seed)
    dataset = SimulationDataset(dataset_root)

    svm_s2 = None
    gp_s2 = None
    tau_s2 = None
    df_s2_cached = None

    # load S2 guidance once, outside the loop
    if strategy in ["svm_and_gp_s2", "best_from_s2"]:
        s2_csv = PROJECT_ROOT / f"scalar_observables_sbi_s2_{dimension}.csv"
        if not s2_csv.exists():
            raise FileNotFoundError(f"S2 CSV not found: {s2_csv}")

        df_s2_cached = pd.read_csv(s2_csv)

        if strategy == "svm_and_gp_s2":
            svm_s2, tau_s2 = train_svm_from_s2(df_s2_cached)
            gp_s2 = train_gp_from_s23(df_s2_cached)
    created = 0

    while created < n_realizations:
        seed_i = int(rng.integers(0, 2**31))

        if strategy == "prior":
            sample = sample_prior(rng, prior_params, fixed_params)

        elif strategy == "svm_and_gp_s2":
            assert svm_s2 is not None
            assert gp_s2 is not None
            assert tau_s2 is not None

            sample = sample_svm_gp_strategy(
                rng,
                svm_s2,
                gp_s2,
                tau=tau_s2,
                n_candidates=100,
                alpha=1.0,
            )

        elif strategy == "svm_and_gp_s3":
            s3_csv = PROJECT_ROOT / f"scalar_observables_sbi_s3_{dimension}.csv"

            if s3_csv.exists():
                df_s3 = pd.read_csv(s3_csv)

                if len(df_s3) >= 20:
                    svm_s3, tau_s3 = train_svm_from_s2(df_s3)
                    gp_s3 = train_gp_from_s23(df_s3)

                    sample = sample_svm_gp_strategy(
                        rng,
                        svm_s3,
                        gp_s3,
                        tau=tau_s3,
                        n_candidates=100,
                        alpha=1.0,
                    )
                else:
                    print("WARNING: too little S3 data → fallback to prior")
                    sample = sample_prior(rng, prior_params, fixed_params)
            else:
                print("WARNING: no S3 CSV yet → fallback to prior")
                sample = sample_prior(rng, prior_params, fixed_params)

        elif strategy == "best_from_s2":
            assert df_s2_cached is not None

            sample = sample_best_from_s2(
                rng,
                df_s2_cached,
                top_k=50,
            )
            sample = jitter_sample(sample, rng)

        elif strategy == "best_from_s3":
            s3_csv = PROJECT_ROOT / f"scalar_observables_sbi_s3_{dimension}.csv"
            df_s3 = pd.read_csv(s3_csv)
            sample = sample_best_from_s3(
                rng,
                df_s3,
                top_k=50,
            )
            sample = jitter_sample(sample, rng)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if dimension == "2d":
            sample["Nmotor"] = N_MOTOR_2D
        else:
            sample["Nmotor"] = N_Motor_3D

        gamma_red = run_simulation(build_spde_params(sample, seed_i, fixed_params, fixed_params_d_tilde, dimension, T_S3, sample["Nmotor"] * REL_MOTOR_EXTRACTION, sample["mu_a"] * REL_MOTOR_EXTRACTION, "Poisson"))
        gamma_full = run_simulation(build_spde_params(sample, seed_i, fixed_params, fixed_params_d_tilde, dimension, T_S3, sample["Nmotor"], sample["mu_a"], "Poisson"))

        save_realization(dataset, sample, gamma_red, gamma_full, strategy, seed_i)

        created += 1
        print(f"{created}/{n_realizations} done ({strategy})")

    print(f"\n✅ S3 dataset built: {dataset_root}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Build S3 dataset")

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
    parser.add_argument("--n-realizations", type=int, default=250)
    parser.add_argument(
        "--strategy",
        choices=["prior", "svm_and_gp_s2", "svm_and_gp_s3", "best_from_s2", "best_from_s3"],
        default="best_from_s3",
    )
    parser.add_argument("--seed", type=int, default=72)

    args = parser.parse_args()

    if args.dataset_root is None:
        suffix = "_3d" if args.dimension == "3d" else "_2d"
        args.dataset_root = os.path.join(CILIA_FOLDER, f"structured/SBI/SBI_S3{suffix}")

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