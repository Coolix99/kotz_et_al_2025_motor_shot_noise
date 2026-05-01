import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

import sys
from pathlib import Path
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


import pandas as pd


try:
    from cilia.datasets.DataSet import DataSet
    from cilia.datasets.Realization import Realization
    from cilia.datasets.DataItem import DataItem
    from cilia.datastructures.DataName import (
        TANGENT_ANGLE_SERIES,
        CONDITION_DESCRIPTION,
        DT_FRAME,
        ARCLENGTH,
    )
    from cilia.datastructures.special_source_names import ORIGINAL
except Exception as exc:
    raise ImportError(
        "Failed to import cilia dataset classes. Ensure the cilia repository is available "
        "and the PYTHONPATH includes the cilia folder or local_config.CILIA_FOLDER is set."
    ) from exc

from config_sbi import fixed_params, prior_params, fixed_params_d_tilde, T_S1, REL_MOTOR_EXTRACTION

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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_features_from_df(df):
    mu = df["mu"].values
    eta = df["eta"].values
    zeta = df["zeta"].values
    mu_a = df["mu_a"].values
    fstar = df["fstar"].values
    beta = df["beta"].values

    mu_a_times_zeta = mu_a * zeta

    X = np.column_stack([
        np.log(mu),
        eta,  # NOT log
        np.log(mu_a_times_zeta),
        np.log(fstar),
        np.log(beta),
    ])

    return X

def train_oscillation_classifier(
    df,
    test_size=0.2,
    random_state=42,
    max_fn_rate=0.01,
):
    X = build_features_from_df(df)
    y = df["oscillatory"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # -------------------------------------------------
    # models
    # -------------------------------------------------
    models = {
        "SVM_rbf_strong": make_pipeline(
            StandardScaler(),
            SVC(probability=True, class_weight={0: 1, 1: 5})
        ),
        "SVM_linear": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, class_weight={0: 1, 1: 3})
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight={0: 1, 1: 3},
            random_state=random_state,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000)
        ),
    }

    # -------------------------------------------------
    # helper: safe threshold
    # -------------------------------------------------
    def find_safe_threshold(y_true, y_prob, max_fn_rate):
        thresholds = np.linspace(0.0, 1.0, 500)
        best_tau = 0.0

        for tau in thresholds:
            rejected = y_prob < tau
            if np.sum(rejected) == 0:
                continue

            fn = np.sum((y_true == 1) & rejected)
            tn = np.sum((y_true == 0) & rejected)

            fn_rate_reject = fn / (fn + tn + 1e-12)

            if fn_rate_reject <= max_fn_rate:
                best_tau = tau

        return best_tau

    # -------------------------------------------------
    # evaluate all
    # -------------------------------------------------
    results = []

    for name, model in models.items():
        print(f"\n==============================")
        print(f"Model: {name}")
        print(f"==============================")

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        tau = find_safe_threshold(y_test, y_prob, max_fn_rate)

        y_pred = (y_prob > tau).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn + 1e-12)  # power
        fp_rate = fp / (fp + tn + 1e-12)

        print(f"tau_safe = {tau:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Power (recall osc): {recall:.4f}")
        print(f"FP rate:            {fp_rate:.4f}")

        results.append({
            "model": name,
            "tau": tau,
            "power": recall,
            "fp_rate": fp_rate,
        })

    # -------------------------------------------------
    # summary
    # -------------------------------------------------
    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['model']:20s} | tau={r['tau']:.3f} | power={r['power']:.3f} | fp={r['fp_rate']:.3f}"
        )

    valid = [r for r in results if r["power"] >= (1.0 - max_fn_rate)]

    if len(valid) == 0:
        print("WARNING: No model satisfies FN constraint, falling back to best power")
        best = max(results, key=lambda r: r["power"])
    else:
        best = min(valid, key=lambda r: r["fp_rate"])

    best_model = models[best["model"]]

    print(
        f"\nBest model: {best['model']} "
        f"(power={best['power']:.4f}, fp={best['fp_rate']:.4f})"
    )

    return best_model, best["tau"]

def build_features_from_sample(sample):
    mu = sample["mu"]
    eta = sample["eta"]
    zeta = sample["zeta"]
    mu_a = sample["mu_a"]
    fstar = sample["fstar"]
    beta = sample["beta"]

    mu_a_times_zeta = mu_a * zeta

    x = np.array([
        np.log(mu),
        eta,
        np.log(mu_a_times_zeta),
        np.log(fstar),
        np.log(beta),
    ]).reshape(1, -1)

    return x

def accept_sample(model, sample, threshold_reject):
    x = build_features_from_sample(sample)

    p_osc = model.predict_proba(x)[0, 1]
    return p_osc > threshold_reject

def run_simulation(params: SpdeParams):
    n_coarse = int(params.T * params.t_sub) + 1
    N = params.n + 1

    # --- allocate flat buffers (C expects contiguous memory) ---
    t = np.zeros(n_coarse, dtype=np.float64)
    gamma = np.zeros((n_coarse, N), dtype=np.float64)

    # IMPORTANT: allocate flattened 3D storage
    nplus_flat = np.zeros(n_coarse * 4 * N, dtype=np.float64)
    nminus_flat = np.zeros(n_coarse * 4 * N, dtype=np.float64)

    # --- call C ---
    ret = lib.spde_simulate(
        ctypes.byref(params),
        t,
        gamma,
        nplus_flat,
        nminus_flat,
    )

    if ret < 0:
        return None, None, None,None

    # --- reshape into (time, component, space) ---
    nplus = nplus_flat.reshape(n_coarse, 4, N)
    nminus = nminus_flat.reshape(n_coarse, 4, N)

    return t, gamma, nplus, nminus

def make_experiment_id(sample: dict, seed: int, strategy: str) -> str:
    fingerprint = {
        "sample": sample,
        "seed": int(seed),
        "strategy": strategy,
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return digest[:16]

def save_realization(
    dataset: SimulationDataset,
    realization: Realization,
    gamma: np.ndarray,
    params: SpdeParams,
    strategy: str,
    sample: dict,
) -> None:
    exp_path = Path(dataset.path) / realization.experiment_id
    exp_path.mkdir(parents=True, exist_ok=True)

    arclength = np.linspace(0.0, 1.0, params.n + 1, dtype=np.float64)

    condition = {
        "mu": float(params.mu),
        "eta": float(params.eta),
        "zeta": float(params.zeta),
        "beta": float(params.beta),
        "fstar": float(params.fstar),

        # original
        "mu_a": float(sample["mu_a"]),
        "Nmotor": float(fixed_params["Nmotor"]),

        # reduced
        "mu_a_reduced": float(params.mu_a),
        "Nmotor_reduced": float(params.Nmotor),

        "motor_extraction_factor": float(REL_MOTOR_EXTRACTION),

        "mode": int(params.mode),
        "dt": float(params.dt),
        "T": float(params.T),
        "stage": "S1",
        "strategy": strategy,
    }

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=TANGENT_ANGLE_SERIES,
            data=gamma,
            dependencies=[ORIGINAL.key],
            algorithm="spde_simulation",
        ),
    )

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=CONDITION_DESCRIPTION,
            data=condition,
            dependencies=[ORIGINAL.key],
            algorithm="spde_metadata",
        ),
    )
    if params.t_sub > 0:
        dt_frame = 1.0 / float(params.t_sub)
    else:
        dt_frame = float("nan")
    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=DT_FRAME,
            data=dt_frame,
            dependencies=[ORIGINAL.key],
            algorithm="spde_metadata",
        ),
    )

    dataset.add_data_to_realization(
        realization,
        DataItem(
            data_name=ARCLENGTH,
            data=arclength,
            dependencies=[ORIGINAL.key],
            algorithm="spde_metadata",
        ),
    )

def build_dataset(
    dataset_root: str,
    n_realizations: int,
    strategy: str,
    seed: int,
    dimension: str,
):
    os.makedirs(dataset_root, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = SimulationDataset(dataset_root)

    existing_ids = {r.experiment_id for r in dataset.realizations}

    created = 0
    while created < n_realizations:
        sample_seed = int(rng.integers(0, 2**31 - 1))

        if strategy == "prior":
            sample = sample_prior(rng, prior_params, fixed_params)
            params = build_spde_params(sample, sample_seed, fixed_params, fixed_params_d_tilde, dimension, T_S1, fixed_params["Nmotor"] * REL_MOTOR_EXTRACTION, sample["mu_a"] * REL_MOTOR_EXTRACTION, "Poisson")

            _, gamma, _, _ = run_simulation(params)

            
            # if plt is not None and created < 3:
            #     plt.figure(figsize=(6, 4))
            #     plt.imshow(gamma, aspect="auto", origin="lower", cmap="viridis")
            #     plt.colorbar(label="psi")
            #     plt.title(f"Debug kymograph #{created}")
            #     plt.tight_layout()
            #     plt.show()

        elif strategy == "svm_guided":
            df = pd.read_csv(f"scalar_observables_sbi_s1_{dimension}.csv")
            model, tau_safe= train_oscillation_classifier(df)
            while True:
                sample = sample_prior(rng, prior_params, fixed_params)

                if accept_sample(model, sample,threshold_reject=tau_safe):
                    break

            params = build_spde_params(sample, sample_seed, fixed_params, fixed_params_d_tilde, dimension, T_S1, fixed_params["Nmotor"] * REL_MOTOR_EXTRACTION, sample["mu_a"] * REL_MOTOR_EXTRACTION, "Poisson")
            _, gamma, _, _ = run_simulation(params)

        experiment_id = make_experiment_id(sample, sample_seed, strategy)

        if experiment_id in existing_ids:
            print(f"{experiment_id} already exists, skipping")
            continue

        realization = Realization(experiment_id)
        
        save_realization(dataset, realization, gamma, params, strategy, sample) # type: ignore

        dataset.realizations.append(realization)
        existing_ids.add(experiment_id)

        created += 1
        print(
            f"Saved realization {created}/{n_realizations}: "
            f"{experiment_id} (strategy={strategy})"
        )

    print(f"\n✅ Built {created} simulations in dataset: {dataset_root}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build S1 simulation dataset using C++ SPDE API and cilia dataset framework."
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
    parser.add_argument("--n-realizations", type=int, default=500)
    parser.add_argument(
        "--strategy",
        choices=["prior", "svm_guided"],
        default="svm_guided",
    )
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.join(CILIA_FOLDER, f"structured/SBI/SBI_S1_{args.dimension}")

    return args


def main():
    args = parse_args()
    build_dataset(
        dataset_root=args.dataset_root,
        n_realizations=args.n_realizations,
        strategy=args.strategy,
        seed=args.seed,
        dimension=args.dimension,
    )


if __name__ == "__main__":
    main()
