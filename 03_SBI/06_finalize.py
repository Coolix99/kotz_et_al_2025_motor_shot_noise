import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pipeline_utils import (
    compute_l1,
    compute_l1_s3,
    compute_l2,
    estimate_scaling,
    infer_biological_params,
    log_bio_prior,
)
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from scipy.spatial.distance import cdist

EPS = 1e-12
B_FIXED = 840.0

def prepare_l1_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["mu_a_times_zeta"] = out["mu_a"] * out["zeta"]
    out["mu_a_div_zeta_0"] = out["mu_a"] / (out["zeta"] + EPS)

    out["log10_mu"] = np.log10(out["mu"].astype(float) + EPS)
    out["log10_mu_a_times_zeta"] = np.log10(out["mu_a_times_zeta"].astype(float) + EPS)
    out["log10_fstar"] = np.log10(out["fstar"].astype(float) + EPS)
    out["log10_beta"] = np.log10(out["beta"].astype(float) + EPS)

    if "amplitude_full" in out.columns and "amplitude_reduced" in out.columns:
        out["sim_A"] = np.sqrt(out["amplitude_full"] * out["amplitude_reduced"])
    else:
        out["sim_A"] = np.nan

    if "f_full" in out.columns and "f_reduced" in out.columns:
        out["sim_f"] = np.sqrt(out["f_full"] * out["f_reduced"])
    else:
        out["sim_f"] = np.nan

    return out

def get_reduced_feature_cols() -> List[str]:
    return [
        "log10_mu",
        "eta",
        "log10_mu_a_times_zeta",
        "log10_fstar",
        "log10_beta",
    ]

def compare_regression_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "logL1",
    n_splits: int = 5,
    random_state: int = 0,
    make_plots: bool = True,
) -> pd.DataFrame:
    work = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    X = work[feature_cols].values.astype(float)
    y = work[target_col].values.astype(float)

    dim = X.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "GP_Matern_ARD_nu25": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(dim),
                    length_scale_bounds=(1e-2, 1e5),
                    nu=2.5,
                ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)),
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=random_state,
            ))
        ]),
        "GP_RBF_ARD": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(
                    length_scale=np.ones(dim),
                    length_scale_bounds=(1e-2, 1e5),
                ) + WhiteKernel(),
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=random_state,
            ))
        ]),
        "GP_RQ": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RationalQuadratic(
                    length_scale=1.0,
                    alpha=1.0,
                ) + WhiteKernel(),
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=random_state,
            ))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=6,
            learning_rate=0.03,
            max_iter=300,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=12, weights="distance"))
        ]),
        "SVR_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))
        ]),
    }

    results = []

    print("=" * 80)
    print("Comparing regression models on reduced logL1 feature space")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n--- Testing: {name} ---")

        cv_rmse = []
        cv_mae = []
        cv_r2 = []

        for train_idx, test_idx in kf.split(X):
            mdl = clone(model)
            mdl.fit(X[train_idx], y[train_idx])
            y_pred = mdl.predict(X[test_idx])

            cv_rmse.append(np.sqrt(mean_squared_error(y[test_idx], y_pred))) # type: ignore
            cv_mae.append(mean_absolute_error(y[test_idx], y_pred)) # type: ignore
            cv_r2.append(r2_score(y[test_idx], y_pred)) # type: ignore

        model.fit(X, y)
        y_fit = model.predict(X)

        train_rmse = np.sqrt(mean_squared_error(y, y_fit)) # type: ignore
        train_mae = mean_absolute_error(y, y_fit) # type: ignore
        train_r2 = r2_score(y, y_fit) # type: ignore

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train MAE : {train_mae:.4f}")
        print(f"Train R²  : {train_r2:.4f}")
        print(f"CV RMSE   : {np.mean(cv_rmse):.4f}")
        print(f"CV MAE    : {np.mean(cv_mae):.4f}")
        print(f"CV R²     : {np.mean(cv_r2):.4f}")

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner = model.named_steps["model"]
            if hasattr(inner, "kernel_"):
                print(f"Kernel    : {inner.kernel_}")

        results.append({
            "name": name,
            "model": model,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "cv_rmse": float(np.mean(cv_rmse)),
            "cv_mae": float(np.mean(cv_mae)),
            "cv_r2": float(np.mean(cv_r2)),
        })

        if make_plots:
            plt.figure(figsize=(5, 5))
            plt.scatter(y, y_fit, alpha=0.5) # type: ignore
            lo = min(y.min(), y_fit.min()) # type: ignore
            hi = max(y.max(), y_fit.max()) # type: ignore
            plt.plot([lo, hi], [lo, hi], "--")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(name)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(5, 4))
            plt.hist(y - y_fit, bins=30)
            plt.title(f"Residuals: {name}")
            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results).sort_values(["cv_rmse", "cv_mae"], ascending=True)

    print("\n" + "=" * 80)
    print("Model ranking")
    print(results_df[["name", "train_rmse", "train_mae", "train_r2", "cv_rmse", "cv_mae", "cv_r2"]])
    print("=" * 80)

    return results_df

def load_df(stage, dim):
    csv_path = f"scalar_observables_sbi_{stage.lower()}_{dim}.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Using CSV: {csv_path}")
    return pd.read_csv(csv_path)

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

def train_gp_on_logL1(df, param_cols):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X_raw = df[param_cols].values
    y = df["logL1"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=np.ones(len(param_cols)), nu=2.5)
        + WhiteKernel()
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )

    gp.fit(X, y)

    return gp, scaler

def sample_from_gp_logL1(
    gp,
    scaler,
    df,
    param_cols,
    n_samples=5000,
    proposal_scale=0.2,
    burnin=1000,
):
    rng = np.random.default_rng(0)

    X_raw = df[param_cols].values
    X = scaler.transform(X_raw)

    # start at best point
    best_idx = np.argmax(df["logL1"].values)
    x_curr = X[best_idx]

    def log_target(x):
        mu = gp.predict(x[None, :])[0]
        return mu

    samples = []
    logp_curr = log_target(x_curr)

    for i in range(n_samples + burnin):
        x_prop = x_curr + rng.normal(0, proposal_scale, size=len(param_cols))
        logp_prop = log_target(x_prop)

        if np.log(rng.uniform()) < (logp_prop - logp_curr):
            x_curr = x_prop
            logp_curr = logp_prop

        if i >= burnin:
            samples.append(x_curr.copy())

    samples = np.array(samples)
    samples_raw = scaler.inverse_transform(samples)

    df_samples = pd.DataFrame(samples_raw, columns=param_cols)
    df_samples["gp_logL1"] = gp.predict(samples)

    return df_samples

def enrich_samples_with_biology(df_samples, dimension):
    records = []

    for _, row in df_samples.iterrows():
        try:
            theta_scaled = {
                "mu": row["mu"],
                "eta": row["eta"],
                "mu_a_times_zeta": row["mu_a"] * row["zeta"],
                "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
                "zeta": row["zeta"],
                "fstar": row["fstar"],
                "beta": row["beta"],
            }

            # approximate sim features → use typical values
            sim_A = row.get("sim_A", 1.0)
            sim_f = row.get("sim_f", 1.0)

            tau, mu_a_div_zeta = estimate_scaling(theta_scaled, sim_A, sim_f)

            mu_a_new = np.sqrt(theta_scaled["mu_a_times_zeta"] * mu_a_div_zeta)
            zeta_new = mu_a_new / mu_a_div_zeta

            params = infer_biological_params(
                theta_scaled,
                B_FIXED,
                mu_a_div_zeta,
                tau, dimension
            )

            rec = row.to_dict()
            rec.update({
                "tau": tau,
                "mu_a_div_zeta": mu_a_div_zeta,
                "mu_a_new": mu_a_new,
                "zeta_new": zeta_new,
            })
            rec.update(params)

        except Exception:
            rec = row.to_dict()

        records.append(rec)

    return pd.DataFrame(records)

def export_samples(df_samples, filename="gp_logL1_samples.csv"):
    df_samples.to_csv(filename, index=False)
    print(f"\nSaved samples → {filename}")

from pipeline_utils import compute_l1, compute_l2, infer_biological_params, log_bio_prior, optimize_B

def analyze_experiment(row, dimension):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {row['experiment_id']}")
    print("=" * 70)

    # --------------------------------------------------------
    # RAW PARAMETERS
    # --------------------------------------------------------
    print("\n--- Raw parameters ---")
    for k in ["mu", "eta", "zeta", "mu_a", "fstar", "beta"]:
        print(f"{k:20s}: {row[k]:.4e}")

    # --------------------------------------------------------
    # SIM FEATURES
    # --------------------------------------------------------
    sim_A = np.sqrt(row["amplitude_full"] * row["amplitude_reduced"])
    sim_f = np.sqrt(row["f_full"] * row["f_reduced"])

    print("\n--- Simulation features ---")
    print(f"{'sim_A':20s}: {sim_A:.4e}")
    print(f"{'sim_f':20s}: {sim_f:.4e}")

    # --------------------------------------------------------
    # THETA SCALED
    # --------------------------------------------------------
    theta_scaled = {
        "mu": row["mu"],
        "eta": row["eta"],
        "mu_a_times_zeta": row["mu_a"] * row["zeta"],
        "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
        "zeta": row["zeta"],
        "fstar": row["fstar"],
        "beta": row["beta"],
    }

    # --------------------------------------------------------
    # LIKELIHOOD L1
    # --------------------------------------------------------
    if ("logQ_full" in row.index) and ("logQ_reduced" in row.index):
        l1 = compute_l1_s3(
            row.get("amplitude_ratio", np.nan),
            row.get("frequency_ratio", np.nan),
            row.get("lambda_full", np.nan),
            row.get("lambda_reduced", np.nan),
            row.get("logQ_full", np.nan),
            row.get("logQ_reduced", np.nan),
        )
        l1_mode = "S3"
    else:
        l1 = compute_l1(
            row.get("amplitude_ratio", np.nan),
            row.get("frequency_ratio", np.nan),
            row.get("lambda_full", np.nan),
            row.get("lambda_reduced", np.nan),
        )
        l1_mode = "S2"

    print(f"\nUsing {l1_mode} likelihood")
    print(f"{'logL1':20s}: {l1:.4f}")

    # ========================================================
    # VERSION 1: FIXED B + DIRECT SCALING
    # ========================================================
    print("\n" + "-" * 70)
    print("VERSION 1: FIXED B + estimate_scaling")
    print("-" * 70)

    tau_v1, mu_a_div_zeta_v1 = estimate_scaling(theta_scaled, sim_A, sim_f)

    print("\n--- Scaling (direct) ---")
    print(f"{'tau':20s}: {tau_v1:.4e}")
    print(f"{'mu_a/zeta':20s}: {mu_a_div_zeta_v1:.4e}")

    mu_a_new_v1 = np.sqrt(theta_scaled["mu_a_times_zeta"] * mu_a_div_zeta_v1)
    zeta_new_v1 = mu_a_new_v1 / mu_a_div_zeta_v1

    print("\n--- Rescaled parameters ---")
    print(f"{'mu_a_new':20s}: {mu_a_new_v1:.4e}")
    print(f"{'zeta_new':20s}: {zeta_new_v1:.4e}")

    params_v1 = infer_biological_params(
        theta_scaled,
        B_FIXED,
        mu_a_div_zeta_v1,
        tau_v1, dimension
    )

    print(f"\n--- Biological parameters (B={B_FIXED:g}) ---")
    for k, v in params_v1.items():
        print(f"{k:20s}: {v:10.4e}")

    l2_v1 = compute_l2(
        sim_A,
        sim_f,
        tau_v1,
        mu_a_div_zeta_v1,
        theta_scaled["mu_a_div_zeta_0"],
    )
    lp_v1 = log_bio_prior(params_v1)
    total_v1 = l1 + l2_v1 + lp_v1

    print("\n--- Contributions ---")
    print(f"{'logL1':20s}: {l1:.4f}")
    print(f"{'logL2':20s}: {l2_v1:.4f}")
    print(f"{'log_prior':20s}: {lp_v1:.4f}")
    print(f"{'log_posterior':20s}: {total_v1:.4f}")

    # ========================================================
    # VERSION 2: OPTIMIZED B, tau, mu_a/zeta
    # ========================================================
    print("\n" + "-" * 70)
    print("VERSION 2: OPTIMIZED B + tau + mu_a/zeta")
    print("-" * 70)

    B_v2, tau_v2, mu_a_div_zeta_v2 = optimize_B(
        theta_scaled,
        sim_A,
        sim_f, dimension
    )

    print("\n--- Optimized scaling ---")
    print(f"{'B_opt':20s}: {B_v2:.4e}")
    print(f"{'tau_opt':20s}: {tau_v2:.4e}")
    print(f"{'mu_a/zeta_opt':20s}: {mu_a_div_zeta_v2:.4e}")

    mu_a_new_v2 = np.sqrt(theta_scaled["mu_a_times_zeta"] * mu_a_div_zeta_v2)
    zeta_new_v2 = mu_a_new_v2 / mu_a_div_zeta_v2

    print("\n--- Rescaled parameters ---")
    print(f"{'mu_a_new':20s}: {mu_a_new_v2:.4e}")
    print(f"{'zeta_new':20s}: {zeta_new_v2:.4e}")

    params_v2 = infer_biological_params(
        theta_scaled,
        B_v2,
        mu_a_div_zeta_v2,
        tau_v2, dimension
    )

    print("\n--- Biological parameters (optimized) ---")
    for k, v in params_v2.items():
        print(f"{k:20s}: {v:10.4e}")

    l2_v2 = compute_l2(
        sim_A,
        sim_f,
        tau_v2,
        mu_a_div_zeta_v2,
        theta_scaled["mu_a_div_zeta_0"],
    )
    lp_v2 = log_bio_prior(params_v2)
    total_v2 = l1 + l2_v2 + lp_v2

    print("\n--- Contributions ---")
    print(f"{'logL1':20s}: {l1:.4f}")
    print(f"{'logL2':20s}: {l2_v2:.4f}")
    print(f"{'log_prior':20s}: {lp_v2:.4f}")
    print(f"{'log_posterior':20s}: {total_v2:.4f}")

    # --------------------------------------------------------
    # CSV COMPARISON
    # --------------------------------------------------------
    print("\n--- CSV comparison ---")
    if "logL1" in row.index:
        print(f"{'logL1 (csv)':20s}: {row['logL1']:.4f}")
    if "logL2" in row.index:
        print(f"{'logL2 (csv)':20s}: {row['logL2']:.4f}")
    if "log_prior" in row.index:
        print(f"{'log_prior (csv)':20s}: {row['log_prior']:.4f}")
    if "log_posterior" in row.index:
        print(f"{'log_posterior (csv)':20s}: {row['log_posterior']:.4f}")
    if "B_opt" in row.index:
        print(f"{'B_opt (csv)':20s}: {row['B_opt']:.4e}")
    if "tau_opt" in row.index:
        print(f"{'tau_opt (csv)':20s}: {row['tau_opt']:.4e}")
    if "mu_a_div_zeta_opt" in row.index:
        print(f"{'mu_a/zeta_opt (csv)':20s}: {row['mu_a_div_zeta_opt']:.4e}")

    # --------------------------------------------------------
    # DELTAS
    # --------------------------------------------------------
    print("\n--- Improvement from optimization ---")
    print(f"{'Δ logL2':20s}: {l2_v2 - l2_v1:+.4f}")
    print(f"{'Δ log_prior':20s}: {lp_v2 - lp_v1:+.4f}")
    print(f"{'Δ log_posterior':20s}: {total_v2 - total_v1:+.4f}")

def compare_gp_models(
    df,
    param_cols=None,
    target_col="log_posterior",
    n_splits=5,
    random_state=0,
    verbose=True,
    make_plots=True,
):
    if param_cols is None:
        param_cols = ["B", "F0", "Fc", "v0", "pi0", "eps0", "K_d2"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X_raw = df[param_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    dim = X.shape[1]

    # ---------------------------------------
    # Define candidate kernels
    # ---------------------------------------
    kernels = {
        "Matern_ARD_nu25": ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e5),
            nu=2.5,
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)),

        "Matern_ARD_nu15": ConstantKernel(1.0) * Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e5),
            nu=1.5,
        ) + WhiteKernel(),

        "RBF_ARD": ConstantKernel(1.0) * RBF(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e5),
        ) + WhiteKernel(),

        "RationalQuadratic": ConstantKernel(1.0) * RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
        ) + WhiteKernel(),

        "Matern_isotropic": ConstantKernel(1.0) * Matern(
            length_scale=1.0,
            nu=2.5,
        ) + WhiteKernel(),

        "RBF_isotropic": ConstantKernel(1.0) * RBF(
            length_scale=1.0,
        ) + WhiteKernel(),

        "RQ_no_noise": ConstantKernel(1.0) * RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
        ),
    }

    results = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print("=" * 80)
    print("Comparing GP models")
    print("=" * 80)

    for name, kernel in kernels.items():
        print(f"\n--- Testing: {name} ---")

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

        # ---- cross-validation ----
        cv_rmse = []
        for train_idx, test_idx in kf.split(X):
            gp.fit(X[train_idx], y[train_idx])
            y_pred = gp.predict(X[test_idx])
            rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
            cv_rmse.append(rmse)

        cv_rmse = np.mean(cv_rmse)

        # ---- fit on full data ----
        gp.fit(X, y)
        y_pred, _ = gp.predict(X, return_std=True) # type: ignore

        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        lml = gp.log_marginal_likelihood()

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"CV RMSE   : {cv_rmse:.4f}")
        print(f"R^2       : {r2:.4f}")
        print(f"LML       : {lml:.2f}")
        print(f"Kernel    : {gp.kernel_}")

        # ---- extract length scales if possible ----
        try:
            matern_part = gp.kernel_.k1.k2 # type: ignore
            if hasattr(matern_part, "length_scale"):
                ls = np.atleast_1d(matern_part.length_scale)
                print("Length scales (std space):", np.round(ls, 3))
        except:
            pass

        results.append({
            "name": name,
            "gp": gp,
            "train_rmse": train_rmse,
            "cv_rmse": cv_rmse,
            "r2": r2,
            "lml": lml,
        })

        # ---- plots ----
        if make_plots:
            plt.figure(figsize=(5, 5))
            plt.scatter(y, y_pred)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "--")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(name)
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.hist(y - y_pred, bins=30)
            plt.title(f"Residuals: {name}")
            plt.show()

    # ---------------------------------------
    # ranking
    # ---------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cv_rmse")

    print("\n" + "=" * 80)
    print("Model ranking (by CV RMSE):")
    print(results_df[["name", "train_rmse", "cv_rmse", "r2"]])
    print("=" * 80)

    return results_df


def _plot_gp_diagnostics(
    work: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_predictions: pd.DataFrame,
    samples_feature_space: pd.DataFrame,
    chain_logp: np.ndarray,
    nn_train: np.ndarray,
):
    import math

    plt.figure(figsize=(6, 6))
    plt.scatter(
        train_predictions[target_col],
        train_predictions["gp_pred_mean"],
        alpha=0.7,
    )
    lo = min(train_predictions[target_col].min(), train_predictions["gp_pred_mean"].min())
    hi = max(train_predictions[target_col].max(), train_predictions["gp_pred_mean"].max())
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel(f"True {target_col}")
    plt.ylabel("GP predicted mean")
    plt.title("Training fit")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(train_predictions["residual"], bins=30)
    plt.xlabel("Residual = true - GP mean")
    plt.ylabel("Count")
    plt.title("Training residuals")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(train_predictions["gp_pred_std"], bins=30)
    plt.xlabel("GP predictive std")
    plt.ylabel("Count")
    plt.title("Predictive uncertainty on training points")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(nn_train, bins=30)
    plt.xlabel("Nearest-neighbor distance (standardized space)")
    plt.ylabel("Count")
    plt.title("Training cloud density")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.plot(chain_logp, lw=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Surrogate log density")
    plt.title("MH trace")
    plt.tight_layout()
    plt.show()

    n = len(feature_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, c in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(work[c], bins=30, density=True, alpha=0.5, label="observed")
        ax.hist(samples_feature_space[c], bins=30, density=True, alpha=0.5, label="sampled")
        ax.set_title(c)
        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Observed vs sampled feature marginals")
    fig.tight_layout()
    plt.show()

    m = min(4, len(feature_cols))
    fig, axes = plt.subplots(m, m, figsize=(3 * m, 3 * m))

    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            if i == j:
                ax.hist(work[feature_cols[i]], bins=25, density=True, alpha=0.4, label="obs")
                ax.hist(samples_feature_space[feature_cols[i]], bins=25, density=True, alpha=0.4, label="samp")
            else:
                ax.scatter(
                    work[feature_cols[j]],
                    work[feature_cols[i]],
                    s=8,
                    alpha=0.2,
                    label="obs" if (i == 0 and j == 1) else None,
                )
                ax.scatter(
                    samples_feature_space[feature_cols[j]],
                    samples_feature_space[feature_cols[i]],
                    s=8,
                    alpha=0.2,
                    label="samp" if (i == 0 and j == 1) else None,
                )
            if i == m - 1:
                ax.set_xlabel(feature_cols[j])
            if j == 0:
                ax.set_ylabel(feature_cols[i])

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Corner view: observed vs sampled")
    fig.tight_layout()
    plt.show()

@dataclass
class GPLikelihoodExplorerResult:
    gp: GaussianProcessRegressor
    x_scaler: StandardScaler
    feature_cols: List[str]
    samples_feature_space: pd.DataFrame
    reconstructed_samples: pd.DataFrame
    train_predictions: pd.DataFrame
    diagnostics: Dict


def explore_likelihood_gp(
    df: pd.DataFrame,
    feature_cols: List[str],
    dimension: str,
    target_col: str = "logL1",
    n_mcmc: int = 20000,
    burnin: int = 3000,
    thin: int = 10,
    proposal_scale: float = 0.2,
    random_state: int = 0,
    distance_penalty_strength: float = 2.0,
    trusted_nn_quantile: float = 0.95,
    verbose: bool = True,
    make_plots: bool = True,
) -> GPLikelihoodExplorerResult:
    rng = np.random.default_rng(random_state)

    needed_cols = feature_cols + [
        target_col,
        "experiment_id",
        "mu_a_div_zeta_0",
        "sim_A",
        "sim_f",
    ]
    work = df[needed_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X_raw = work[feature_cols].values.astype(float)
    y = work[target_col].values.astype(float)

    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X_raw)

    if verbose:
        print("=" * 80)
        print("explore_likelihood_gp: starting")
        print(f"Rows used            : {len(work)}")
        print(f"Feature columns      : {feature_cols}")
        print(f"Target column        : {target_col}")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(len(feature_cols)),
            length_scale_bounds=(1e-2, 1e5),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=random_state,
    )

    gp.fit(X, y) # type: ignore

    y_pred, y_std = gp.predict(X, return_std=True) # type: ignore
    train_rmse = np.sqrt(mean_squared_error(y, y_pred)) # type: ignore
    train_mae = mean_absolute_error(y, y_pred) # type: ignore
    train_r2 = r2_score(y, y_pred) # type: ignore

    dmat = cdist(X, X)
    np.fill_diagonal(dmat, np.inf)
    nn_train = dmat.min(axis=1)
    trusted_radius = np.quantile(nn_train, trusted_nn_quantile)

    if verbose:
        print("-" * 80)
        print("GP diagnostics:")
        print(f"  RMSE           : {train_rmse:.6g}")
        print(f"  MAE            : {train_mae:.6g}")
        print(f"  R²             : {train_r2:.6g}")
        print(f"  kernel         : {gp.kernel_}")
        print(f"  trusted radius : {trusted_radius:.6g}")

    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)

    def inside_box(x_std: np.ndarray) -> bool:
        return np.all(x_std >= X_mins) and np.all(x_std <= X_maxs) # type: ignore

    def nearest_data_distance(x_std: np.ndarray) -> float:
        return np.min(np.linalg.norm(X - x_std[None, :], axis=1))

    def gp_log_density(x_std: np.ndarray) -> float:
        if not inside_box(x_std):
            return -np.inf

        mu_pred, std_pred = gp.predict(x_std[None, :], return_std=True) # type: ignore
        mu_pred = float(mu_pred[0])
        std_pred = float(std_pred[0])

        d = nearest_data_distance(x_std)
        excess = max(0.0, d - trusted_radius)
        penalty = distance_penalty_strength * excess ** 2
        uncert_penalty = 0.25 * std_pred

        return mu_pred - penalty - uncert_penalty

    best_idx = int(np.argmax(y)) # type: ignore
    x_curr = X[best_idx].copy()
    logp_curr = gp_log_density(x_curr)

    if verbose:
        print("-" * 80)
        print("Starting Metropolis-Hastings in reduced feature space...")
        print(f"  n_mcmc         : {n_mcmc}")
        print(f"  burnin         : {burnin}")
        print(f"  thin           : {thin}")
        print(f"  proposal_scale : {proposal_scale}")
        print(f"  start logp     : {logp_curr:.6g}")

    chain = np.zeros((n_mcmc, len(feature_cols)))
    chain_logp = np.zeros(n_mcmc)
    accepted = 0

    for t in range(n_mcmc):
        x_prop = x_curr + rng.normal(0.0, proposal_scale, size=len(feature_cols))
        logp_prop = gp_log_density(x_prop)

        if np.isfinite(logp_prop):
            log_alpha = logp_prop - logp_curr
            if np.log(rng.uniform()) < log_alpha:
                x_curr = x_prop
                logp_curr = logp_prop
                accepted += 1

        chain[t] = x_curr
        chain_logp[t] = logp_curr

        if verbose and ((t + 1) % max(1, n_mcmc // 10) == 0):
            print(
                f"  iter {t+1:>7d}/{n_mcmc}, "
                f"acceptance={accepted/(t+1):.3f}, "
                f"current_logp={logp_curr:.4f}"
            )

    acceptance_rate = accepted / n_mcmc

    kept = chain[burnin::thin]
    kept_raw = x_scaler.inverse_transform(kept)
    samples_feature_space = pd.DataFrame(kept_raw, columns=feature_cols)

    mu_s, std_s = gp.predict(kept, return_std=True) # type: ignore
    samples_feature_space["gp_logL1_mean"] = mu_s
    samples_feature_space["gp_logL1_std"] = std_s

    # nearest-neighbor back-mapping into full parameter space
    nn_ids = cdist(kept, X).argmin(axis=1)
    borrowed = work.iloc[nn_ids].reset_index(drop=True)

    reconstructed = samples_feature_space.copy()
    reconstructed["source_experiment_id"] = borrowed["experiment_id"].values
    reconstructed["mu_a_div_zeta_0"] = borrowed["mu_a_div_zeta_0"].values
    reconstructed["sim_A"] = borrowed["sim_A"].values
    reconstructed["sim_f"] = borrowed["sim_f"].values

    reconstructed["mu"] = 10 ** reconstructed["log10_mu"]
    reconstructed["mu_a_times_zeta"] = 10 ** reconstructed["log10_mu_a_times_zeta"]
    reconstructed["fstar"] = 10 ** reconstructed["log10_fstar"]
    reconstructed["beta"] = 10 ** reconstructed["log10_beta"]

    reconstructed["zeta"] = np.sqrt(
        reconstructed["mu_a_times_zeta"] / (reconstructed["mu_a_div_zeta_0"] + EPS)
    )
    reconstructed["mu_a"] = reconstructed["mu_a_div_zeta_0"] * reconstructed["zeta"]

    enriched_rows = []
    for _, row in reconstructed.iterrows():
        theta_scaled = {
            "mu": row["mu"],
            "eta": row["eta"],
            "mu_a_times_zeta": row["mu_a_times_zeta"],
            "mu_a_div_zeta_0": row["mu_a_div_zeta_0"],
            "zeta": row["zeta"],
            "fstar": row["fstar"],
            "beta": row["beta"],
        }

        tau, mu_a_div_zeta = estimate_scaling(theta_scaled, row["sim_A"], row["sim_f"])

        mu_a_new = np.sqrt(theta_scaled["mu_a_times_zeta"] * mu_a_div_zeta)
        zeta_new = mu_a_new / mu_a_div_zeta

        params = infer_biological_params(
            theta_scaled,
            B_FIXED,
            mu_a_div_zeta,
            tau, dimension
        )

        rec = row.to_dict()
        rec.update({
            "tau": tau,
            "mu_a_div_zeta": mu_a_div_zeta,
            "mu_a_new": mu_a_new,
            "zeta_new": zeta_new,
        })
        rec.update(params)
        enriched_rows.append(rec)

    reconstructed_samples = pd.DataFrame(enriched_rows)

    train_predictions = work.copy()
    train_predictions["gp_pred_mean"] = y_pred
    train_predictions["gp_pred_std"] = y_std
    train_predictions["residual"] = y - y_pred

    diagnostics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "acceptance_rate": acceptance_rate,
        "trusted_radius": trusted_radius,
        "kernel": str(gp.kernel_),
        "n_rows_used": len(work),
        "n_samples_retained": len(samples_feature_space),
    }

    if make_plots:
        _plot_gp_diagnostics(
            work=work,
            feature_cols=feature_cols,
            target_col=target_col,
            train_predictions=train_predictions,
            samples_feature_space=samples_feature_space,
            chain_logp=chain_logp,
            nn_train=nn_train,
        )

    return GPLikelihoodExplorerResult(
        gp=gp,
        x_scaler=x_scaler,
        feature_cols=feature_cols,
        samples_feature_space=samples_feature_space,
        reconstructed_samples=reconstructed_samples,
        train_predictions=train_predictions,
        diagnostics=diagnostics,
    )

def plot_mcmc_density(samples: pd.DataFrame):
    if "F0" not in samples.columns or "v0" not in samples.columns:
        return
    plt.figure(figsize=(6, 5))
    plt.hexbin(samples["F0"], samples["v0"], gridsize=40)
    plt.xlabel("F0 [pN]")
    plt.ylabel("v0 [µm/s]")
    plt.title("Sample density: v0 vs F0")
    plt.colorbar(label="counts")
    plt.tight_layout()
    plt.show()

# =====================================================
# POSTERIOR FEATURE PREPARATION / EXPLORATION
# =====================================================

@dataclass
class GPPosteriorExplorerResult:
    gp: GaussianProcessRegressor
    x_scaler: StandardScaler
    feature_cols: List[str]
    samples_feature_space: pd.DataFrame
    train_predictions: pd.DataFrame
    diagnostics: Dict

def prepare_posterior_feature_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    out = df.copy()

    needed = [
        "mu", "eta", "zeta", "mu_a", "fstar", "beta",
        "tau_opt", "mu_a_div_zeta_opt", "B_opt",
    ]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"Missing required column for posterior feature table: {c}")

    rows = []
    for _, row in out.iterrows():
        try:
            theta_scaled = {
                "mu": row["mu"],
                "eta": row["eta"],
                "mu_a_times_zeta": row["mu_a"] * row["zeta"],
                "mu_a_div_zeta_0": row["mu_a"] / (row["zeta"] + EPS),
                "zeta": row["zeta"],
                "fstar": row["fstar"],
                "beta": row["beta"],
            }

            params = infer_biological_params(
                theta_scaled,
                row["B_opt"],
                row["mu_a_div_zeta_opt"],
                row["tau_opt"], dimension
            )

            rec = row.to_dict()
            rec.update(params)
            rows.append(rec)
        except Exception:
            rec = row.to_dict()
            rec.update({
                "K_d2": np.nan,
                "B": np.nan,
                "F0": np.nan,
                "Fc": np.nan,
                "v0": np.nan,
                "v0_eps0": np.nan,
                "pi0": np.nan,
                "b": np.nan,
            })
            rows.append(rec)

    return pd.DataFrame(rows)

def get_posterior_feature_cols() -> List[str]:
    return [
        "K_d2",
        "B",
        "F0",
        "Fc",
        "v0",
        "v0_eps0",
        "pi0",
        "b",
    ]

def compare_posterior_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_posterior",
    n_splits: int = 5,
    random_state: int = 0,
    make_plots: bool = True,
) -> pd.DataFrame:
    work = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    X = work[feature_cols].values.astype(float)
    y = work[target_col].values.astype(float)

    dim = X.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "GP_Matern_ARD_nu25": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(dim),
                    length_scale_bounds=(1e-3, 1e6),
                    nu=2.5,
                ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e2)),
                normalize_y=True,
                n_restarts_optimizer=8,
                random_state=random_state,
            ))
        ]),
        "GP_RBF_ARD": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=np.ones(dim),
                    length_scale_bounds=(1e-3, 1e6),
                ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e2)),
                normalize_y=True,
                n_restarts_optimizer=8,
                random_state=random_state,
            ))
        ]),
        "GP_RQ": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(
                    length_scale=1.0,
                    alpha=1.0,
                ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e2)),
                normalize_y=True,
                n_restarts_optimizer=8,
                random_state=random_state,
            ))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=6,
            learning_rate=0.03,
            max_iter=400,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=12, weights="distance"))
        ]),
        "SVR_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))
        ]),
    }

    results = []

    print("=" * 80)
    print("Comparing regression models on biological posterior feature space")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n--- Testing: {name} ---")

        cv_rmse, cv_mae, cv_r2 = [], [], []

        for train_idx, test_idx in kf.split(X):
            mdl = clone(model)
            mdl.fit(X[train_idx], y[train_idx])
            y_pred = mdl.predict(X[test_idx])

            cv_rmse.append(np.sqrt(mean_squared_error(y[test_idx], y_pred))) # type: ignore
            cv_mae.append(mean_absolute_error(y[test_idx], y_pred))# type: ignore
            cv_r2.append(r2_score(y[test_idx], y_pred))# type: ignore

        model.fit(X, y)
        y_fit = model.predict(X)

        train_rmse = np.sqrt(mean_squared_error(y, y_fit))# type: ignore
        train_mae = mean_absolute_error(y, y_fit)# type: ignore
        train_r2 = r2_score(y, y_fit)# type: ignore

        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train MAE : {train_mae:.4f}")
        print(f"Train R²  : {train_r2:.4f}")
        print(f"CV RMSE   : {np.mean(cv_rmse):.4f}")
        print(f"CV MAE    : {np.mean(cv_mae):.4f}")
        print(f"CV R²     : {np.mean(cv_r2):.4f}")

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner = model.named_steps["model"]
            if hasattr(inner, "kernel_"):
                print(f"Kernel    : {inner.kernel_}")

        results.append({
            "name": name,
            "model": model,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "cv_rmse": float(np.mean(cv_rmse)),
            "cv_mae": float(np.mean(cv_mae)),
            "cv_r2": float(np.mean(cv_r2)),
        })

        if make_plots:
            plt.figure(figsize=(5, 5))
            plt.scatter(y, y_fit, alpha=0.5)# type: ignore
            lo = min(y.min(), y_fit.min())# type: ignore
            hi = max(y.max(), y_fit.max())# type: ignore
            plt.plot([lo, hi], [lo, hi], "--")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(name)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(5, 4))
            plt.hist(y - y_fit, bins=30)
            plt.title(f"Residuals: {name}")
            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results).sort_values(["cv_rmse", "cv_mae"], ascending=True)

    print("\n" + "=" * 80)
    print("Posterior model ranking")
    print(results_df[["name", "train_rmse", "train_mae", "train_r2", "cv_rmse", "cv_mae", "cv_r2"]])
    print("=" * 80)

    return results_df

def explore_posterior_gp(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_posterior",
    n_mcmc: int = 20000,
    burnin: int = 3000,
    thin: int = 10,
    proposal_scale: float = 0.2,
    random_state: int = 0,
    distance_penalty_strength: float = 2.0,
    trusted_nn_quantile: float = 0.95,
    verbose: bool = True,
    make_plots: bool = True,
) -> GPPosteriorExplorerResult:
    rng = np.random.default_rng(random_state)

    work = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X_raw = work[feature_cols].values.astype(float)
    y = work[target_col].values.astype(float)

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_raw)

    pca = PCA(whiten=True)
    X = pca.fit_transform(X_scaled)

    if verbose:
        print("=" * 80)
        print("explore_posterior_gp: starting")
        print(f"Rows used            : {len(work)}")
        print(f"Feature columns      : {feature_cols}")
        print(f"Target column        : {target_col}")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(len(feature_cols)),
            length_scale_bounds=(1e-3, 1e6),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e2))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=random_state,
    )

    gp.fit(X, y)# type: ignore

    y_pred, y_std = gp.predict(X, return_std=True)# type: ignore
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))# type: ignore
    train_mae = mean_absolute_error(y, y_pred)# type: ignore
    train_r2 = r2_score(y, y_pred)# type: ignore

    dmat = cdist(X, X)
    np.fill_diagonal(dmat, np.inf)
    nn_train = dmat.min(axis=1)
    trusted_radius = np.quantile(nn_train, trusted_nn_quantile)

    if verbose:
        print("-" * 80)
        print("Posterior GP diagnostics:")
        print(f"  RMSE           : {train_rmse:.6g}")
        print(f"  MAE            : {train_mae:.6g}")
        print(f"  R²             : {train_r2:.6g}")
        print(f"  kernel         : {gp.kernel_}")
        print(f"  trusted radius : {trusted_radius:.6g}")

    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)

    def inside_box(x_std: np.ndarray) -> bool:
        return np.all(x_std >= X_mins) and np.all(x_std <= X_maxs)# type: ignore

    def nearest_data_distance(x_std: np.ndarray) -> float:
        return np.min(np.linalg.norm(X - x_std[None, :], axis=1))

    def gp_log_density(x_std: np.ndarray) -> float:
        if not inside_box(x_std):
            return -np.inf

        mu_pred, std_pred = gp.predict(x_std[None, :], return_std=True)# type: ignore
        mu_pred = float(mu_pred[0])
        std_pred = float(std_pred[0])

        d = nearest_data_distance(x_std)
        excess = max(0.0, d - trusted_radius)
        penalty = distance_penalty_strength * excess ** 2
        uncert_penalty = 0.25 * std_pred

        return mu_pred - penalty - uncert_penalty

    best_idx = int(np.argmax(y))# type: ignore
    x_curr = X[best_idx].copy()
    logp_curr = gp_log_density(x_curr)

    if verbose:
        print("-" * 80)
        print("Starting Metropolis-Hastings in posterior feature space...")
        print(f"  n_mcmc         : {n_mcmc}")
        print(f"  burnin         : {burnin}")
        print(f"  thin           : {thin}")
        print(f"  proposal_scale : {proposal_scale}")
        print(f"  start logp     : {logp_curr:.6g}")

    chain = np.zeros((n_mcmc, len(feature_cols)))
    chain_logp = np.zeros(n_mcmc)
    accepted = 0

    for t in range(n_mcmc):
        x_prop = x_curr + rng.normal(0.0, proposal_scale, size=len(feature_cols))
        logp_prop = gp_log_density(x_prop)

        if np.isfinite(logp_prop):
            log_alpha = logp_prop - logp_curr
            if np.log(rng.uniform()) < log_alpha:
                x_curr = x_prop
                logp_curr = logp_prop
                accepted += 1

        chain[t] = x_curr
        chain_logp[t] = logp_curr

        if verbose and ((t + 1) % max(1, n_mcmc // 10) == 0):
            print(
                f"  iter {t+1:>7d}/{n_mcmc}, "
                f"acceptance={accepted/(t+1):.3f}, "
                f"current_logp={logp_curr:.4f}"
            )

    acceptance_rate = accepted / n_mcmc

    kept = chain[burnin::thin]
    kept_scaled = pca.inverse_transform(kept)
    kept_raw = x_scaler.inverse_transform(kept_scaled)
    samples_feature_space = pd.DataFrame(kept_raw, columns=feature_cols)

    mu_s, std_s = gp.predict(kept, return_std=True)# type: ignore
    samples_feature_space["gp_logposterior_mean"] = mu_s
    samples_feature_space["gp_logposterior_std"] = std_s

    train_predictions = work.copy()
    train_predictions["gp_pred_mean"] = y_pred
    train_predictions["gp_pred_std"] = y_std
    train_predictions["residual"] = y - y_pred

    diagnostics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "acceptance_rate": acceptance_rate,
        "trusted_radius": trusted_radius,
        "kernel": str(gp.kernel_),
        "n_rows_used": len(work),
        "n_samples_retained": len(samples_feature_space),
    }

    if make_plots:
        _plot_gp_diagnostics(
            work=train_predictions,
            feature_cols=feature_cols,
            target_col=target_col,
            train_predictions=train_predictions,
            samples_feature_space=samples_feature_space,
            chain_logp=chain_logp,
            nn_train=nn_train,
        )

    return GPPosteriorExplorerResult(
        gp=gp,
        x_scaler=x_scaler,
        feature_cols=feature_cols,
        samples_feature_space=samples_feature_space,
        train_predictions=train_predictions,
        diagnostics=diagnostics,
    )

# def explore_posterior_extra_trees(
#     df: pd.DataFrame,
#     feature_cols: List[str],
#     target_col: str = "log_posterior",
#     n_mcmc: int = 20000,
#     burnin: int = 3000,
#     thin: int = 10,
#     proposal_scale: float = 0.2,
#     random_state: int = 0,
#     distance_penalty_strength: float = 2.0,
#     trusted_nn_quantile: float = 0.95,
#     verbose: bool = True,
#     make_plots: bool = True,
# ):
#     rng = np.random.default_rng(random_state)

#     work = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()

#     X_raw = work[feature_cols].values.astype(float)
#     y = work[target_col].values.astype(float)

#     # scale features (important for distance penalty + proposal)
#     x_scaler = StandardScaler()
#     X = x_scaler.fit_transform(X_raw)

#     # ---------------------------------------
#     # Train ExtraTrees
#     # ---------------------------------------
#     model = ExtraTreesRegressor(
#         n_estimators=500,
#         min_samples_leaf=2,
#         random_state=random_state,
#         n_jobs=-1,
#     )
#     model.fit(X, y)

#     y_pred = model.predict(X)

#     train_rmse = np.sqrt(mean_squared_error(y, y_pred))
#     train_mae = mean_absolute_error(y, y_pred)
#     train_r2 = r2_score(y, y_pred)

#     # ---------------------------------------
#     # uncertainty proxy (tree variance)
#     # ---------------------------------------
#     all_tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
#     y_std = all_tree_preds.std(axis=1)

#     # ---------------------------------------
#     # nearest neighbor stats
#     # ---------------------------------------
#     dmat = cdist(X, X)
#     np.fill_diagonal(dmat, np.inf)
#     nn_train = dmat.min(axis=1)
#     trusted_radius = np.quantile(nn_train, trusted_nn_quantile)

#     if verbose:
#         print("=" * 80)
#         print("ExtraTrees posterior surrogate")
#         print(f"  RMSE           : {train_rmse:.6g}")
#         print(f"  MAE            : {train_mae:.6g}")
#         print(f"  R²             : {train_r2:.6g}")
#         print(f"  trusted radius : {trusted_radius:.6g}")

#     X_mins = X.min(axis=0)
#     X_maxs = X.max(axis=0)

#     def inside_box(x):
#         return np.all(x >= X_mins) and np.all(x <= X_maxs)

#     def nearest_data_distance(x):
#         return np.min(np.linalg.norm(X - x[None, :], axis=1))

#     def predict_with_uncertainty(x):
#         preds = np.array([t.predict(x[None, :])[0] for t in model.estimators_])
#         return preds.mean(), preds.std()

#     def predict_mean(x):
#         return model.predict(x[None, :])[0]

#     def log_density(x):
#         if not inside_box(x):
#             return -np.inf

#         mu = predict_mean(x)

#         d = nearest_data_distance(x)
#         excess = max(0.0, d - trusted_radius)

#         penalty = distance_penalty_strength * excess**2

#         return mu - penalty

#     # ---------------------------------------
#     # MCMC
#     # ---------------------------------------
#     best_idx = int(np.argmax(y))
#     x_curr = X[best_idx].copy()
#     logp_curr = log_density(x_curr)

#     chain = np.zeros((n_mcmc, len(feature_cols)))
#     chain_logp = np.zeros(n_mcmc)

#     accepted = 0

#     for t in range(n_mcmc):
#         x_prop = x_curr + rng.normal(0.0, proposal_scale, size=len(feature_cols))
#         logp_prop = log_density(x_prop)

#         if np.isfinite(logp_prop):
#             if np.log(rng.uniform()) < (logp_prop - logp_curr):
#                 x_curr = x_prop
#                 logp_curr = logp_prop
#                 accepted += 1

#         chain[t] = x_curr
#         chain_logp[t] = logp_curr

#         if verbose and ((t + 1) % (n_mcmc // 10) == 0):
#             print(f"iter {t+1}/{n_mcmc}  acc={accepted/(t+1):.3f}  logp={logp_curr:.3f}")

#     acceptance_rate = accepted / n_mcmc

#     # ---------------------------------------
#     # Collect samples
#     # ---------------------------------------
#     kept = chain[burnin::thin]
#     kept_raw = x_scaler.inverse_transform(kept)

#     samples = pd.DataFrame(kept_raw, columns=feature_cols)

#     mu_s = []
#     std_s = []
#     for x in kept:
#         m, s = predict_with_uncertainty(x)
#         mu_s.append(m)
#         std_s.append(s)

#     samples["et_logposterior_mean"] = mu_s
#     samples["et_logposterior_std"] = std_s

#     # ---------------------------------------
#     # diagnostics
#     # ---------------------------------------
#     train_predictions = work.copy()
#     train_predictions["pred_mean"] = y_pred
#     train_predictions["pred_std"] = y_std
#     train_predictions["residual"] = y - y_pred

#     diagnostics = {
#         "train_rmse": train_rmse,
#         "train_mae": train_mae,
#         "train_r2": train_r2,
#         "acceptance_rate": acceptance_rate,
#         "trusted_radius": trusted_radius,
#         "n_rows_used": len(work),
#         "n_samples_retained": len(samples),
#     }

#     if make_plots:
#         _plot_gp_diagnostics(
#             work=train_predictions,
#             feature_cols=feature_cols,
#             target_col=target_col,
#             train_predictions=train_predictions,
#             samples_feature_space=samples,
#             chain_logp=chain_logp,
#             nn_train=nn_train,
#         )

#     return samples, diagnostics

from sklearn.decomposition import PCA
def explore_posterior_distance_only(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_posterior",  # unused, kept for compatibility
    n_mcmc: int = 20000,
    burnin: int = 3000,
    thin: int = 10,
    proposal_scale: float = 0.2,
    random_state: int = 0,
    distance_penalty_strength: float = 2.0,
    trusted_nn_quantile: float = 0.95,
    verbose: bool = True,
    make_plots: bool = True,
    ) -> GPPosteriorExplorerResult:


    rng = np.random.default_rng(random_state)

    # ---------------------------------------
    # DATA PREP (same as before)
    # ---------------------------------------
    work = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X_raw = work[feature_cols].values.astype(float)

    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X_raw)
    pca = PCA(whiten=True)
    X = pca.fit_transform(X)

    if verbose:
        print("=" * 80)
        print("explore_posterior_distance_only: starting")
        print(f"Rows used       : {len(work)}")
        print(f"Feature columns : {feature_cols}")

    # ---------------------------------------
    # nearest-neighbor structure
    # ---------------------------------------
    dmat = cdist(X, X)
    np.fill_diagonal(dmat, np.inf)
    nn_train = dmat.min(axis=1)

    trusted_radius = np.quantile(nn_train, trusted_nn_quantile)

    if verbose:
        print("-" * 80)
        print("Distance-only diagnostics:")
        print(f"  trusted radius : {trusted_radius:.6g}")

    # ---------------------------------------
    # bounding box
    # ---------------------------------------
    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)

    def inside_box(x):
        return np.all(x >= X_mins) and np.all(x <= X_maxs)

    def nearest_data_distance(x):
        return np.min(np.linalg.norm(X - x[None, :], axis=1))

    # ---------------------------------------
    # PURE DISTANCE LOG DENSITY
    # ---------------------------------------
    def log_density(x):
        if not inside_box(x):
            return -np.inf

        d = nearest_data_distance(x)
        excess = max(0.0, d - trusted_radius)

        penalty = distance_penalty_strength * excess**2

        return -penalty  # ONLY geometry

    # ---------------------------------------
    # START POINT (random data point)
    # ---------------------------------------
    start_idx = rng.integers(len(X))
    x_curr = X[start_idx].copy()
    logp_curr = log_density(x_curr)

    if verbose:
        print("-" * 80)
        print("Starting MCMC (distance-only)...")
        print(f"  start logp : {logp_curr:.6g}")

    chain = np.zeros((n_mcmc, len(feature_cols)))
    chain_logp = np.zeros(n_mcmc)
    accepted = 0

    for t in range(n_mcmc):
        x_prop = x_curr + rng.normal(0.0, proposal_scale, size=len(feature_cols))
        logp_prop = log_density(x_prop)

        if np.isfinite(logp_prop):
            if np.log(rng.uniform()) < (logp_prop - logp_curr):
                x_curr = x_prop
                logp_curr = logp_prop
                accepted += 1

        chain[t] = x_curr
        chain_logp[t] = logp_curr

        if verbose and ((t + 1) % max(1, n_mcmc // 10) == 0):
            print(
                f"  iter {t+1:>7d}/{n_mcmc}, "
                f"acceptance={accepted/(t+1):.3f}, "
                f"logp={logp_curr:.4f}"
            )

    acceptance_rate = accepted / n_mcmc

    # ---------------------------------------
    # samples
    # ---------------------------------------
    kept = chain[burnin::thin]
    kept_scaled = pca.inverse_transform(kept)
    kept_raw = x_scaler.inverse_transform(kept_scaled)

    samples_feature_space = pd.DataFrame(kept_raw, columns=feature_cols)

    # fake "prediction" columns so plotting still works
    samples_feature_space["gp_logposterior_mean"] = 0.0
    samples_feature_space["gp_logposterior_std"] = 0.0

    # ---------------------------------------
    # training predictions (dummy)
    # ---------------------------------------
    train_predictions = work.copy()
    train_predictions["gp_pred_mean"] = 0.0
    train_predictions["gp_pred_std"] = 0.0
    train_predictions["residual"] = 0.0

    diagnostics = {
        "acceptance_rate": acceptance_rate,
        "trusted_radius": trusted_radius,
        "n_rows_used": len(work),
        "n_samples_retained": len(samples_feature_space),
        "mode": "distance_only",
    }

    if make_plots:
        _plot_gp_diagnostics(
            work=train_predictions,
            feature_cols=feature_cols,
            target_col=target_col,
            train_predictions=train_predictions,
            samples_feature_space=samples_feature_space,
            chain_logp=chain_logp,
            nn_train=nn_train,
        )

    return GPPosteriorExplorerResult(
        gp=None,  # no GP # type: ignore
        x_scaler=x_scaler,
        feature_cols=feature_cols,
        samples_feature_space=samples_feature_space,
        train_predictions=train_predictions,
        diagnostics=diagnostics,
    )



# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dim",
        choices=["2d", "3d"],
        default="3d",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Optional: override experiment_ids",
    )

    args = parser.parse_args()


    stage = "S3"

    # DEFAULT IDS (per dimension)
    default_ids = {
        "2d": ["afb41c1d26502de4"],
        "3d": ["e4b43770665d2ab2"],
    }

    # use user-provided ids OR defaults
    ids = args.ids if args.ids is not None else default_ids[args.dim]

    print(f"\nRunning analysis for {args.dim.upper()} (stage={stage})")
    print(f"Using IDs: {ids}")

    df = load_df(stage, args.dim)

    for exp_id in ids:
        row = df[df["experiment_id"] == exp_id]

        if len(row) == 0:
            print(f"\n⚠️ Experiment not found: {exp_id}")
            print("Available IDs (first 5):", df["experiment_id"].head().tolist())
            continue

        analyze_experiment(row.iloc[0], dimension=args.dim)
    
   
    
    # =====================================================
    # GP POSTERIOR EXPLORATION (log_posterior)
    # =====================================================
    print("\n" + "=" * 80)
    print("Running biological-space posterior exploration (log_posterior)")
    print("=" * 80)

    df_post = prepare_posterior_feature_table(df, dimension=args.dim)
    posterior_feature_cols = get_posterior_feature_cols()


    posterior_threshold = -4.5#2d: -6#-15.0
    l1_threshold = -1.5

    df_post_clean = df_post[
        np.isfinite(df_post["log_posterior"]) &
        np.isfinite(df_post["logL1"]) &
        (df_post["log_posterior"] > posterior_threshold) &
        (df_post["logL1"] > l1_threshold)
    ].copy()

    print(f"\nFiltered posterior dataset:")
    print(f"  total rows                 : {len(df_post)}")
    print(f"  after posterior threshold  : {(np.isfinite(df_post['log_posterior']) & (df_post['log_posterior'] > posterior_threshold)).sum()}")
    print(f"  after joint thresholds     : {len(df_post_clean)}")
    print(f"    with log_posterior > {posterior_threshold}")
    print(f"    and  logL1 > {l1_threshold}")
    # return
    # posterior_results_df = compare_posterior_models(
    #     df_post_clean,
    #     feature_cols=posterior_feature_cols,
    #     target_col="log_posterior",
    #     make_plots=True,
    # )

    # print("\nTop posterior models:")
    # print(
    #     posterior_results_df.head(10)[
    #         ["name", "cv_rmse", "cv_mae", "cv_r2", "train_r2"]
    #     ]
    # )

    posterior_res = explore_posterior_gp(#explore_posterior_distance_only
        df_post_clean,
        feature_cols=posterior_feature_cols,
        target_col="log_posterior",
        n_mcmc=50000,
        burnin=10000,
        thin=20,
        proposal_scale=0.2,
        random_state=42,
        distance_penalty_strength=1.0,
        trusted_nn_quantile=0.95,
        make_plots=True,
        verbose=True,
    )
    print("\nSampled posterior-space points:")
    print(posterior_res.samples_feature_space.head())

    print("\nPosterior diagnostics:")
    print(posterior_res.diagnostics)

    export_samples(
        posterior_res.samples_feature_space,
        f"gp_logposterior_samples_{args.dim}.csv",
    )

    plot_mcmc_density(posterior_res.samples_feature_space)

    # samples, diagnostics = explore_posterior_extra_trees(
    #     df_post_clean,
    #     feature_cols=posterior_feature_cols,
    #     target_col="log_posterior",
    #     n_mcmc=20000,
    #     burnin=3000,
    #     thin=10,
    #     proposal_scale=0.2,
    #     random_state=42,
    #     distance_penalty_strength=2.0,
    #     trusted_nn_quantile=0.95,
    #     make_plots=True,
    #     verbose=True,
    # )

    # print("\nSampled posterior-space points:")
    # print(samples.head())

    # print("\nPosterior diagnostics:")
    # print(diagnostics)

    # export_samples(
    #     samples,
    #     f"et_logposterior_samples_{args.dim}.csv",
    # )

    # plot_mcmc_density(samples)

   

if __name__ == "__main__":
    main()
