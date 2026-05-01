import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def build_ml_features_from_df(df: pd.DataFrame) -> np.ndarray:
    mu = df["mu"].to_numpy(dtype=float)
    eta = df["eta"].to_numpy(dtype=float)
    zeta = df["zeta"].to_numpy(dtype=float)
    mu_a = df["mu_a"].to_numpy(dtype=float)
    fstar = df["fstar"].to_numpy(dtype=float)
    beta = df["beta"].to_numpy(dtype=float)

    mu_a_times_zeta = mu_a * zeta

    X = np.column_stack([
        np.log(mu),
        eta,
        np.log(mu_a_times_zeta),
        np.log(fstar),
        np.log(beta),
    ])
    return X


def build_ml_features_from_sample(sample: dict) -> np.ndarray:
    mu = float(sample["mu"])
    eta = float(sample["eta"])
    zeta = float(sample["zeta"])
    mu_a = float(sample["mu_a"])
    fstar = float(sample["fstar"])
    beta = float(sample["beta"])

    mu_a_times_zeta = mu_a * zeta

    x = np.array([
        np.log(mu),
        eta,
        np.log(mu_a_times_zeta),
        np.log(fstar),
        np.log(beta),
    ], dtype=float).reshape(1, -1)
    return x


def train_svm_from_dataframe(
    df: pd.DataFrame,
    class_weight=None,
    kernel: str = "rbf",
    probability: bool = True,
):
    if class_weight is None:
        class_weight = {0: 1, 1: 5}

    X = build_ml_features_from_df(df)
    y = df["oscillatory"].to_numpy(dtype=int)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=kernel, # type: ignore
            probability=probability,
            class_weight=class_weight,
        ),
    )
    model.fit(X, y)
    return model


def svm_accept_sample(model, sample: dict, tau: float = 0.1) -> bool:
    x = build_ml_features_from_sample(sample)
    p = model.predict_proba(x)[0, 1]
    return bool(p > tau)


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def diagnose_gp_uncertainty(y_true, y_pred, y_std):
    residuals = np.abs(y_true - y_pred)

    corr = np.corrcoef(residuals, y_std)[0, 1]

    print("\n=== UNCERTAINTY DIAGNOSTICS ===")
    print(f"corr(|error|, std) = {corr:.3f}")

    # coverage test
    within_1sigma = np.mean(residuals < y_std)
    within_2sigma = np.mean(residuals < 2*y_std)

    print(f"Within 1σ: {within_1sigma:.3f} (ideal ~0.68)")
    print(f"Within 2σ: {within_2sigma:.3f} (ideal ~0.95)")

def plot_gp_diagnostics(y_true, y_pred, y_std):
    plt.figure(figsize=(6, 6))

    plt.errorbar(
        y_true,
        y_pred,
        yerr=y_std,
        fmt='o',
        alpha=0.5
    )

    lims = [min(y_true), max(y_true)]
    plt.plot(lims, lims, 'k--')

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("GP Predictions")

    plt.tight_layout()
    plt.show()

def train_gp_from_dataframe(df: pd.DataFrame, target_col: str = "score"):
    X = build_ml_features_from_df(df)
    y = df[target_col].to_numpy(dtype=float)

    kernel = (
        C(1.0, (1e-2, 1e3)) *
        RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    gp.fit(X, y)
    return gp

def find_safe_threshold(y_true, y_prob, max_fn_rate=0.01):
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

def train_svm_from_s2(df_s2, max_fn_rate=0.01):
    # --- define label: BOTH oscillate ---
    y = (
        (df_s2["oscillatory_full"].fillna(0) == 1) &
        (df_s2["oscillatory_reduced"].fillna(0) == 1)
    ).astype(int)

    df = df_s2.assign(oscillatory=y)

    X = build_ml_features_from_df(df)
    y = df["oscillatory"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(probability=True, class_weight={0: 1, 1: 5}),
    )

    model.fit(X_train, y_train)

    # --- compute safe threshold ---
    y_prob = model.predict_proba(X_test)[:, 1]
    tau = find_safe_threshold(y_test, y_prob, max_fn_rate=max_fn_rate)

    # print(f"[S2 SVM] tau_safe = {tau:.4f}")

    return model, tau

def gp_ucb_score(gp, sample: dict, alpha=1.0) -> float:
    x = build_ml_features_from_sample(sample)
    mean, std = gp.predict(x, return_std=True)
    return float(mean[0] + alpha*std[0])