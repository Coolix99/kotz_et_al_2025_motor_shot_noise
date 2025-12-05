import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from scipy.stats import  beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config_sbi import (
    mean_mu,
    log_std_mu,
    alpha_eta,
    beta_eta,
    mean_mu_a_times_zeta,
    log_std_mu_a_times_zeta,
    mean_fstar, log_std_fstar,
    mean_beta, log_std_beta,
    feature_keys,log_keys,parameter_keys
)

# -----------------------------------------------------------
# priors
# -----------------------------------------------------------

# ---- Prior definitions ----
def sample_mu(rng):
    val = rng.normal(np.log10(mean_mu), log_std_mu)
    return 10 ** val

def sample_eta(rng):
    # Beta distribution in [0,1]
    return beta.rvs(alpha_eta, beta_eta, random_state=rng)

def sample_mu_a_times_zeta(rng):  # log10-normal
    val = rng.normal(np.log10(mean_mu_a_times_zeta), log_std_mu_a_times_zeta)
    return 10 ** val

def sample_fstar(rng):
    val = rng.normal(np.log10(mean_fstar), log_std_fstar)
    return 10 ** val

def sample_beta(rng):
    val = rng.normal(np.log10(mean_beta), log_std_beta)
    return 10 ** val + 2.0

def sample_from_prior(rng, n: int):
    """
    Draw n samples from the 3D prior:
        mu  ~ log10-normal(mean_mu, log_std_mu)
        eta ~ Beta(alpha_eta, beta_eta)
        mu_a_times_zeta ~ log10-normal(mean_mu_a_times_zeta, log_std_mu_a_times_zeta)
    Returns:
        samples : np.ndarray of shape (n, 3)
        keys    : ['mu', 'eta', 'mu_a_times_zeta']
    """
    # Each parameter’s independent draw
    mu_samples = [sample_mu(rng) for _ in range(n)]
    eta_samples = [sample_eta(rng) for _ in range(n)]
    mu_a_zeta_samples = [sample_mu_a_times_zeta(rng) for _ in range(n)]

    fstar_samples = [sample_fstar(rng) for _ in range(n)]
    beta_samples = [sample_beta(rng) for _ in range(n)]

    samples = np.column_stack([mu_samples, eta_samples, mu_a_zeta_samples, fstar_samples, beta_samples])
    return samples, ["mu", "eta", "mu_a_times_zeta", "fstar", "beta"]


# Compact hash for saving
def prior_settings_hash():
    data = (mean_mu, log_std_mu, alpha_eta, beta_eta, mean_mu_a_times_zeta, log_std_mu_a_times_zeta)
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]

# -----------------------------------------------------------
# load
# -----------------------------------------------------------

def load_s1_data(csv_path):
    """Load summary CSV and return (X, y_lik, y_bool, y_wave, feature_keys)."""
    if not os.path.exists(csv_path):
        logging.warning(f"No CSV found at {csv_path}")
        return None, None, None, None, None

    df = pd.read_csv(csv_path)
    if len(df) < 5:
        logging.warning(f"Too few samples in {csv_path}")
        return None, None, None, None, None

    # Parameter names
    

    # Copy data and transform selectively
    X = df[feature_keys].copy().values
    for i, k in enumerate(feature_keys):
        if k in log_keys:
            X[:, i] = np.log10(X[:, i] + 1e-12)

    # Extract targets
    L = df["likelihood"].values
    y_lik = np.log(L + 1e-12)
    y_bool = (L > 0).astype(int)
    y_wave = df["wavelength"].values if "wavelength" in df.columns else None

    return X, y_lik, y_bool, y_wave, feature_keys

def load_s2_data(csv_path):
    """
    Load S2 summary CSV and return feature/target arrays for GP training.

    Features:
        X = [log10(mu), eta (linear), log10(mu_a * zeta)]

    Returns
    -------
    X, y_lik, y_bool, y_aux, feature_keys
      - y_aux is a dict with keys:
          "wavelength_norm", "wavelength_pert",
          "variance_amp_norm", "variance_amp_pert",
          "freq_ratio"
    """
    import pandas as pd
    import numpy as np
    if not os.path.exists(csv_path):
        logging.warning(f"[S2] CSV not found: {csv_path}")
        return None, None, None, None, None

    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["likelihood"]+parameter_keys)
    df['mu_a_times_zeta']=df['mu_a']*df['zeta']

    if len(df) < 5:
        logging.warning(f"[S2] Too few valid rows in {csv_path}")
        return None, None, None, None, None

    # features

    X = df[feature_keys].copy().values
    for i, k in enumerate(feature_keys):
        if k in log_keys:
            X[:, i] = np.log10(X[:, i] + 1e-12)

    # targets
    y_lik = np.log(df["likelihood"].values + 1e-12)
    y_bool = (df["likelihood"].values > 0).astype(int)

    # auxiliary observables
    aux_keys = [
        "wavelength_norm",
        "wavelength_pert",
        "variance_amp_norm",
        "variance_amp_pert",
        "freq_ratio",
    ]
    y_aux = {k: df[k].values for k in aux_keys if k in df.columns}

    return X, y_lik, y_bool, y_aux, feature_keys

def load_s3_data(csv_path):
    """
    Load S3 summary CSV and return features + observables for GP fitting.
    """
    import pandas as pd
    import numpy as np
    if not os.path.exists(csv_path):
        logging.warning(f"[S3] CSV not found: {csv_path}")
        return None, None, None, None, None

    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["likelihood"]+parameter_keys)
    df['mu_a_times_zeta']=df['mu_a']*df['zeta']

    if len(df) < 5:
        logging.warning(f"[S3] Too few valid rows in {csv_path}")
        return None, None, None, None, None

    X = df[feature_keys].copy().values
    for i, k in enumerate(feature_keys):
        if k in log_keys:
            X[:, i] = np.log10(X[:, i] + 1e-12)

    y_lik = np.log(df["likelihood"].values + 1e-12)
    y_bool = (df["likelihood"].values > 0).astype(int)

    aux_keys = [
        "wavelength_norm",
        "wavelength_pert",
        "variance_amp_norm",
        "variance_amp_pert",
        "freq_ratio",
        "Q_norm",
        "Q_pert",
    ]
    y_aux = {k: df[k].values for k in aux_keys if k in df.columns}

    return X, y_lik, y_bool, y_aux, feature_keys


# -----------------------------------------------------------
# gp helper
# -----------------------------------------------------------

def fit_gp_scalar(X, y):
    """Fit GP on a scalar observable (e.g., log-likelihood, wavelength)."""
    kernel = (
        C(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.ones(X.shape[1]), nu=2.5)
        + WhiteKernel()
    )
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3),
    )
    gp.fit(X, y)
    return gp

def fit_gp_safe(X, y):
    """Fit GP after filtering NaNs and invalid values."""
    mask = np.isfinite(y)
    n_valid = np.sum(mask)
    if n_valid < 5:
        logging.warning(f"Too few valid samples for GP({n_valid})")
        return None
    return fit_gp_scalar(X[mask], y[mask])

def fit_classifier(X, y_bool):
    """Fit a Random Forest classifier for oscillatory vs. non-oscillatory behavior."""
    # Handle degenerate case: all 0 or all 1
    if len(np.unique(y_bool)) < 2:
        logging.warning("Classifier: only one class present in y_bool — skipping fit.")
        return {
            "model": None,
            "class_balance": {int(y_bool[0]): len(y_bool)},
        }

    # Simple and robust classifier
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=0,
            n_jobs=-1
        )
    )

    clf.fit(X, y_bool)
    return clf

def acquisition_scores(acq, mu, sigma, y_best, beta=2.0):
    """Compute acquisition scores for given strategy."""
    if acq == "EI":
        from scipy.stats import norm
        Z = (mu - y_best) / (sigma + 1e-9)
        scores = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
    elif acq == "UCB":
        scores = mu + beta * sigma
    elif acq == "MES":
        # simplified entropy-based approach (not full MES)
        scores = np.log(sigma + 1e-9)
    else:
        raise ValueError(f"Unknown acquisition: {acq}")
    return scores

