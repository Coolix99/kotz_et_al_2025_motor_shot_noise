import os
from cuda.tests.test_cython import item
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cilia.datasets.DataSet import DataSet
from cilia.datasets.Realization import Realization

from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
    CONDITION_DESCRIPTION,
    A, F, GEYER, Q,OSCILLATORY,
    DataName
)
from cilia.datastructures.data_schemas import ScalarSchema

from cilia.transformers.Transformer import (
    data_transformer,
    run_transformer_on_dataset,
    run_transformer_on_realization,
    run_viewer_on_dataset,
)

from cilia.transformers.a_transformer import estimate_amplitude_psd_t
from cilia.transformers.f_transformer import estimate_f_from_tangent_angle_power_t
from cilia.transformers.geyer_transformer import get_geyer_fit_cycle_t
from cilia.transformers.phase_transformer import (
    protophase_from_spatial_modes_t,
    estimate_phase_from_protophase_t,
)
from cilia.transformers.periodic_avg_transformer import periodic_avg_t
from cilia.transformers.q_transformer import estimate_Q_msd_fixed_tau_t

class SimulationDataset(DataSet):
    def __init__(self, path: str):
        super().__init__(path)

    def _initialize_realizations(self):
        if not os.path.isdir(self.path):
            self.realizations = []
            return
        exp_dirs = [
            d for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        self.realizations = [Realization(exp_id) for exp_id in sorted(exp_dirs)]

@data_transformer(
    inputs=(TANGENT_ANGLE_SERIES,),
    outputs=(OSCILLATORY,),
)
def detect_oscillation_t(gamma):
    # gamma: (Nt, Ns)

    if gamma.ndim != 2 or gamma.shape[0] < 10:
        return {OSCILLATORY.key: False}
    if not np.isfinite(gamma).all():
        return {OSCILLATORY.key: False}
    # global variance (quick reject)
    std = np.std(gamma)
    if std < 1e-6:
        return {OSCILLATORY.key: False}
    if std > 1e10:
        return {OSCILLATORY.key: False}

    # remove temporal mean per spatial point
    gamma_centered = gamma - np.mean(gamma, axis=0, keepdims=True)
    fft = np.fft.rfft(gamma_centered, axis=0)
    power_s = np.abs(fft) ** 2  # shape: (Nt_freq, Ns)
    power = np.mean(power_s, axis=1)  # shape: (Nt_freq,)

    Nt = len(power)
    power[0] = 0

    k_peak = int(np.argmax(power))
    # frequency window constraint
    k_min = 3
    k_max = Nt // 3

    valid_frequency = (k_peak >= k_min) and (k_peak <= k_max)
    peak = power[k_peak]
    mean_power = np.mean(power)
    is_oscillatory = (peak > 2 * mean_power) and valid_frequency

    # print(
    #     f"[oscillation check] Nt={Nt}, k_peak={k_peak}, "
    #     f"period≈{Nt / max(k_peak,1):.2f}, "
    #     f"peak/mean={peak/mean_power:.2f}, "
    #     f"valid_freq={valid_frequency}, "
    #     f"→ oscillatory={is_oscillatory}"
    # )

    # # kymograph
    # plt.figure(figsize=(6, 4))
    # plt.imshow(gamma, aspect="auto", origin="lower", cmap="viridis")
    # plt.colorbar(label="psi")
    # plt.title(f"osc={is_oscillatory}, k={k_peak}")
    # plt.tight_layout()
    # plt.show()



    return {
        OSCILLATORY.key: bool(is_oscillatory)
    }

def count_realizations(dataset_path):
    if not os.path.isdir(dataset_path):
        return 0
    return len([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

def collect_conditions(dataset: DataSet) -> pd.DataFrame:
    rows = []

    def on_combo(realization, combo, combo_rows):
        row = combo_rows[0]
        value = row["value"]
        if not isinstance(value, dict):
            return
        rows.append({
            "experiment_id": realization.experiment_id,
            "mu_a": value.get("mu_a"),
            "mu": value.get("mu"),
            "eta": value.get("eta"),
            "zeta": value.get("zeta"),
            "beta": value.get("beta"),
            "fstar": value.get("fstar"),
            "Nmotor": value.get("Nmotor"),
            "strategy": value.get("strategy"),
        })

    run_viewer_on_dataset(dataset, inputs=(CONDITION_DESCRIPTION,), on_combo=on_combo)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df = df.drop_duplicates(subset=["experiment_id"])
    df["experiment_id"] = df["experiment_id"].astype(str)
    return df

def run_pipeline_s1(dataset: DataSet):
    segment_ids = [
        item.id
        for realization in dataset
        for item in realization.data_items.values()
        if item.data_name.key == TANGENT_ANGLE_SERIES.key
    ]

    run_transformer_on_dataset(
        dataset,
        detect_oscillation_t,
        allowed_ids={"tangent_angle_series": segment_ids},
        skip_existing=True,
    )

def collect_scalar_dense_s1(dataset, conditions_df, out_csv, dimension):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        out = {"experiment_id": exp_id}
        items = realization.data_items.values()

        def get_items(key):
            return [it for it in items if it.data_name.key == key]

        def extract_scalar(items, getter):
            vals = []
            for it in items:
                try:
                    v = it.resolve(dataset, os.path.join(dataset.path, exp_id))
                    vals.append(getter(v))
                except Exception:
                    continue
            return np.nanmean(vals) if vals else np.nan

        # --- S1 target ---
        out["oscillatory"] = extract_scalar(
            get_items(OSCILLATORY.key),
            lambda v: float(v),  # bool → 0/1
        )

        rows.append(out)

        # Unload cached data to prevent memory leak
        realization.unload_data()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    cols = [
        "experiment_id",
        "mu_a", "mu", "eta", "zeta", "beta", "fstar", "Nmotor", "strategy",
        "oscillatory",
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    return df

def depends_on_algorithm(realization, item, algorithm_name):
    stack = list(item.dependencies)
    visited = set()

    while stack:
        dep_id = stack.pop()
        if dep_id in visited:
            continue
        visited.add(dep_id)

        dep = realization.data_items.get(dep_id)
        if dep is None:
            continue

        if (
            dep.data_name.key == TANGENT_ANGLE_SERIES.key
            and dep.algorithm == algorithm_name
        ):
            return True

        stack.extend(dep.dependencies)

    return False

def run_pipeline_s2(dataset: DataSet):
    print("\nRunning S2 pipeline")

    valid_realizations = []

    for realization in dataset:

        reduced_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == TANGENT_ANGLE_SERIES.key
            and item.algorithm == "reduced"
        ]

        if not reduced_ids:
            continue

        # --- reduced oscillation ---
        run_transformer_on_realization(
            dataset,
            realization,
            detect_oscillation_t,
            allowed_ids={"tangent_angle_series": reduced_ids},
        )

        osc_reduced = [
            item for item in realization.data_items.values()
            if item.data_name.key == OSCILLATORY.key
            and set(item.dependencies).intersection(reduced_ids)
        ]

        if not any(bool(item.resolve(dataset, os.path.join(dataset.path, realization.experiment_id))) for item in osc_reduced):
            continue

        # --- full ---
        full_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == TANGENT_ANGLE_SERIES.key
            and item.algorithm == "full"
        ]

        if not full_ids:
            continue

        run_transformer_on_realization(
            dataset,
            realization,
            detect_oscillation_t,
            allowed_ids={"tangent_angle_series": full_ids},
        )

        osc_full = [
            item for item in realization.data_items.values()
            if item.data_name.key == OSCILLATORY.key
            and set(item.dependencies).intersection(full_ids)
        ]

        if not any(bool(item.resolve(dataset, os.path.join(dataset.path, realization.experiment_id))) for item in osc_full):
            continue

        # --- passed gating ---
        valid_realizations.append(realization.experiment_id)

        tangent_ids = reduced_ids + full_ids

        for tr in (estimate_amplitude_psd_t, estimate_f_from_tangent_angle_power_t):
            run_transformer_on_realization(
                dataset,
                realization,
                tr,
                allowed_ids={"tangent_angle_series": tangent_ids},
            )


        run_transformer_on_realization(
            dataset,
            realization,
            protophase_from_spatial_modes_t,
            allowed_ids={"tangent_angle_series": tangent_ids},
        )

        run_transformer_on_realization(
            dataset,
            realization,
            estimate_phase_from_protophase_t,
            allowed_algorithms=[protophase_from_spatial_modes_t.algorithm],
        )
        #print(realization.experiment_id)

        run_transformer_on_realization(
            dataset,
            realization,
            periodic_avg_t,
            allowed_algorithms_by_key={
                estimate_phase_from_protophase_t.outputs[0].key: [
                    estimate_phase_from_protophase_t.algorithm
                ],
            },
            match_ancestor_keys={TANGENT_ANGLE_SERIES.key},
        )

        run_transformer_on_realization(
            dataset,
            realization,
            get_geyer_fit_cycle_t,
        )

    print(f"S2 done: {len(valid_realizations)} valid realizations")

from config_sbi import (
    DENSITY_PER_GAP_2D,
    DENSITY_PER_GAP_3D,
    LIKELIHOOD_1_CONFIG,
    BIO_PRIORS,
    LIKELIHOOD_2_CONFIG,
)
from config_sbi import (
    LIKELIHOOD_1_CONFIG,
    LIKELIHOOD_1_Q_CONFIG,
    BIO_PRIORS,
    LIKELIHOOD_2_CONFIG,
)
from config_sbi import BIO_CONSTANTS
from scipy.optimize import minimize

def gaussian_loglik(x, mu, sigma):
    if not np.isfinite(x):
        return -np.inf
    return -0.5 * ((x - mu) / sigma) ** 2

def gaussian_loglik_optimal_sigma(x, mu, sigma_obs):
    if not np.isfinite(x):
        return -np.inf
    return -0.5 *(1+np.log(2*np.pi*(sigma_obs*sigma_obs+(mu-x)*(mu-x))))

def compute_l1(ampl_ratio, freq_ratio, lam_norm, lam_pert):
    cfg = LIKELIHOOD_1_CONFIG

    l = 0.0

    # wavelength (absolute)
    if np.isfinite(lam_norm):
        l += gaussian_loglik_optimal_sigma(lam_norm, cfg["wavelength"]["norm"],cfg["wavelength"]["sigma"])
    if np.isfinite(lam_pert):
        l += gaussian_loglik_optimal_sigma(lam_pert, cfg["wavelength"]["pert"],cfg["wavelength"]["sigma"])

    # ratios
    l += gaussian_loglik_optimal_sigma(ampl_ratio, cfg["amplitude_ratio"]["expected"],cfg["amplitude_ratio"]["sigma"])
    l += gaussian_loglik_optimal_sigma(freq_ratio, cfg["frequency_ratio"]["expected"],cfg["frequency_ratio"]["sigma"])

    return l

def estimate_scaling(theta_scaled, sim_A, sim_f):
    cfg = LIKELIHOOD_2_CONFIG

    if not all(np.isfinite([sim_A, sim_f])):
        return np.nan, np.nan

    mu_a_div_zeta = (cfg["amplitude"]["mean"]/sim_A)*(cfg["amplitude"]["mean"]/sim_A) * theta_scaled['mu_a_div_zeta_0']
    tau = sim_f/cfg["frequency"]["mean"] #in seconds if f in hz

    return tau, mu_a_div_zeta

def compute_l2(sim_A, sim_f, tau, mu_a_div_zeta, mu_a_div_zeta_0):
    cfg = LIKELIHOOD_2_CONFIG

    if not all(np.isfinite([sim_A, sim_f, tau, mu_a_div_zeta])):
        return -np.inf

    # convert simulation → physical units
    f_phys = sim_f / tau
    A_phys = sim_A * np.sqrt(mu_a_div_zeta /mu_a_div_zeta_0)

    l = 0.0
    l += gaussian_loglik_optimal_sigma(f_phys, cfg["frequency"]["mean"],cfg["frequency"]["sigma"])
    l += gaussian_loglik_optimal_sigma(A_phys, cfg["amplitude"]["mean"],cfg["amplitude"]["sigma"])
    # print('f_phys', 'cfg["frequency"]["mean"]',f_phys, cfg["frequency"]["mean"], gaussian_loglik(f_phys, cfg["frequency"]["mean"], cfg["frequency"]["sigma"]))
    # print('A_phys', 'cfg["amplitude"]["mean"]',A_phys, cfg["amplitude"]["mean"], gaussian_loglik(A_phys, cfg["amplitude"]["mean"], cfg["amplitude"]["sigma"]))
    return l

def infer_biological_params(theta_scaled, B, mu_a_div_zeta, tau, dimension):
    L = BIO_CONSTANTS["L"]
    d = BIO_CONSTANTS["d"]
    rho = DENSITY_PER_GAP_2D if dimension == '2d' else DENSITY_PER_GAP_3D

    eta = theta_scaled["eta"]
    mu_a_times_zeta = theta_scaled["mu_a_times_zeta"]
    fstar = theta_scaled["fstar"]
    mu = theta_scaled["mu"]
    beta = theta_scaled["beta"]

    mu_a = np.sqrt(mu_a_times_zeta * mu_a_div_zeta)
    zeta = mu_a / mu_a_div_zeta

    # definitions (from paper)

    # mu_a = d ρ F0 L^2 / B
    F0 = mu_a * B / (d * rho * L*L)

    # zeta = d / (v0 τ)
    v0 = d / (zeta * tau)

    # f* = F0 / Fc
    Fc = F0 / fstar if fstar > 0 else np.nan

    # eta = π0 τ
    pi0 = eta / tau if tau > 0 else np.nan

    # tau = 1 / (π0 + ε0)
    eps0 = (1.0 / tau) - pi0 if tau > 0 else np.nan

    # mu = dd K LL/B
    K_d2 = mu * B/L/L

    v0_eps0 = v0 / eps0 if eps0 > 0 else np.nan

    b = beta*B*tau/d/L/L

    return {
        "B": B,
        "F0": F0,
        "Fc": Fc,
        "v0": v0,
        "pi0": pi0,
        "eps0": eps0,
        "K_d2": K_d2,
        "v0_eps0": v0_eps0,
        "b": b,
    }

def log_bio_prior(params):
    l = 0.0

    for key, prior in BIO_PRIORS.items():
        val = params.get(key)
        if val is None or val <= 0 or not np.isfinite(val):
            return -np.inf

        logv = np.log10(val)
        l += gaussian_loglik(logv, prior["mean_log10"], prior["sigma_log10"])
    return l



def optimize_B(theta_scaled, sim_A, sim_f, dimension):
    tau_init, mu_a_div_zeta_init = estimate_scaling(theta_scaled, sim_A, sim_f)
    # print(f"Initial scaling estimates: tau={tau_init:.4e}, mu_a/zeta={mu_a_div_zeta_init:.4e}")
    if not np.isfinite(tau_init):
        return np.nan, np.nan, np.nan
    mu_a_div_zeta_0 = theta_scaled['mu_a_div_zeta_0']
    # work in log-space
    x0 = np.array([
        np.log10(1000.0),               # B init
        np.log10(tau_init),
        np.log10(mu_a_div_zeta_init),
    ])

    def objective(x):
        logB, log_tau, log_mu_a_div_zeta = x

        B = 10 ** logB
        tau = 10 ** log_tau
        mu_a_div_zeta = 10 ** log_mu_a_div_zeta

        params = infer_biological_params(
            theta_scaled, B, mu_a_div_zeta, tau, dimension
        )

        l2 = compute_l2(sim_A, sim_f, tau, mu_a_div_zeta, mu_a_div_zeta_0)
        lp = log_bio_prior(params)

        val = l2 + lp
        # print('l2',l2,'lp',lp, 'val',val)
        # print('x',x)
        if not np.isfinite(val):
            return 1e6

        return -val

    #check this
    bounds = [
        (2, 4),    # B
        (np.log10(tau_init)-0.2, np.log10(tau_init)+0.2),   # tau
        (np.log10(mu_a_div_zeta_init)-0.2, np.log10(mu_a_div_zeta_init)+0.2),   # mu_a/zeta
    ]

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", options={
        "ftol": 1e-4,
        "gtol": 1e-4,
        "maxiter": 200,
    })
    if not res.success:
        print(f"Optimization failed: {res.message}")
        return np.nan, np.nan, np.nan

    logB, log_tau, log_mu_a_div_zeta = res.x
    # print('bounds and value',np.log10(mu_a_div_zeta_init)-1, np.log10(mu_a_div_zeta_init)+1,log_mu_a_div_zeta)
    # print(f"Optimization success: {res.message}")
    # print(f"Optimal log10(B): {logB:.4f}, log10(tau): {log_tau:.4f}, log10(mu_a/zeta): {log_mu_a_div_zeta:.4f}")
    # print(f"Optimal B: {10**logB:.4e}, tau: {10**log_tau:.4e}, mu_a/zeta: {10**log_mu_a_div_zeta:.4e}")
    # raise
    return (
        10 ** logB,
        10 ** log_tau,
        10 ** log_mu_a_div_zeta,
    )

def collect_scalar_dense_s2(dataset, conditions_df, out_csv, dimension):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        exp_path = os.path.join(dataset.path, exp_id)
        items = list(realization.data_items.values())

        out = {
            "experiment_id": exp_id,
            "oscillatory_reduced": np.nan,
            "oscillatory_full": np.nan,
            "amplitude_reduced": np.nan,
            "amplitude_full": np.nan,
            "lambda_reduced": np.nan,
            "lambda_full": np.nan,
            "f_reduced": np.nan,
            "f_full": np.nan,
        }

        def safe_resolve(item):
            try:
                return item.resolve(dataset, exp_path)
            except Exception:
                return None

        def dependency_hits_algorithm(item, algorithm_name):
            return depends_on_algorithm(realization, item, algorithm_name)

        def get_matching_items(data_key, algorithm_name):
            matched = []
            for item in items:
                if item.data_name.key != data_key:
                    continue
                if dependency_hits_algorithm(item, algorithm_name):
                    matched.append(item)
            return matched

        def extract_scalar_value(item):
            value = safe_resolve(item)
            if value is None:
                return np.nan
            if isinstance(value, dict):
                return float(value.get("metadata", np.nan))
            try:
                return float(value)
            except Exception:
                return np.nan

        def extract_first_valid_scalar(data_key, algorithm_name):
            vals = []
            for item in get_matching_items(data_key, algorithm_name):
                val = extract_scalar_value(item)
                if np.isfinite(val):
                    vals.append(val)
            if len(vals) == 0:
                return np.nan
            return float(np.nanmean(vals))

        def extract_first_valid_geyer_lambda(algorithm_name):
            vals = []
            for item in get_matching_items(GEYER.key, algorithm_name):
                value = safe_resolve(item)
                if not isinstance(value, dict):
                    continue
                lam = value.get("lambda", np.nan)
                L = value.get("L", np.nan)
                try:
                    lam = float(lam)
                    L = float(L)
                except Exception:
                    continue
                if np.isfinite(lam) and np.isfinite(L) and L != 0:
                    vals.append(lam / L)
            if len(vals) == 0:
                return np.nan
            return float(np.nanmean(vals))

        out["oscillatory_reduced"] = extract_first_valid_scalar(OSCILLATORY.key, "reduced")
        out["oscillatory_full"] = extract_first_valid_scalar(OSCILLATORY.key, "full")

        out["amplitude_reduced"] = extract_first_valid_scalar(A.key, "reduced")
        out["amplitude_full"] = extract_first_valid_scalar(A.key, "full")

        out["f_reduced"] = extract_first_valid_scalar(F.key, "reduced")
        out["f_full"] = extract_first_valid_scalar(F.key, "full")

        out["lambda_reduced"] = extract_first_valid_geyer_lambda("reduced")
        out["lambda_full"] = extract_first_valid_geyer_lambda("full")

        ampl_ratio = out["amplitude_reduced"] / out["amplitude_full"]  if np.isfinite(out["amplitude_reduced"]) else np.nan
        freq_ratio =  out["f_reduced"] / out["f_full"] if np.isfinite(out["f_reduced"]) else np.nan

        out["amplitude_ratio"] = ampl_ratio
        out["frequency_ratio"] = freq_ratio

        l1 = compute_l1(
            ampl_ratio,
            freq_ratio,
            out["lambda_full"],
            out["lambda_reduced"],
        )
        out["logL1"] = l1


        rows.append(out)

        # Unload cached data to prevent memory leak
        realization.unload_data()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    for i, row in df.iterrows():

        sim_A = np.sqrt(row["amplitude_full"] * row["amplitude_reduced"])
        sim_f = np.sqrt(row["f_full"] * row["f_reduced"])

        theta_scaled = {
            "mu": row["mu"],
            "eta": row["eta"],
            "mu_a_times_zeta": row["mu_a"] * row["zeta"],
            "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
            "zeta": row["zeta"],
            "fstar": row["fstar"],
            "beta": row["beta"],
        }

        B_opt, tau_opt, mu_a_div_zeta_opt = optimize_B(
            theta_scaled,
            sim_A,
            sim_f,
            dimension
        )

        df.loc[i, "tau_opt"] = tau_opt
        df.loc[i, "mu_a_div_zeta_opt"] = mu_a_div_zeta_opt
        df.loc[i, "B_opt"] = B_opt

        if np.isfinite(tau_opt):
            l2 = compute_l2(
                sim_A,
                sim_f,
                tau_opt,
                mu_a_div_zeta_opt,
                theta_scaled["mu_a_div_zeta_0"]
            )

            params = infer_biological_params(
                theta_scaled,
                B_opt,
                mu_a_div_zeta_opt,
                tau_opt, dimension
            )

            lp = log_bio_prior(params)
            df.loc[i, "log_posterior"] = row["logL1"] + l2 + lp
        else:
            df.loc[i, "log_posterior"] = -np.inf




    cols = [
        "experiment_id",
        "mu_a", "mu", "eta", "zeta", "beta", "fstar", "Nmotor", "strategy",
        "oscillatory_reduced", "oscillatory_full",
        "amplitude_reduced", "amplitude_full",
        "lambda_reduced", "lambda_full",
        "f_reduced", "f_full",
    ]
    cols += [
        "amplitude_ratio",
        "frequency_ratio",
        "logL1",
        "tau_opt",
        "mu_a_div_zeta_opt",
        "B_opt",
        "log_posterior",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]


    df.to_csv(out_csv, index=False)
    return df

def compute_l1_s3(
    ampl_ratio,
    freq_ratio,
    lam_norm,
    lam_pert,
    logQ_full,
    logQ_reduced,
):
    cfg = LIKELIHOOD_1_CONFIG
    qcfg = LIKELIHOOD_1_Q_CONFIG

    l = 0.0

    # wavelength (absolute)
    if np.isfinite(lam_norm):
        l += gaussian_loglik_optimal_sigma(lam_norm, cfg["wavelength"]["norm"], cfg["wavelength"]["sigma"])
    if np.isfinite(lam_pert):
        l += gaussian_loglik_optimal_sigma(lam_pert, cfg["wavelength"]["pert"], cfg["wavelength"]["sigma"])

    # ratios
    l += gaussian_loglik_optimal_sigma(ampl_ratio, cfg["amplitude_ratio"]["expected"], cfg["amplitude_ratio"]["sigma"])
    l += gaussian_loglik_optimal_sigma(freq_ratio, cfg["frequency_ratio"]["expected"], cfg["frequency_ratio"]["sigma"])

    # Q terms in log-space
    l += gaussian_loglik_optimal_sigma(logQ_full, qcfg["full"]["mean"], qcfg["full"]["sigma"])
    l += gaussian_loglik_optimal_sigma(logQ_reduced, qcfg["reduced"]["mean"], qcfg["reduced"]["sigma"])

    return l

def run_pipeline_s3(dataset: DataSet):
    print("\nRunning S3 pipeline")

    valid_realizations = []

    for realization in dataset:

        # -----------------------------
        # REDUCED SYSTEM
        # -----------------------------
        reduced_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == TANGENT_ANGLE_SERIES.key
            and item.algorithm == "reduced"
        ]

        if not reduced_ids:
            continue

        run_transformer_on_realization(
            dataset,
            realization,
            detect_oscillation_t,
            allowed_ids={"tangent_angle_series": reduced_ids},
        )

        osc_reduced = [
            item for item in realization.data_items.values()
            if item.data_name.key == OSCILLATORY.key
            and set(item.dependencies).intersection(reduced_ids)
        ]

        if not any(
            bool(item.resolve(dataset, os.path.join(dataset.path, realization.experiment_id)))
            for item in osc_reduced
        ):
            continue

        # -----------------------------
        # FULL SYSTEM
        # -----------------------------
        full_ids = [
            item.id for item in realization.data_items.values()
            if item.data_name.key == TANGENT_ANGLE_SERIES.key
            and item.algorithm == "full"
        ]

        if not full_ids:
            continue

        run_transformer_on_realization(
            dataset,
            realization,
            detect_oscillation_t,
            allowed_ids={"tangent_angle_series": full_ids},
        )

        osc_full = [
            item for item in realization.data_items.values()
            if item.data_name.key == OSCILLATORY.key
            and set(item.dependencies).intersection(full_ids)
        ]

        if not any(
            bool(item.resolve(dataset, os.path.join(dataset.path, realization.experiment_id)))
            for item in osc_full
        ):
            continue

        # -----------------------------
        # PASSED GATING
        # -----------------------------
        valid_realizations.append(realization.experiment_id)

        tangent_ids = reduced_ids + full_ids

        # --- SAME FEATURE SET AS S2 (for now) ---
        for tr in (estimate_amplitude_psd_t, estimate_f_from_tangent_angle_power_t):
            run_transformer_on_realization(
                dataset,
                realization,
                tr,
                allowed_ids={"tangent_angle_series": tangent_ids},
            )

        run_transformer_on_realization(
            dataset,
            realization,
            protophase_from_spatial_modes_t,
            allowed_ids={"tangent_angle_series": tangent_ids},
        )

        run_transformer_on_realization(
            dataset,
            realization,
            estimate_phase_from_protophase_t,
            allowed_algorithms=[protophase_from_spatial_modes_t.algorithm],
        )

        run_transformer_on_realization(
            dataset,
            realization,
            periodic_avg_t,
            allowed_algorithms_by_key={
                estimate_phase_from_protophase_t.outputs[0].key: [
                    estimate_phase_from_protophase_t.algorithm
                ],
            },
            match_ancestor_keys={TANGENT_ANGLE_SERIES.key},
        )

        run_transformer_on_realization(
            dataset,
            realization,
            get_geyer_fit_cycle_t,
        )

        run_transformer_on_realization(
            dataset,
            realization,
            estimate_Q_msd_fixed_tau_t,
            allowed_algorithms=[estimate_phase_from_protophase_t.algorithm]
        )

    print(f"S3 done: {len(valid_realizations)} valid realizations")

def collect_scalar_dense_s3(dataset, conditions_df, out_csv, dimension):
    rows = []

    for realization in dataset:
        exp_id = realization.experiment_id
        exp_path = os.path.join(dataset.path, exp_id)
        items = list(realization.data_items.values())

        out = {
            "experiment_id": exp_id,
            "oscillatory_reduced": np.nan,
            "oscillatory_full": np.nan,
            "amplitude_reduced": np.nan,
            "amplitude_full": np.nan,
            "lambda_reduced": np.nan,
            "lambda_full": np.nan,
            "f_reduced": np.nan,
            "f_full": np.nan,
            "Q_reduced": np.nan,
            "Q_full": np.nan,
            "logQ_reduced": np.nan,
            "logQ_full": np.nan,
        }

        def safe_resolve(item):
            try:
                return item.resolve(dataset, exp_path)
            except Exception:
                return None

        def dependency_hits_algorithm(item, algorithm_name):
            return depends_on_algorithm(realization, item, algorithm_name)

        def get_matching_items(data_key, algorithm_name):
            matched = []
            for item in items:
                if item.data_name.key != data_key:
                    continue
                if dependency_hits_algorithm(item, algorithm_name):
                    matched.append(item)
            return matched

        def extract_scalar_value(item):
            value = safe_resolve(item)
            if value is None:
                return np.nan

            if isinstance(value, dict):
                return float(value.get("metadata", np.nan))

            try:
                return float(value)
            except Exception:
                return np.nan

        def extract_first_valid_scalar(data_key, algorithm_name):
            vals = []
            for item in get_matching_items(data_key, algorithm_name):
                val = extract_scalar_value(item)
                if np.isfinite(val):
                    vals.append(val)
            if len(vals) == 0:
                return np.nan
            return float(np.nanmean(vals))

        def extract_first_valid_geyer_lambda(algorithm_name):
            vals = []
            for item in get_matching_items(GEYER.key, algorithm_name):
                value = safe_resolve(item)
                if not isinstance(value, dict):
                    continue

                lam = value.get("lambda", np.nan)
                L = value.get("L", np.nan)

                try:
                    lam = float(lam)
                    L = float(L)
                except Exception:
                    continue

                if np.isfinite(lam) and np.isfinite(L) and L != 0:
                    vals.append(lam / L)

            if len(vals) == 0:
                return np.nan
            return float(np.nanmean(vals))

        out["oscillatory_reduced"] = extract_first_valid_scalar(OSCILLATORY.key, "reduced")
        out["oscillatory_full"] = extract_first_valid_scalar(OSCILLATORY.key, "full")

        out["amplitude_reduced"] = extract_first_valid_scalar(A.key, "reduced")
        out["amplitude_full"] = extract_first_valid_scalar(A.key, "full")

        out["f_reduced"] = extract_first_valid_scalar(F.key, "reduced")
        out["f_full"] = extract_first_valid_scalar(F.key, "full")

        out["lambda_reduced"] = extract_first_valid_geyer_lambda("reduced")
        out["lambda_full"] = extract_first_valid_geyer_lambda("full")

        out["Q_reduced"] = extract_first_valid_scalar(Q.key, "reduced")
        out["Q_full"] = extract_first_valid_scalar(Q.key, "full")

        if np.isfinite(out["Q_reduced"]) and out["Q_reduced"] > 0:
            out["logQ_reduced"] = float(np.log10(out["Q_reduced"]))

        if np.isfinite(out["Q_full"]) and out["Q_full"] > 0:
            out["logQ_full"] = float(np.log10(out["Q_full"]))

        ampl_ratio = (
            out["amplitude_reduced"] / out["amplitude_full"]
            if np.isfinite(out["amplitude_reduced"]) and np.isfinite(out["amplitude_full"]) and out["amplitude_full"] != 0
            else np.nan
        )
        freq_ratio = (
            out["f_reduced"] / out["f_full"]
            if np.isfinite(out["f_reduced"]) and np.isfinite(out["f_full"]) and out["f_full"] != 0
            else np.nan
        )

        out["amplitude_ratio"] = ampl_ratio
        out["frequency_ratio"] = freq_ratio

        l1 = compute_l1_s3(
            ampl_ratio,
            freq_ratio,
            out["lambda_full"],
            out["lambda_reduced"],
            out["logQ_full"],
            out["logQ_reduced"],
        )
        out["logL1"] = l1

        rows.append(out)

        realization.unload_data()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["experiment_id"] = df["experiment_id"].astype(str)
    conditions_df["experiment_id"] = conditions_df["experiment_id"].astype(str)

    df = df.merge(conditions_df, on="experiment_id", how="left")

    for i, row in df.iterrows():
        sim_A = np.sqrt(row["amplitude_full"] * row["amplitude_reduced"])
        sim_f = np.sqrt(row["f_full"] * row["f_reduced"])

        theta_scaled = {
            "mu": row["mu"],
            "eta": row["eta"],
            "mu_a_times_zeta": row["mu_a"] * row["zeta"],
            "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
            "zeta": row["zeta"],
            "fstar": row["fstar"],
            "beta": row["beta"],
        }

        B_opt, tau_opt, mu_a_div_zeta_opt = optimize_B(
            theta_scaled,
            sim_A,
            sim_f, dimension
        )
        
        df.loc[i, "tau_opt"] = tau_opt
        df.loc[i, "mu_a_div_zeta_opt"] = mu_a_div_zeta_opt
        df.loc[i, "B_opt"] = B_opt

        if np.isfinite(tau_opt):
            l2 = compute_l2(
                sim_A,
                sim_f,
                tau_opt,
                mu_a_div_zeta_opt,
                theta_scaled["mu_a_div_zeta_0"],
            )

            params = infer_biological_params(
                theta_scaled,
                B_opt,
                mu_a_div_zeta_opt,
                tau_opt, dimension
            )

            lp = log_bio_prior(params)
            df.loc[i, "logL2"] = l2
            df.loc[i, "log_prior"] = lp
            df.loc[i, "log_posterior"] = row["logL1"] + l2 + lp
        else:
            df.loc[i, "log_posterior"] = -np.inf

    cols = [
        "experiment_id",
        "mu_a", "mu", "eta", "zeta", "beta", "fstar", "Nmotor", "strategy",
        "oscillatory_reduced", "oscillatory_full",
        "amplitude_reduced", "amplitude_full",
        "lambda_reduced", "lambda_full",
        "f_reduced", "f_full",
        "Q_reduced", "Q_full",
        "logQ_reduced", "logQ_full",
        "amplitude_ratio",
        "frequency_ratio",
        "logL1",
        "tau_opt",
        "mu_a_div_zeta_opt",
        "B_opt",
        "log_posterior",
        "logL2",
        "log_prior",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    return df
