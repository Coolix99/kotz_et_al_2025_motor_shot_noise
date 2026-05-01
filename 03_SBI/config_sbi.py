import numpy as np

prior_params = {
    "mean_mu": 10,
    "log_std_mu": 1.0,

    "alpha_eta": 1.1,
    "beta_eta": 1.1,

    "mean_mu_a_times_zeta": 4000,
    "log_std_mu_a_times_zeta": 1.5,

    "mean_fstar": 2,
    "log_std_fstar": 0.5,

    "mean_beta": 2,
    "log_std_beta": 1.0,
}

fixed_params_d_tilde = {
    "d_tilde_1": 0.220,
    "d_tilde_2": 0.337,
    "d_tilde_3": 0.296,
    "d_tilde_4": 0.117,
}
N_MOTOR_2D = 85
N_Motor_3D = 20

fixed_params = {
    "n": 100,
    "t_sub": 10,
    "dt": 1e-4,
    
    "mu_a_div_zeta": 1600,

    "ZETA_GAMMA_THRESHOLD": 13.0,
    "LAMBDA_SMALL_THRESHOLD": 1e-3,
}

parameter_keys = ["mu", "eta", "mu_a","zeta" , "fstar", "beta"]
feature_keys = ["mu", "eta", "mu_a_times_zeta" , "fstar", "beta"]
log_keys = ["mu", "mu_a_times_zeta", "fstar", "beta"]  # which are in log space

UPPER_SIGMA_BOUND = 4.0


T_S1 = 40.0 
T_S2 = 60.0
REL_MOTOR_EXTRACTION = 0.74
LIKELIHOOD_1_CONFIG = {
    "wavelength": {
        "norm": 2.251,
        "pert": 1.949,
        "sigma": 0.071,
    },
    "amplitude_ratio": {
        "expected": 0.467,
        "sigma": 0.056,
    },
    "frequency_ratio": {
        "expected": 0.819,
        "sigma": 0.054,
    },
}

LIKELIHOOD_2_CONFIG = {
    "frequency": {
        "mean": 62.16,
        "sigma": 2.04,
    },
    "amplitude": {
        "mean": 0.612,
        "sigma": 0.036,
    },
}
# Biological priors (log10-space Gaussians)
BIO_CONSTANTS = {
    "L": 10.0,     # um (typical Chlamy ~10–12 µm)
    "d": 0.2,      # um (axoneme diameter scale)    
}
DENSITY_PER_GAP_2D = 1000.0  # motors / µm (effective density)
DENSITY_PER_GAP_3D = 200.0  # motors / µm (effective density)

BIO_PRIORS = {
    "K_d2": {
        "mean_log10": np.log10(80.0),
        "sigma_log10": 0.06,
        "unit": "pN",
    },
    "B": {
        "mean_log10": np.log10(1000.0),
        "sigma_log10": 0.15,
        "unit": "pN um^2",
    },
    "F0": {
        "mean_log10": np.log10(3.0),
        "sigma_log10": 0.5,#0.25,
        "unit": "pN",
    },
    "Fc": {
        "mean_log10": np.log10(1.0),
        "sigma_log10": 1.0,#0.5,
        "unit": "pN",
    },
    "v0": {
        "mean_log10": np.log10(5.0),
        "sigma_log10": 0.4,
        "unit": "um/s",
    },
    "v0_eps0": { #vo/epsilon in mum
        "mean_log10": np.log10(0.03),
        "sigma_log10": 0.75,
        "unit": "um",
    },
    "pi0": {
        "mean_log10": np.log10(1000),
        "sigma_log10": 1.0,
        "unit": "s^-1",
    },
    "b": {
        "mean_log10": np.log10(1.0),
        "sigma_log10": 2.0,
        "unit": "pN s um^-1",
    },
}

T_S3 = 500
LIKELIHOOD_1_Q_CONFIG = {
    "full": {
        "mean": 2.058,
        "sigma": 0.044,
    },
    "reduced": {
        "mean": 1.130,
        "sigma": 0.136,
    },
}
