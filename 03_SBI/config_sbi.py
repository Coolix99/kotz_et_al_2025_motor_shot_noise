# ---- Prior definitions ----
mean_mu = 10
log_std_mu = 1.0
alpha_eta = 1.1
beta_eta = 1.1
mean_mu_a_times_zeta = 1000
log_std_mu_a_times_zeta = 1.5

mean_fstar = 2
log_std_fstar = 0.5
mean_beta = 2
log_std_beta = 1.0

# Fixed simulation parameters
fixed_params = {
    "n": 100,
    "t_sub": 10,        
    "dt": 1e-4,
    "Nmotor": 85,
    "mu_a_div_zeta": 1600 # scales the amplitude
}

parameter_keys = ["mu", "eta", "mu_a","zeta" , "fstar", "beta"]
feature_keys = ["mu", "eta", "mu_a_times_zeta" , "fstar", "beta"]
log_keys = ["mu", "mu_a_times_zeta", "fstar", "beta"]  # log space

UPPER_SIGMA_BOUND = 4.0

# S1 S2
T_S1 = 40.0 
T_S2 = 60.0

rel_motor_extraction=0.74
expected_wavelength_norm = 2.1
expected_wavelength_pert = 1.8
sigma_wavelength = 0.1

expected_amplitude_ratio = 0.48
sigma_amplitude_ratio = 0.1

expected_freq_ratio = 0.77
sigma_freq_ratio = 0.5

#S3 
T_S3 = 500
logQ_min_norm = 1.8
logQ_min_pert = 0.8
sigma_logQ    = 0.2
