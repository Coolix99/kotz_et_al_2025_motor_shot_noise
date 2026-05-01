#!/usr/bin/env python3

import ctypes
import os
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import json
import sys

from cilia.datasets.DataSet import DataSet

# -----------------------------------------------------------------------------
# ENUM
# -----------------------------------------------------------------------------
class SimulationMode(int):
    Deterministic = 0
    Poisson = 1
    WhiteNoise = 2
    ConstWhiteNoise = 3
    ConstWhiteNoiseOpen = 4
    ConstWhiteNoisePeriodic = 5
    CM_PS = 6
    ThreeD_Deterministic = 7
    ThreeD_Poisson = 8


# -----------------------------------------------------------------------------
# PARAM STRUCT
# -----------------------------------------------------------------------------
class SpdeParams(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("fstar", ctypes.c_double),
        ("mu", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("zeta", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("t_sub", ctypes.c_double),
        ("T", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("ZETA_GAMMA_THRESHOLD", ctypes.c_double),
        ("seed", ctypes.c_ulonglong),
        ("Nmotor", ctypes.c_double),
        ("LAMBDA_SMALL_THRESHOLD", ctypes.c_double),
        ("ps_rate", ctypes.c_double),
        ("d_tilde_1", ctypes.c_double),
        ("d_tilde_2", ctypes.c_double),
        ("d_tilde_3", ctypes.c_double),
        ("d_tilde_4", ctypes.c_double),
        ("mode", ctypes.c_int),
        ("mu_a", ctypes.c_double),
    ]


# -----------------------------------------------------------------------------
# C++ API loader (shared)
# -----------------------------------------------------------------------------
def load_c_api(project_root):
    LIB_PATH = project_root / "01_cpp_Sim" / "build" / "libspde_capi.so"
    LIB_PATH = LIB_PATH.resolve()
    if not LIB_PATH.exists():
        raise FileNotFoundError(
            f"Shared library not found at {LIB_PATH}. Build the C++ project first."
        )

    lib = ctypes.CDLL(str(LIB_PATH))

    lib.spde_simulate.argtypes = [
        ctypes.POINTER(SpdeParams),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ]
    lib.spde_simulate.restype = ctypes.c_int
    lib.spde_simulate_cmps.restype = ctypes.c_int

    return lib


# -----------------------------------------------------------------------------
# PRIOR
# -----------------------------------------------------------------------------
def sample_prior(rng, prior_params, fixed_params):
    mu = rng.lognormal(
        np.log(prior_params["mean_mu"]),
        prior_params["log_std_mu"]
    )

    eta = rng.beta(
        prior_params["alpha_eta"],
        prior_params["beta_eta"]
    )

    mu_a_times_zeta = rng.lognormal(
        np.log(prior_params["mean_mu_a_times_zeta"]),
        prior_params["log_std_mu_a_times_zeta"]
    )

    zeta = np.sqrt(mu_a_times_zeta / fixed_params["mu_a_div_zeta"])
    mu_a = fixed_params["mu_a_div_zeta"] * zeta

    fstar = rng.lognormal(
        np.log(prior_params["mean_fstar"]),
        prior_params["log_std_fstar"]
    )

    beta = rng.lognormal(
        np.log(prior_params["mean_beta"]),
        prior_params["log_std_beta"]
    )

    return {
        "mu": float(mu),
        "eta": float(eta),
        "zeta": float(zeta),
        "mu_a": float(mu_a),
        "fstar": float(fstar),
        "beta": float(beta),
    }


# -----------------------------------------------------------------------------
# PARAM BUILDING
# -----------------------------------------------------------------------------
def build_spde_params(sample, seed, fixed_params, fixed_params_d_tilde, dimension, T, Nmotor, mu_a, mode_suffix="Poisson"):
    if dimension == "3d":
        d1, d2, d3, d4 = (
            fixed_params_d_tilde["d_tilde_1"],
            fixed_params_d_tilde["d_tilde_2"],
            fixed_params_d_tilde["d_tilde_3"],
            fixed_params_d_tilde["d_tilde_4"]
        )
        mode = getattr(SimulationMode, f"ThreeD_{mode_suffix}")
    else:  # 2d
        d1, d2, d3, d4 = -1.0, -1.0, -1.0, -1.0
        mode = getattr(SimulationMode, mode_suffix)

    return SpdeParams(
        int(fixed_params["n"]),
        sample["fstar"],
        sample["mu"],
        sample["eta"],
        sample["zeta"],
        sample["beta"],
        fixed_params["t_sub"],
        T,
        fixed_params["dt"],
        fixed_params["ZETA_GAMMA_THRESHOLD"],
        int(seed),
        float(Nmotor),
        fixed_params["LAMBDA_SMALL_THRESHOLD"],
        -1.0,
        d1,
        d2,
        d3,
        d4,
        int(mode),
        float(mu_a),
    )


# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------
def run_simulation(lib, params):
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
        return None

    # --- reshape into (time, component, space) ---
    nplus = nplus_flat.reshape(n_coarse, 4, N)
    nminus = nminus_flat.reshape(n_coarse, 4, N)

    return t, gamma, nplus, nminus


# -----------------------------------------------------------------------------
# DATASET CLASS
# -----------------------------------------------------------------------------
class SimulationDataset(DataSet):
    def __init__(self, path: str):
        super().__init__(path)

    def _initialize_realizations(self):
        if not os.path.isdir(self.path):
            return

        exp_dirs = [
            d for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        from cilia.datasets.Realization import Realization
        self.realizations = [Realization(exp_id) for exp_id in sorted(exp_dirs)]

    def add_realization(self, realization):
        self.realizations.append(realization)





# -----------------------------------------------------------------------------
# CLI HELPERS
# -----------------------------------------------------------------------------
def add_common_args(parser):
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
    parser.add_argument("--seed", type=int, default=42)


def setup_project_paths():
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

    return PROJECT_ROOT, CILIA_FOLDER