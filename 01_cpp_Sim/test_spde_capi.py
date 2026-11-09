import ctypes
import numpy as np
import os
from enum import IntEnum

# ---------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------
LIB_PATH = os.path.abspath("01_cpp_Sim/build/libspde_capi.so")
lib = ctypes.CDLL(LIB_PATH)

# ---------------------------------------------------------------------
# SimulationMode enum (MUST match C++)
# ---------------------------------------------------------------------
class SimulationMode(IntEnum):
    Deterministic = 0
    Poisson = 1
    WhiteNoise = 2
    ConstWhiteNoise = 3
    ConstWhiteNoiseOpen = 4
    ConstWhiteNoisePeriodic = 5
    CM_PS = 6
    ThreeD_Deterministic = 7
    ThreeD_Poisson = 8

# ---------------------------------------------------------------------
# Base Params (ParamsBase / Params)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# CM+PS Params (extends base)
# ---------------------------------------------------------------------
class SpdeParamsCMPS(ctypes.Structure):
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
        ("ps_rate", ctypes.c_double),
        ("LAMBDA_SMALL_THRESHOLD", ctypes.c_double),
        ("mu_a", ctypes.c_double),
        ("rest_rate", ctypes.c_double),
        ("mode", ctypes.c_int),
    ]


# ---------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------
lib.spde_simulate.argtypes = [
    ctypes.POINTER(SpdeParams),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
lib.spde_simulate.restype = ctypes.c_int

lib.spde_simulate_cmps.argtypes = [
    ctypes.POINTER(SpdeParamsCMPS),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
]
lib.spde_simulate_cmps.restype = ctypes.c_int

# ---------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------
def run_simulation(params, ps_rate=None, rest_rate=None):
    """
    params  : SpdeParams
    ps_rate : None or float (activates CM_PS)
    rest_rate : None or float (activates CM_PS)
    """
    n_coarse = int(params.T * params.t_sub) + 1
    N = params.n + 1

    is_3d = params.mode in (SimulationMode.ThreeD_Deterministic, SimulationMode.ThreeD_Poisson)

    t = np.zeros(n_coarse)
    gamma = np.zeros((n_coarse, N))
    if is_3d:
        nplus = np.zeros((n_coarse, 4, N))
        nminus = np.zeros((n_coarse, 4, N))
    else:
        nplus = np.zeros((n_coarse, N))
        nminus = np.zeros((n_coarse, N))
    
    if ps_rate is None:
        ret = lib.spde_simulate(
            ctypes.byref(params),
            t, gamma, nplus, nminus
        )
    else:
        if rest_rate is None:
            raise ValueError("CM_PS requires rest_rate")
        pcmps = SpdeParamsCMPS(
            params.n,
            params.fstar,
            params.mu,
            params.eta,
            params.zeta,
            params.beta,
            params.t_sub,
            params.T,
            params.dt,
            params.ZETA_GAMMA_THRESHOLD,
            params.seed,
            params.Nmotor,
            ps_rate,
            params.LAMBDA_SMALL_THRESHOLD,
            params.mu_a,
            rest_rate,
            SimulationMode.CM_PS,
        )

        ret = lib.spde_simulate_cmps(
            ctypes.byref(pcmps),
            t, gamma, nplus, nminus
        )

    if ret < 0:
        raise RuntimeError("Simulation failed")

    return t, gamma, nplus, nminus

# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------
def main():
    base = SpdeParams(
        n=100,
        fstar=2.0,
        mu=10.0,
        eta=0.096,
        zeta=0.96,
        beta=2.0,
        t_sub=20.0,
        T=10.0,
        dt=1e-4,
        mu_a=1000.0,
        Nmotor=1000.0,
        mode=SimulationMode.Poisson,
        seed=42,
        ZETA_GAMMA_THRESHOLD=13.0,
        LAMBDA_SMALL_THRESHOLD=1e-3,
        d_tilde_1=0.44,
        d_tilde_2=0.67,
        d_tilde_3=0.59,
        d_tilde_4=0.23,
    )

    print("=== Poisson ===")
    t, g, _, _ = run_simulation(base)
    print("gamma[-1, :5] =", g[-1, :5])

    print("\n=== Deterministic (Nmotor < 0) ===")
    base.Nmotor = -1.0
    t, g, _, _ = run_simulation(base)
    print("gamma[-1, :5] =", g[-1, :5])

    print("\n=== CM+PS ===")
    base.Nmotor = 1000.0
    base.mode = SimulationMode.CM_PS
    t, g, _, _ = run_simulation(base, ps_rate=1.0, rest_rate=1.0)
    print("gamma[-1, :5] =", g[-1, :5])

    print("\n=== CM+PS → Poisson fallback (ps_rate < 0) ===")
    t, g, _, _ = run_simulation(base, ps_rate=-1.0, rest_rate=1.0)
    print("gamma[-1, :5] =", g[-1, :5])

    print("\n=== 3D Poisson ===")
    base.mode = SimulationMode.ThreeD_Poisson
    t, g, np, nm = run_simulation(base)
    print("gamma[-1, :5] =", g[-1, :5])
    print("nplus.shape =", np.shape)
    print("nminus.shape =", nm.shape)

    print("\n=== 3D Deterministic ===")
    base.mode = SimulationMode.ThreeD_Deterministic
    t, g, np, nm = run_simulation(base)
    print("gamma[-1, :5] =", g[-1, :5])
    print("nplus.shape =", np.shape)
    print("nminus.shape =", nm.shape)

if __name__ == "__main__":
    main()
