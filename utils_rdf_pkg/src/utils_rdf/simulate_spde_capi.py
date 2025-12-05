import ctypes
import numpy as np
import os
# import multiprocessing as mp

# mp.set_start_method("spawn", force=True)

# ----------------------------------------------------------------------
# Load compiled library
# ----------------------------------------------------------------------
_here   = os.path.dirname(__file__)
_libpath = os.path.join(_here, "libspde_capi.so")

if not os.path.exists(_libpath):
    raise FileNotFoundError(f"Shared library not found at {_libpath}")

_lib = ctypes.CDLL(_libpath)


# ----------------------------------------------------------------------
# SimulationMode enum (must match C++ enum order)
#
# enum class SimulationMode {
#     Deterministic,          // 0
#     Poisson,                // 1
#     WhiteNoise,             // 2
#     ConstWhiteNoise,        // 3
#     ConstWhiteNoiseOpen,    // 4
#     ConstWhiteNoisePeriodic // 5
# };
# ----------------------------------------------------------------------
MODE_DETERMINISTIC          = 0
MODE_POISSON                = 1
MODE_WHITE_NOISE            = 2
MODE_CONST_WHITE_NOISE      = 3
MODE_CONST_WHITE_NOISE_OPEN = 4
MODE_CONST_WHITE_NOISE_PER  = 5

_MODE_NAME_TO_INT = {
    "deterministic": MODE_DETERMINISTIC,
    "det":           MODE_DETERMINISTIC,

    "poisson":       MODE_POISSON,
    "default":       MODE_POISSON,

    "white_noise":   MODE_WHITE_NOISE,
    "wn":            MODE_WHITE_NOISE,

    "const_white_noise":        MODE_CONST_WHITE_NOISE,
    "cwn":                      MODE_CONST_WHITE_NOISE,

    "const_white_noise_open":   MODE_CONST_WHITE_NOISE_OPEN,
    "cwn_open":                 MODE_CONST_WHITE_NOISE_OPEN,

    "const_white_noise_periodic": MODE_CONST_WHITE_NOISE_PER,
    "cwn_periodic":                MODE_CONST_WHITE_NOISE_PER,
}

# ----------------------------------------------------------------------
# Define parameter struct
#   MUST match C++ spde::Params in order and types
#
# struct Params {
#     int    n;
#     double fstar;
#     double mu;
#     double eta;
#     double zeta;
#     double beta;
#     double t_sub;
#     double T;
#     double dt;
#     double mu_a;
#     double Nmotor;
#     SimulationMode mode;
#     unsigned long long seed;
#     double ZETA_GAMMA_THRESHOLD;
#     double LAMBDA_SMALL_THRESHOLD;
# };
# ----------------------------------------------------------------------
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
        ("mu_a", ctypes.c_double),
        ("Nmotor", ctypes.c_double),
        ("mode", ctypes.c_int),            # <--- NEW FIELD
        ("seed", ctypes.c_ulonglong),
        ("ZETA_GAMMA_THRESHOLD", ctypes.c_double),
        ("LAMBDA_SMALL_THRESHOLD", ctypes.c_double),
    ]

# ----------------------------------------------------------------------
# Configure C function signature
# extern "C" int spde_simulate(const spde::Params* p,
#                              double* t_out,
#                              double* gamma_out,
#                              double* nplus_out,
#                              double* nminus_out);
# ----------------------------------------------------------------------
_lib.spde_simulate.argtypes = [
    ctypes.POINTER(SpdeParams),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
]
_lib.spde_simulate.restype = ctypes.c_int

# ----------------------------------------------------------------------
# Direct simulation call (main interface)
# ----------------------------------------------------------------------
def simulate_episode(params: dict, seed: int):
    """
    Run the SPDE simulation using the compiled C++ backend (direct call).

    Parameters
    ----------
    params : dict
        Must contain at least:
            n, Nmotor, fstar, mu, eta, zeta, beta, t_sub, T, dt, mu_a

        Optional:
            ZETA_GAMMA_THRESHOLD (default 13.0)
            LAMBDA_SMALL_THRESHOLD (default 1e-3)
            mode : str, one of
                "poisson" (default), "white_noise"/"wn",
                "const_white_noise"/"cwn",
                "const_white_noise_open"/"cwn_open",
                "const_white_noise_periodic"/"cwn_periodic",
                "deterministic"/"det"

            Note: if Nmotor < 0, the C++ code will *force* deterministic mode,
            regardless of `mode` here (mirrors the main executable).
    seed : int
        RNG seed.

    Returns
    -------
    t_coarse, gamma, nplus, nminus : np.ndarray
        Arrays truncated to the actually used number of coarse time steps
        (n_out returned by the C function).
    """

    n      = int(params["n"])
    Nmotor = float(params["Nmotor"])
    fstar  = float(params["fstar"])
    mu     = float(params["mu"])
    eta    = float(params["eta"])
    zeta   = float(params["zeta"])
    beta   = float(params["beta"])
    t_sub  = float(params["t_sub"])
    T      = float(params["T"])
    dt     = float(params["dt"])
    mu_a   = float(params["mu_a"])

    ZETA_GAMMA_THRESHOLD  = float(params.get("ZETA_GAMMA_THRESHOLD", 13.0))
    LAMBDA_SMALL_THRESHOLD = float(params.get("LAMBDA_SMALL_THRESHOLD", 1e-3))

    # Map optional mode string to enum int (default: Poisson)
    mode_str = str(params.get("mode", "poisson")).lower()
    mode_int = _MODE_NAME_TO_INT.get(mode_str, MODE_POISSON)

    cparams = SpdeParams(
        n=n,
        fstar=fstar,
        mu=mu,
        eta=eta,
        zeta=zeta,
        beta=beta,
        t_sub=t_sub,
        T=T,
        dt=dt,
        mu_a=mu_a,
        Nmotor=Nmotor,
        mode=mode_int,
        seed=int(seed),
        ZETA_GAMMA_THRESHOLD=ZETA_GAMMA_THRESHOLD,
        LAMBDA_SMALL_THRESHOLD=LAMBDA_SMALL_THRESHOLD,
    )

    # Pre-allocate based on expected sizes; C++ will fill & tell us n_out
    n_coarse = int(T * t_sub) + 1
    N_nodes  = n + 1

    t_coarse = np.zeros(n_coarse, dtype=np.float64)
    gamma    = np.zeros((n_coarse, N_nodes), dtype=np.float64)
    nplus    = np.zeros((n_coarse, N_nodes), dtype=np.float64)
    nminus   = np.zeros((n_coarse, N_nodes), dtype=np.float64)

    n_out = _lib.spde_simulate(
        ctypes.byref(cparams),
        t_coarse,
        gamma,
        nplus,
        nminus,
    )

    if n_out < 0:
        raise RuntimeError("spde_simulate() reported an error (return < 0)")

    # Truncate to the actually filled number of coarse steps
    n_out = int(n_out)
    return (
        t_coarse[:n_out],
        gamma[:n_out],
        nplus[:n_out],
        nminus[:n_out],
    )
