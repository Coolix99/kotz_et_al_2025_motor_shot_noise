import numpy as np

from cilia.datastructures.DataName import (
    CONDITION_DESCRIPTION,
    PERIODIC_AVG,
    F,
    TORQUE_POWER,
)
from cilia.transformers.Transformer import data_transformer
from cilia.physics.torque_power import compute_torque_power_bounds


@data_transformer(
    inputs=(PERIODIC_AVG, F, CONDITION_DESCRIPTION),
    outputs=(TORQUE_POWER,),
)
def avg_torque_power_t(
    periodic_avg_dict,
    f,
    description,
    *,
    B: float = 840.0,   # pN µm²
    d2K: float = 80.0,  # pN
    d: float = 0.2,     # µm = 200 nm
    di: np.ndarray = np.array([0.220, 0.337, 0.296, 0.117]),
):
    psi_avg = periodic_avg_dict["mean"]
    #print(description)

    kcl_to_relN_WT = {
        0: 1.00,
        50: 0.96,
        100: 0.93,
        200: 0.87,
        300: 0.80,
        400: 0.74,
    }

    N_MOTORS = {
        "WT": 17500,
        "ODA": 7500,
    }

    # -----------------------------------------
    # Extract genotype and KCl concentration
    # -----------------------------------------
    sexp = description["sexp"]          # e.g. "WT_KCl"
    genotype = sexp.split("_")[0]       # "WT" or "ODA"
    KCl = description["KCl"]            # mM

    # -----------------------------------------
    # Determine number of motors
    # -----------------------------------------
    Nmotor = N_MOTORS[genotype]

    relN = np.interp(
        KCl,
        np.array(sorted(kcl_to_relN_WT.keys()), dtype=float),
        np.array(
            [kcl_to_relN_WT[k] for k in sorted(kcl_to_relN_WT.keys())],
            dtype=float,
        ),
    )

    Nmotor *= relN

    # motors / µm / gap
    rho_gap = Nmotor / 9.0 / 10.0

    # -----------------------------------------
    # Validate input waveform
    # -----------------------------------------
    if psi_avg.ndim != 2 or psi_avg.shape[0] < 10:
        raise ValueError(
            "periodic_avg_dict['mean'] must have shape (Nt, Ns) with Nt >= 10."
        )

    Nt = psi_avg.shape[0]
    dt = 1.0 / (f * Nt)
    s = np.linspace(0.0, 10.0, psi_avg.shape[1])  # µm

    # -----------------------------------------
    # Run analysis
    # -----------------------------------------
    result = compute_torque_power_bounds(
        theta=psi_avg,
        s=s,
        dt=dt,
        B=B,
        d2K=d2K,
        d=d,
        di=di,
        rho_gap=rho_gap,
        debug_plot=False
    )

    # -----------------------------------------
    # Return full result structure
    # -----------------------------------------
    return {
        TORQUE_POWER.key: result
    }