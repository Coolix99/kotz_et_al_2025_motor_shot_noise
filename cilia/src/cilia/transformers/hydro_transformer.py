import numpy as np

from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SEGMENTS,
    ARCLENGTH,
    DT_FRAME,
    PERIODIC_AVG, F
)
from cilia.datastructures.DataName import HYDRO_CONTRIBUTION
from cilia.transformers.Transformer import data_transformer

from cilia.physics.rft import (
    RFTParams,
    compute_hydro_elastic_balance,
)


@data_transformer(
    inputs=(TANGENT_ANGLE_SEGMENTS, DT_FRAME, ARCLENGTH),
    outputs=(HYDRO_CONTRIBUTION,),
)
def segment_hydrodynamics_t(
    tangent_angle_segments,
    DT_FRAME,
    s,
    *,
    B: float = 840.0, #pNmum^2
    d2K: float = 80.0, #pN
    xi_par: float = 0.7e-3, #pNsmum^-2
    xi_perp_ratio: float = 1.8,
):

    xi_perp = xi_perp_ratio * xi_par
    params = RFTParams(xi_perp=xi_perp, xi_par=xi_par)

    total_weight = 0.0

    sum_free = 0.0
    sum_fixed = 0.0
    sum_chlamy = 0.0
    sum_P_free = 0.0
    sum_P_fixed = 0.0
    sum_P_chlamy = 0.0

    for seg in tangent_angle_segments:

        theta = np.asarray(seg, dtype=float)

        if theta.ndim != 2 or theta.shape[0] < 10:
            continue

        Nt = theta.shape[0]

        res = compute_hydro_elastic_balance(
            theta=theta,
            s=s,
            dt=DT_FRAME,
            B=B,
            d2K=d2K,
            params=params,
        )

        mean_R_free   = float(res["mean_R_free"])
        mean_R_fixed  = float(res["mean_R_fixed"])
        mean_R_chlamy = float(res["mean_R_chlamy"])
        mean_P_free   = float(res["mean_power_free"])
        mean_P_fixed  = float(res["mean_power_fixed"])
        mean_P_chlamy = float(res["mean_power_chlamy"])

        # weight by segment duration
        weight = Nt * DT_FRAME

        sum_free   += weight * mean_R_free
        sum_fixed  += weight * mean_R_fixed
        sum_chlamy += weight * mean_R_chlamy
        sum_P_free   += weight * mean_P_free
        sum_P_fixed  += weight * mean_P_fixed
        sum_P_chlamy += weight * mean_P_chlamy

        total_weight += weight

    if total_weight == 0:
        return {
            HYDRO_CONTRIBUTION.key: {
                "mean_R_free": np.nan,
                "mean_R_fixed": np.nan,
                "mean_R_chlamy": np.nan,
                "mean_power_free": np.nan,
                "mean_power_fixed": np.nan,
                "mean_power_chlamy": np.nan,
            }
        }

    return {
        HYDRO_CONTRIBUTION.key: {
            "mean_R_free":   sum_free   / total_weight,
            "mean_R_fixed":  sum_fixed  / total_weight,
            "mean_R_chlamy": sum_chlamy / total_weight,
            "mean_power_free":   sum_P_free   / total_weight,
            "mean_power_fixed":  sum_P_fixed  / total_weight,
            "mean_power_chlamy": sum_P_chlamy / total_weight,
        }
    }

import numpy as np

from cilia.datastructures.DataName import (
    TANGENT_ANGLE_SERIES,
    ARCLENGTH,
    DT_FRAME,
)
from cilia.datastructures.DataName import HYDRO_CONTRIBUTION
from cilia.transformers.Transformer import data_transformer

from cilia.physics.rft import (
    RFTParams,
    compute_hydro_elastic_balance,
)


@data_transformer(
    inputs=(TANGENT_ANGLE_SERIES, DT_FRAME, ARCLENGTH),
    outputs=(HYDRO_CONTRIBUTION,),
)
def hydrodynamics_t(
    tangent_angle_series,
    DT_FRAME,
    s,
    *,
    B: float = 840.0,     # pN µm²
    d2K: float = 80.0,    # pN
    xi_par: float = 0.7e-3,  # pN s µm⁻²
    xi_perp_ratio: float = 1.8,
):
    """
    Hydrodynamics contribution computed on full time series
    (no segmentation).
    """

    theta = np.asarray(tangent_angle_series, dtype=float)

    # sanity check (same as segment version)
    if theta.ndim != 2 or theta.shape[0] < 10:
        return {
            HYDRO_CONTRIBUTION.key: {
                "mean_R_free": np.nan,
                "mean_R_fixed": np.nan,
                "mean_R_chlamy": np.nan,
            }
        }

    # parameters
    xi_perp = xi_perp_ratio * xi_par
    params = RFTParams(xi_perp=xi_perp, xi_par=xi_par)

    # compute once on full series
    res = compute_hydro_elastic_balance(
        theta=theta,
        s=s,
        dt=DT_FRAME,
        B=B,
        d2K=d2K,
        params=params,
    )

    return {
        HYDRO_CONTRIBUTION.key: {
            "mean_R_free": float(res["mean_R_free"]),
            "mean_R_fixed": float(res["mean_R_fixed"]),
            "mean_R_chlamy": float(res["mean_R_chlamy"]),
        }
    }


@data_transformer(
    inputs=(PERIODIC_AVG, F, ARCLENGTH),
    outputs=(HYDRO_CONTRIBUTION,),
)
def avg_hydrodynamics_t(
    periodic_avg_dict,
    f,
    s,
    *,
    B: float = 840.0, #pNmum^2
    d2K: float = 80.0, #pN
    xi_par: float = 0.7e-3, #pNsmum^-2
    xi_perp_ratio: float = 1.8,
):
    psi_avg = periodic_avg_dict["mean"]
    xi_perp = xi_perp_ratio * xi_par
    params = RFTParams(xi_perp=xi_perp, xi_par=xi_par)

    if psi_avg.ndim != 2 or psi_avg.shape[0] < 10:
        raise

    Nt = psi_avg.shape[0]
    dt = 1.0 / (f * Nt) *4e-3
    s=s*10
    print('WARNING BUGGY HACK FOR SIMULATION')
    res = compute_hydro_elastic_balance(
        theta=psi_avg,
        s=s,
        dt=dt,
        B=B,
        d2K=d2K,
        params=params,
    )

    mean_R_free   = float(res["mean_R_free"])
    mean_R_fixed  = float(res["mean_R_fixed"])
    mean_R_chlamy = float(res["mean_R_chlamy"])

    

    return {
        HYDRO_CONTRIBUTION.key: {
            "mean_R_free": mean_R_free,
            "mean_R_fixed": mean_R_fixed,
            "mean_R_chlamy": mean_R_chlamy,
        }
    }

