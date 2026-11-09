from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import TANGENT_ANGLE_SERIES, LAM, ARCLENGTH, PERIODIC_AVG
from cilia.algorithms.l_estimation import estimate_lambda_from_tangent_angle_series, estimate_lambda_from_cycle

@data_transformer(
    inputs=(TANGENT_ANGLE_SERIES,ARCLENGTH),
    outputs=(LAM,),
)
def estimate_lambda_from_tangent_angle_series_t(tangent_angles, arclength):
    lam = estimate_lambda_from_tangent_angle_series(
        tangent_angles, arclength
    )
    return {
        LAM.key: lam
    }

@data_transformer(
    inputs=(PERIODIC_AVG, ARCLENGTH),
    outputs=(LAM,),
)
def get_lambda_cycle_t(periodic_avg_dict, arclength):
    psi_avg = periodic_avg_dict["mean"]
    uncertainty = periodic_avg_dict["uncertainty"]
    lam = estimate_lambda_from_cycle(
        psi_avg,uncertainty,
        arclength,
    )
    return {
        LAM.key: lam
    }
