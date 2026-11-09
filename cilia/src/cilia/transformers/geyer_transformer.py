from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import  GEYER, PERIODIC_AVG
from cilia.algorithms.geyer_fit import geyer_fit_from_tangent_angle_series, geyer_fit_from_cycle



@data_transformer(
    inputs=(PERIODIC_AVG, ),
    outputs=(GEYER,),
)
def get_geyer_fit_cycle_t(periodic_avg_dict, ):
    psi_avg = periodic_avg_dict["mean"]
    uncertainty = periodic_avg_dict["uncertainty"]
    geyer_fit_res = geyer_fit_from_cycle(
        psi_avg=psi_avg,
        uncertainty=uncertainty,
    )
    return {
        GEYER.key: geyer_fit_res
    }