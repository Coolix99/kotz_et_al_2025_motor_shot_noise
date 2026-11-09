from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import  A, PERIODIC_AVG,TANGENT_ANGLE
import numpy as np
from cilia.algorithms.a_estimation import (
    estimate_amplitude_mean_square,
    estimate_amplitude_psd,
    amplitude_first_harmonic_cycle,
    amplitude_mean_square_cycle
)




@data_transformer(
    inputs=(PERIODIC_AVG,),
    outputs=(A,),
)
def get_a_psd_cycle_t(periodic_avg_dict, ):
    psi_avg = periodic_avg_dict["mean"]
    A_cycle = amplitude_first_harmonic_cycle(psi_avg, s)

    return {
        A.key: A_cycle
    }

@data_transformer(
    inputs=(PERIODIC_AVG, ),
    outputs=(A,),
)
def get_a_ms_cycle_t(periodic_avg_dict,):
    psi_avg = periodic_avg_dict["mean"]
    A_cycle = amplitude_mean_square_cycle(psi_avg,s)

    return {
        A.key: A_cycle
    }


@data_transformer(
    inputs=(TANGENT_ANGLE, ),
    outputs=(A,),
)
def segment_amplitude_psd_t(tangent_angle_segments):
    amplitudes = []
    weights = []
    for seg in tangent_angle_segments:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue  
        T_i = seg.shape[0]
        a_i = estimate_amplitude_psd(seg)
        if not np.isfinite(a_i):
            continue
        amplitudes.append(a_i)
        weights.append(T_i)
    if len(amplitudes) == 0:
        return {A.key: np.nan}

    amplitudes = np.array(amplitudes, dtype=float)
    weights = np.array(weights, dtype=float)
    A_weighted = np.sum(weights * amplitudes) / np.sum(weights)

    return {
        A.key: float(A_weighted)
    }