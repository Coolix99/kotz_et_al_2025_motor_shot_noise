from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import GLOBAL_PHASE, F, DT_FRAME
from cilia.datastructures.DataName import TANGENT_ANGLE
import numpy as np
from cilia.algorithms.f_estimation import (
    estimate_f_endpoint,
    estimate_f_from_tangent_angle_power,
)
from cilia.datastructures.DataName import GLOBAL_PHASE

@data_transformer(
    inputs=(TANGENT_ANGLE, DT_FRAME),
    outputs=(F,),
)
def segment_f_from_tangent_angle_power_t(tangent_angle_segments, dt_frame):
    freqs = []
    weights = []

    for seg in tangent_angle_segments:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[0] < 30:
            continue 
        T_i = seg.shape[0]
        f_i = estimate_f_from_tangent_angle_power(seg, dt_frame)
        freqs.append(f_i)
        weights.append(T_i)
    if len(freqs) == 0:
        return {F.key: np.nan}
    freqs = np.array(freqs, dtype=float)
    weights = np.array(weights, dtype=float)
    f_weighted = np.sum(weights * freqs) / np.sum(weights)

    return {
        F.key: float(f_weighted)
    }

@data_transformer(
    inputs=(GLOBAL_PHASE, DT_FRAME),
    outputs=(F,),
)
def segment_f_phase_endpoints_t(phase_segments, dt_frame):
    freqs = []
    weights = []

    for seg_data in phase_segments:
        seg = np.asarray(seg_data['phase'], dtype=float)
        if seg.ndim != 1 or len(seg) < 30:
            continue
        T_i = len(seg)
        f_i = estimate_f_endpoint(seg, dt_frame)
        freqs.append(f_i)
        weights.append(T_i)

    if len(freqs) == 0:
        return {F.key: np.nan}

    freqs = np.asarray(freqs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    f_weighted = np.sum(weights * freqs) / np.sum(weights)

    return {
        F.key: float(f_weighted)
    }
