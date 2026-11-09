from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import   Q #ML_PHASE_DIFF

from cilia.algorithms.q_estimation import (
    estimate_Q_msd_fixed_tau, estimate_Q_msd_auto_interval, 
    # estimate_Q_msd_bridge_corrected_full_range,
    # estimate_Q_from_exp_phase_correlation, estimate_Q_from_fft_peak, estimate_Q_from_tangent_angle_lorentz_width, estimate_ou_phase_ml,estimate_phase_diffusion_mle
)



from cilia.datastructures.DataName import GLOBAL_PHASE
import numpy as np


@data_transformer(
    inputs=(GLOBAL_PHASE,),
    outputs=(Q,),
)
def segment_Q_msd_fixed_tau_t(phase_segments):
    qs = []
    weights = []
    for seg_data in phase_segments:
        seg = np.asarray(seg_data['phase'], dtype=float)
        if seg.ndim != 1 or len(seg) < 5:
            continue
        T_i = len(seg)
        q_i, D, _ = estimate_Q_msd_fixed_tau(seg, debug_plot=False)
        qs.append(q_i)
        weights.append(T_i)

    if len(qs) == 0:
        return {Q.key: np.nan}

    qs = np.asarray(qs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    q_weighted = np.sum(weights * qs) / np.sum(weights)

    return {
        Q.key: float(q_weighted)
    }

@data_transformer(
    inputs=(GLOBAL_PHASE,),
    outputs=(Q,),
)
def segment_Q_msd_auto_interval_t(phase_segments):
    qs = []
    weights = []

    for seg_data in phase_segments:
        seg = np.asarray(seg_data['phase'], dtype=float)
        if seg.ndim != 1 or len(seg) < 5:
            continue
        T_i = len(seg)
        q_i, D, _ = estimate_Q_msd_auto_interval(seg, debug_plot=False)
        qs.append(q_i)
        weights.append(T_i)

    if len(qs) == 0:
        return {Q.key: np.nan}

    qs = np.asarray(qs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    q_weighted = np.sum(weights * qs) / np.sum(weights)

    return {
        Q.key: float(q_weighted)
    }
