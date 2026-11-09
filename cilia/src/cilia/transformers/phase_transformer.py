import numpy as np
from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import    TANGENT_ANGLE, GLOBAL_PHASE

from cilia.algorithms.phase_estimation import (
    estimate_phase_from_protophase, protophase_from_spatial_modes, 
    protophase_from_first_mode_hilbert
)



@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(GLOBAL_PHASE,),
)
def protophase_from_spatial_modes_segment_t(tangent_angle_segments):
    # --- convert to arrays and store lengths ---
    segments = [np.asarray(seg, dtype=float) for seg in tangent_angle_segments]
    lengths = [seg.shape[0] for seg in segments]

    if len(segments) == 0:
        return {GLOBAL_PHASE.key: []}

    # --- stack along time ---
    Y_full = np.vstack(segments)  # shape (sum_T, Ns)
    # --- run standard algorithm ---
    result = protophase_from_spatial_modes(Y_full)
    protophi_full = result['protophase']
    rel_amplitude_full = result['rel_amplitude']
    percentile_1 = result['percentile_1']
    variance_explained = result['variance_explained']
    
    # --- split back into segments ---
    phase_segments = []
    idx = 0
    for L in lengths:
        phase_segments.append({
            'phase': protophi_full[idx:idx+L],
            'rel_amplitude': rel_amplitude_full[idx:idx+L],
            'percentile_1': percentile_1,
            'variance_explained': variance_explained,
        })
        idx += L

    return {
        GLOBAL_PHASE.key: phase_segments
    }

@data_transformer(
    inputs=(GLOBAL_PHASE,),
    outputs=(GLOBAL_PHASE,),
)
def segment_phase_from_protophase_t(phase_segments):
    if len(phase_segments) == 0:
        return {GLOBAL_PHASE.key: []}

    corrected_segments = []

    for seg_data in phase_segments:
        phi = np.asarray(seg_data['phase'], dtype=float)

        if phi.size == 0:
            corrected_segments.append({
                'phase': phi,
                'rel_amplitude': seg_data.get('rel_amplitude', None),
                'percentile_1': seg_data.get('percentile_1', None),
                'variance_explained': seg_data.get('variance_explained', None),
            })
            continue

        phi_corrected = estimate_phase_from_protophase(phi)

        corrected_segments.append({
            'phase': phi_corrected,
            'rel_amplitude': seg_data.get('rel_amplitude', None),
            'percentile_1': seg_data.get('percentile_1', None),
            'variance_explained': seg_data.get('variance_explained', None),
        })

    return {
        GLOBAL_PHASE.key: corrected_segments
    }
