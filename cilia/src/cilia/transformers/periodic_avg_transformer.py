from cilia.transformers.Transformer import data_transformer
from cilia.algorithms.periodicAverage import periodic_avg
import numpy as np
from cilia.datastructures.DataName import GLOBAL_PHASE, TANGENT_ANGLE, PERIODIC_AVG


@data_transformer(
    inputs=(GLOBAL_PHASE, TANGENT_ANGLE),
    outputs=(PERIODIC_AVG,),
)
def periodic_avg_segment_t(phase_segments, tangent_angle_segments):
    if len(phase_segments) == 0:
        return {
            PERIODIC_AVG.key: {
                "mean": None,
                "uncertainty": np.nan,
            }
        }

    # --- convert and stack ---
    phase_list = []
    angle_list = []

    for phi_seg_data, psi_seg in zip(phase_segments, tangent_angle_segments):
        phi_seg = np.asarray(phi_seg_data['phase'], dtype=float)
        psi_seg = np.asarray(psi_seg, dtype=float)

        if phi_seg.ndim != 1:
            continue
        if psi_seg.ndim != 2:
            continue
        if psi_seg.shape[0] != phi_seg.shape[0]:
            continue

        phase_list.append(phi_seg)
        angle_list.append(psi_seg)

    if len(phase_list) == 0:
        return {
            PERIODIC_AVG.key: {
                "mean": None,
                "uncertainty": np.nan,
            }
        }

    phase_full = np.concatenate(phase_list, axis=0)
    angle_full = np.vstack(angle_list)

    # --- run standard algorithm ---
    avg, uncertainty = periodic_avg(
        phase_full,
        angle_full,
    )
    return {
        PERIODIC_AVG.key: {
            "mean": avg,
            "uncertainty": uncertainty,
        }
    }
