import numpy as np
from cilia.transformers.Transformer import data_transformer
from cilia.datastructures.DataName import (#LOCAL_PHASE_SERIES, 
#                                            TANGENT_ANGLE_SERIES, 
                                           LENGTH, 
TANGENT_ANGLE, 
# PHASE_DEFECTS_SEGMENTS, 
# LOCAL_PHASE_SEGMENTS, 
DT_FRAME,
# CORR_LENGTH, 
# NON_ISO, 
# PHASE_DEFECTS_SERIES, 
# CORR_FLUC, 
F)

from cilia.algorithms.local_phase_estimation import (
    get_local_phase, 
)
from cilia.algorithms.phase_defects import detect_and_clean_defects
from cilia.algorithms.correlation_length import get_correlation_length, get_correlation_fluctuation
from cilia.algorithms.non_iso import get_non_iso_local

# @data_transformer(
#     inputs=(TANGENT_ANGLE,),
#     outputs=(LOCAL_PHASE_SEGMENTS,),
# )
# def segment_local_phase_t(tangent_angle_segments):
#     out = []
#     for tangent_angle_series in tangent_angle_segments:
#         phase, amplitude, aux = get_local_phase(tangent_angle_series)
#         out.append({"phase": phase, "amplitude": amplitude, "aux": aux})
#     return {LOCAL_PHASE_SEGMENTS.key: out}


# @data_transformer(
#     inputs=(TANGENT_ANGLE_SERIES,),
#     outputs=(LOCAL_PHASE_SERIES,),
# )
# def local_phase_t(tangent_angle_series):
#     phase, amplitude, aux = get_local_phase(tangent_angle_series)
#     return {LOCAL_PHASE_SERIES.key: {"phase": phase, "amplitude": amplitude, "aux": aux}}


# @data_transformer(
#     inputs=(LOCAL_PHASE_SEGMENTS, DT_FRAME),
#     outputs=(PHASE_DEFECTS_SEGMENTS,),
# )
# def segment_defect_rate_t(
#     local_phase_segments, dt_frame,
#     *,
#     theta_t: int = 0.5, # rel Periode
#     theta_s: int = 0.1, # rel length
#     msd_thr: float = 5.0,
#     amp_thr: float = 0.05,
#     require_min_abs_winding: float = np.pi,
#     exclude_margin_s: int = 1,
# ):
#     """
#     For each segment dict in local_phase_segments:
#       expects keys: "phase", "amplitude", "aux"

#     Stores per segment:
#       {
#         "raw": (N,3) array [t,s,charge],
#         "merged": (M,3),
#         "clean": (K,3),
#         "good_s_interval": [s_min, s_max],
#       }
#     """
#     out = []

#     for seg in local_phase_segments:
#         phase = seg["phase"]
#         aux = seg.get("aux", {})

#         defects = detect_and_clean_defects(
#             phase,
#             aux,
#             theta_t=theta_t,
#             theta_s=theta_s,
#             msd_thr=msd_thr,
#             amp_thr=amp_thr,
#             require_min_abs_winding=require_min_abs_winding,
#             exclude_margin_s=exclude_margin_s,
#             dt=dt_frame,
#         )
#         out.append(defects)

#     return {PHASE_DEFECTS_SEGMENTS.key: out}

@data_transformer(
    inputs=(LOCAL_PHASE_SEGMENTS,),
    outputs=(CORR_LENGTH,),
)
def segment_correlation_length_t(local_phase_segments):
    phase_list = []

    for seg in local_phase_segments:
        phase = seg.get("phase", None)
        if phase is None:
            continue

        phase = np.asarray(phase)

        if phase.ndim != 2 or phase.shape[0] < 50 or phase.shape[1] < 10:
            continue

        phase_list.append(phase)

    if len(phase_list) == 0:
        return {CORR_LENGTH.key: np.nan}

    # -------------------------------------------------
    # concatenate along time axis
    # -------------------------------------------------
    phase_all = np.concatenate(phase_list, axis=0)  # (sum_T, Ns)
    valid_mask = np.all(np.isfinite(phase_all), axis=1)
    phase_all = phase_all[valid_mask]
    xi, _ = get_correlation_length(phase_all)

    return {
        CORR_LENGTH.key: float(xi) if np.isfinite(xi) else np.nan
    }

@data_transformer(
    inputs=(LOCAL_PHASE_SERIES,),
    outputs=(CORR_LENGTH,),
)
def correlation_length_t(local_phase_dict):
    phase = local_phase_dict.get("phase", None)

    if phase is None:
        return {CORR_LENGTH.key: np.nan}

    phase = np.asarray(phase)

    if phase.ndim != 2 or phase.shape[0] < 50 or phase.shape[1] < 10:
        return {CORR_LENGTH.key: np.nan}

    # remove invalid rows
    valid_mask = np.all(np.isfinite(phase), axis=1)
    phase = phase[valid_mask]

    if len(phase) == 0:
        return {CORR_LENGTH.key: np.nan}

    xi, _ = get_correlation_length(phase)

    return {
        CORR_LENGTH.key: float(xi) if np.isfinite(xi) else np.nan
    }

def _empty_result():
    return {
        "lambda_t": np.array([]),
        "mean_a_t": np.array([]),

        "lambda_autocorr": np.array([]),
        "mean_a_autocorr": np.array([]),
        "lambda_mean_a_crosscorr": np.array([]),

        "amp_corr_dt_ds": np.zeros((0, 0)),

        # 🔥 NEW lag definitions
        "lags_dt_idx": np.array([], dtype=int),
        "lags_t_phys": np.array([], dtype=float),
        "lags_t_rel": np.array([], dtype=float),

        "lags_s_idx": np.array([], dtype=int),
        "lags_s_rel": np.array([], dtype=float),

        "too_small": True,
    }

@data_transformer(
    inputs=(LOCAL_PHASE_SERIES, F, DT_FRAME),
    outputs=(CORR_FLUC,),
)
def correlations_and_fluctuations_t(local_phase_dict, f, dt):
    phase = local_phase_dict.get("phase", None)
    amplitude = local_phase_dict.get("amplitude", None)
    if phase is None or amplitude is None:
        return {CORR_FLUC.key: _empty_result()}

    phase = np.asarray(phase)
    amplitude = np.asarray(amplitude)

    # -------------------------------------------------
    # basic sanity checks
    # -------------------------------------------------
    if phase.ndim != 2 or amplitude.ndim != 2:
        return {CORR_FLUC.key: _empty_result()}

    Nt, Ns = phase.shape

    if Nt < 50 or Ns < 10:
        return {CORR_FLUC.key: _empty_result()}

    # -------------------------------------------------
    # remove invalid frames
    # -------------------------------------------------
    valid_mask = np.all(np.isfinite(phase), axis=1) & np.all(np.isfinite(amplitude), axis=1)

    phase = phase[valid_mask]
    amplitude = amplitude[valid_mask]

    if len(phase) == 0:
        return {CORR_FLUC.key: _empty_result()}

    # -------------------------------------------------
    # core computation
    # -------------------------------------------------
    result = get_correlation_fluctuation(
        phase,
        amplitude,
        f,dt,
        debug_plot=False,
    )

    # -------------------------------------------------
    # ensure schema compliance
    # -------------------------------------------------
    return {
        CORR_FLUC.key: {
            "lambda_t": np.asarray(result["lambda_t"]),
            "mean_a_t": np.asarray(result["mean_a_t"]),

            "lambda_autocorr": np.asarray(result["lambda_autocorr"]),
            "mean_a_autocorr": np.asarray(result["mean_a_autocorr"]),
            "lambda_mean_a_crosscorr": np.asarray(result["lambda_mean_a_crosscorr"]),

            "amp_corr_dt_ds": np.asarray(result["amp_corr_dt_ds"]),

            # 🔥 NEW (correct)
            "lags_dt_idx": np.asarray(result["lags_dt_idx"], dtype=int),
            "lags_t_phys": np.asarray(result["lags_t_phys"], dtype=float),
            "lags_t_rel": np.asarray(result["lags_t_rel"], dtype=float),

            "lags_s_idx": np.asarray(result["lags_s_idx"], dtype=int),
            "lags_s_rel": np.asarray(result["lags_s_rel"], dtype=float),

            "too_small": False,
        }
    }

from cilia.datastructures.DataName import SEGMENT_CORR_FLUC
from cilia.datastructures.DataName import LOCAL_PHASE_SEGMENTS
@data_transformer(
    inputs=(LOCAL_PHASE_SEGMENTS, F, DT_FRAME),
    outputs=(SEGMENT_CORR_FLUC,),
)
def segment_correlations_and_fluctuations_t(local_phase_segments, f, dt):

    if local_phase_segments is None:
        return {
            SEGMENT_CORR_FLUC.key: {
                "lambda_t": np.array([]),
                "mean_a_t": np.array([]),
                "lambda_autocorr": np.array([]),
                "mean_a_autocorr": np.array([]),
                "lambda_mean_a_crosscorr": np.array([]),
                "amp_corr_dt_ds": np.array([[]]),
                "lags_dt_idx": np.array([], dtype=int),
                "lags_t_phys": np.array([], dtype=float),
                "lags_t_rel": np.array([], dtype=float),

                "lags_s_idx": np.array([], dtype=int),
                "lags_s_rel": np.array([], dtype=float),
                "too_small": True,
            }
        }

    results = []

    for seg in local_phase_segments:

        phase = seg.get("phase")
        amplitude = seg.get("amplitude")

        if phase is None or amplitude is None:
            continue

        phase = np.asarray(phase)
        amplitude = np.asarray(amplitude)

        if phase.ndim != 2 or amplitude.ndim != 2:
            continue

        if phase.shape[0] < 20 or phase.shape[1] < 10:
            continue

        try:
            res = get_correlation_fluctuation(
                phase,
                amplitude,
                f,
                dt,
                debug_plot=False,
            )
        except Exception:
            continue

        results.append(res)

    # ----------------------------------------
    # merge segments
    # ----------------------------------------
    if len(results) == 0:
        return {
            SEGMENT_CORR_FLUC.key: {
                "lambda_t": np.array([]),
                "mean_a_t": np.array([]),
                "lambda_autocorr": np.array([]),
                "mean_a_autocorr": np.array([]),
                "lambda_mean_a_crosscorr": np.array([]),
                "amp_corr_dt_ds": np.array([[]]),
                "lags_t": np.array([]),
                "lags_t_rel": np.array([]),
                "lags_s": np.array([]),
                "too_small": True,
            }
        }

    # concatenate time series
    lambda_all = np.concatenate([r["lambda_t"] for r in results if len(r["lambda_t"]) > 0])
    amp_all = np.concatenate([r["mean_a_t"] for r in results if len(r["mean_a_t"]) > 0])

    # average correlations (simple mean)
    def _avg(key):
        arrs = [r[key] for r in results if key in r]
        if len(arrs) == 0:
            return np.array([])
        L = min(len(a) for a in arrs)
        arrs = [a[:L] for a in arrs]
        return np.nanmean(arrs, axis=0)

    def _get_first_valid(key, default):
        for r in results:
            if key in r and r[key] is not None:
                return r[key]
        return default

    out = {
        "lambda_t": lambda_all,
        "mean_a_t": amp_all,
        "lambda_autocorr": _avg("lambda_autocorr"),
        "mean_a_autocorr": _avg("mean_a_autocorr"),
        "lambda_mean_a_crosscorr": _avg("lambda_mean_a_crosscorr"),
        "amp_corr_dt_ds": results[0]["amp_corr_dt_ds"],
        "lags_dt_idx": np.asarray(_get_first_valid("lags_dt_idx", np.array([], int)), dtype=int),
        "lags_t_phys": np.asarray(_get_first_valid("lags_t_phys", np.array([], float)), dtype=float),
        "lags_t_rel": np.asarray(_get_first_valid("lags_t_rel", np.array([], float)), dtype=float),

        "lags_s_idx": np.asarray(_get_first_valid("lags_s_idx", np.array([], int)), dtype=int),
        "lags_s_rel": np.asarray(_get_first_valid("lags_s_rel", np.array([], float)), dtype=float),
        "too_small": False,
    }

    return {SEGMENT_CORR_FLUC.key: out}


@data_transformer(
    inputs=(LOCAL_PHASE_SERIES,),
    outputs=(NON_ISO,),
)
def non_iso_t(local_phase_dict):
    phase = local_phase_dict.get("phase", None)

    if phase is None:
        return {NON_ISO.key: np.nan}

    phase = np.asarray(phase)

    if phase.ndim != 2 or phase.shape[0] < 50:
        return {NON_ISO.key: np.nan}

    val = get_non_iso_local(phase)

    return {
        NON_ISO.key: float(val) if np.isfinite(val) else np.nan
    }

@data_transformer(
    inputs=(LOCAL_PHASE_SERIES, DT_FRAME),
    outputs=(PHASE_DEFECTS_SERIES,),
)
def defect_rate_t(
    local_phase_dict,
    dt_frame,
    *,
    theta_t: float = 0.5,
    theta_s: float = 0.1,
    msd_thr: float = 5.0,
    amp_thr: float = 0.05,
    require_min_abs_winding: float = np.pi,
    exclude_margin_s: int = 1,
):
    """
    Detect and clean phase defects on full time series.
    Returns a single defect dictionary (not a list).
    """

    phase = local_phase_dict.get("phase", None)
    aux = local_phase_dict.get("aux", {})

    if phase is None:
        return {PHASE_DEFECTS_SERIES.key: {
            "raw": np.empty((0, 3), dtype=int),
            "merged": np.empty((0, 3), dtype=int),
            "clean": np.empty((0, 3), dtype=int),
            "good_s_interval": np.array([0, 0], dtype=int),
            "effective_Nt": 0,
            "effective_T": 0.0,
        }}

    defects = detect_and_clean_defects(
        phase,
        aux,
        theta_t=theta_t,
        theta_s=theta_s,
        msd_thr=msd_thr,
        amp_thr=amp_thr,
        require_min_abs_winding=require_min_abs_winding,
        exclude_margin_s=exclude_margin_s,
        dt=dt_frame,
    )

    return {
        PHASE_DEFECTS_SERIES.key: defects
    }

