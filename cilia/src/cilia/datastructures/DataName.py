from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from cilia.datastructures.data_schemas import ScalarSchema, ArraySchema, DictSchema, DataSchema, ListSchema

_DATA_NAME_REGISTRY: Dict[str, "DataName"] = {}

@dataclass(frozen=True)
class DataName:
    key: str
    schema: DataSchema
    description: Optional[str] = None

    def __post_init__(self):
        if self.key in _DATA_NAME_REGISTRY:
            raise ValueError(
                f"DataName '{self.key}' already registered"
            )
        _DATA_NAME_REGISTRY[self.key] = self

    def __hash__(self) -> int:
        return hash(self.key)

    @staticmethod
    def get(key: str) -> "DataName":
        try:
            return _DATA_NAME_REGISTRY[key]
        except KeyError:
            raise KeyError(
                f"Unknown DataName '{key}'. "
                f"Did you forget to define it in DataName.py?"
            )


TORQUE_POWER = DataName(
    key="torque_power",
    description=(
        "Torque/power analysis including q95 and q80 envelope fits, "
        "motor parameter bounds for several motor models, total work, "
        "and ATP consumption."
    ),
    schema=DictSchema(
        fields={
            "q85": DictSchema(
                fields={
                    # strict scalar bounds
                    "Fstall_min": ScalarSchema(float),
                    "v0_min_detach": ScalarSchema(float),
                    "v0_min_linear_drag": ScalarSchema(float),

                    # ideal motor with detachment: step function
                    "ideal_step_v0": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "ideal_step_F0": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # linear force-velocity with detachment
                    "v0_grid_detach": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "F0_linear_detach": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # linear force-velocity allowing drag
                    "v0_grid_linear_drag": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "F0_linear_drag": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # fitted superellipse boundary in (|dgamma/dt|, |tau|)
                    "x_vals": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "tau_vals": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                }
            ),
            "q75": DictSchema(
                fields={
                    # strict scalar bounds
                    "Fstall_min": ScalarSchema(float),
                    "v0_min_detach": ScalarSchema(float),
                    "v0_min_linear_drag": ScalarSchema(float),

                    # ideal motor with detachment: step function
                    "ideal_step_v0": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "ideal_step_F0": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # linear force-velocity with detachment
                    "v0_grid_detach": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "F0_linear_detach": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # linear force-velocity allowing drag
                    "v0_grid_linear_drag": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "F0_linear_drag": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),

                    # fitted superellipse boundary in (|dgamma/dt|, |tau|)
                    "x_vals": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                    "tau_vals": ArraySchema(
                        ndim=1,
                        shape=(None,),
                        dtype=np.floating,
                    ),
                }
            ),
            "work": ScalarSchema(float),
            "N_ATP": ScalarSchema(float),
        }
    ),
)

TANGENT_ANGLE = DataName(
    key="tangent_angle",
    description="List of tangent angle segments; each is (n_frames_segment, n_s); rad , equidistant sampled",
    schema=ListSchema(
        element_schema=ArraySchema(
            ndim=2,
            shape=(None, None),
            dtype=np.floating,
        )
    ),
)

MEAN_SHAPE= DataName(
    key="mean_shape",
    description="(n_s); rad , equidistant sampled",
    schema=ArraySchema(
            ndim=1,
            shape=(None),
            dtype=np.floating,
        )
)

GLOBAL_PHASE = DataName(
    key="global_phase",
    description="List of phase segments with diagnostics",
    schema=ListSchema(
        element_schema=DictSchema(
            fields={
                "phase": ArraySchema(
                    ndim=1,
                    shape=(None,),
                    dtype=np.floating,
                ),
                "rel_amplitude": ArraySchema(
                    ndim=1,
                    shape=(None,),
                    dtype=np.floating,
                ),
                "percentile_1": ScalarSchema(float),
                "variance_explained": ScalarSchema(float),
            }
        )
    )
)

DT_FRAME = DataName(
    key="dt_frame",
    description="time between frames in ms",
    schema=ScalarSchema(float),
)

F = DataName(
    key="f",
    description="Frequency in Hz",
    schema=ScalarSchema(float),
)

A = DataName(
    key="a",
    description="Amplitude",
    schema=ScalarSchema(float),
)

Q = DataName(
    key="q",
    description="Quality factor (dimensionless)",
    schema=ScalarSchema(float),
)

LAM = DataName(
    key="lam",
    description="relative Wavelength lambda/L",
    schema=ScalarSchema(float),
)

LENGTH = DataName(
    key="length",
    description="cilium length in mum",
    schema=ScalarSchema(float),
)

CONDITION_DESCRIPTION = DataName(
    key="condition_descritption",
    description="Description of the condition",
    schema=ScalarSchema(dict),
)

GEYER = DataName(
    key="geyer",
    description="Geyer-mode decomposition of tangent angle waveform",
    schema=DictSchema(
        fields={
            # profiles
            "psi0_s": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
            "a_s": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
            "phi_s": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
            "keep_mask": ArraySchema(ndim=1, shape=(None,), dtype=bool),

            # Legendre coefficients
            "psi0_coeffs": DictSchema(
                fields={
                    "0": ScalarSchema(float),
                    "1": ScalarSchema(float),
                    "2": ScalarSchema(float),
                }
            ),
            "a_coeffs": DictSchema(
                fields={
                    "0": ScalarSchema(float),
                    "1": ScalarSchema(float),
                    "2": ScalarSchema(float),
                }
            ),
            "phi_coeffs": DictSchema(
                fields={
                    "0": ScalarSchema(float),
                    "1": ScalarSchema(float),
                    "2": ScalarSchema(float),
                }
            ),

            # scalars
            "lambda": ScalarSchema(float), #relative
            "r2_full_profiles": ScalarSchema(float),
            "r2_matrix": ArraySchema(ndim=2, shape=(None,None), dtype=np.floating),
        }
    ),
)

PERIODIC_AVG = DataName(
    key="periodic_avg",
    description=(
        "Phase-averaged tangent angle Ψ̄(φ,s) together with "
        "estimated global uncertainty of the limit cycle."
    ),
    schema=DictSchema(
        fields={
            "mean": ArraySchema(
                ndim=2,
                shape=(None, None),
                dtype=np.floating,
            ),
            "uncertainty": ScalarSchema(float),
        }
    ),
)

R2_OBSERVABLES = DataName(
    key="r2_observables",
    description=(
        "R2 values quantifying reconstruction quality of observable reductions. "
        "For the Geyer reduction, geyer_r2_a_phi_matrix has rows "
        "[a_const, a_linear, a_quadratic, a_full] and columns "
        "[phi_const, phi_linear, phi_quadratic, phi_full]."
    ),
    schema=DictSchema(
        fields={
            # mean shape pipeline
            "mean_shape_linear": ScalarSchema(float),
            "mean_shape_curvature_only": ScalarSchema(float),

            # gamma pipeline before Geyer profile reductions
            "gamma_pca2": ScalarSchema(float),
            "gamma_limit_cycle": ScalarSchema(float),

            # Geyer profile-reduction matrix:
            # rows: a_const, a_linear, a_quadratic, a_full
            # cols: phi_const, phi_linear, phi_quadratic, phi_full
            "geyer_r2_a_phi_matrix": ArraySchema(
                ndim=2,
                shape=(None, None),
                dtype=np.floating,
            ),

            # convenience aliases from the matrix
            "gamma_geyer_full_profiles": ScalarSchema(float),
            "gamma_geyer_a_const_phi_linear": ScalarSchema(float),
            "gamma_geyer_a_const_phi_quadratic": ScalarSchema(float),
            "gamma_geyer_a_linear_phi_linear": ScalarSchema(float),
            "gamma_geyer_a_quadratic_phi_quadratic": ScalarSchema(float),

            # profile-only reductions, not reconstruction of gamma
            "a_profile_const": ScalarSchema(float),
            "a_profile_linear": ScalarSchema(float),
            "a_profile_quadratic": ScalarSchema(float),
            "phi_profile_const": ScalarSchema(float),
            "phi_profile_linear": ScalarSchema(float),
            "phi_profile_quadratic": ScalarSchema(float),

            # useful diagnostics
            "n_gamma_frames": ScalarSchema(float),
            "n_s": ScalarSchema(float),
            "n_phi": ScalarSchema(float),
        }
    ),
)
# LOCAL_PHASE_SERIES = DataName(
#     key="local_phase_series",
#     description="Local phase and amplitude from tangent angle series",
#     schema=DictSchema(
#         fields={
#             "phase": ArraySchema(
#                 ndim=2,
#                 shape=(None, None),
#                 dtype=np.floating,
#             ),
#             "amplitude": ArraySchema(
#                 ndim=2,
#                 shape=(None, None),
#                 dtype=np.floating,
#             ),
#             "aux": DictSchema(
#                 fields={
#                     "T": ScalarSchema(int),
#                     "wt": ScalarSchema(int),
#                     "ws": ScalarSchema(int),

#                     "too_small": ScalarSchema(bool),

#                     # per-s diagnostics (shape: (Ns,))
#                     "amp_p01_s": ArraySchema(
#                         ndim=1,
#                         shape=(None,),
#                         dtype=np.floating,
#                     ),
#                     "msd_dphi_norm_s": ArraySchema(
#                         ndim=1,
#                         shape=(None,),
#                         dtype=np.floating,
#                     ),
#                 }
#             ),
#         }
#     ),
# )

# CORR_FLUC = DataName(
#     key="corr_fluc",
#     description="Fluctuations and correlations of wavelength and amplitude",
#     schema=DictSchema(
#         fields={
#             # ----------------------------------------
#             # time series
#             # ----------------------------------------
#             "lambda_t": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),
#             "mean_a_t": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),

#             # ----------------------------------------
#             # correlations
#             # ----------------------------------------
#             "lambda_autocorr": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),
#             "mean_a_autocorr": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),
#             "lambda_mean_a_crosscorr": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),

#             # ----------------------------------------
#             # 2D amplitude correlation
#             # ----------------------------------------
#             "amp_corr_dt_ds": ArraySchema(
#                 ndim=2,
#                 shape=(None, None),
#                 dtype=np.floating,
#             ),

#             "lags_dt_idx": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.integer,
#             ),
#             "lags_t_phys": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),
#             "lags_t_rel": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),

#             "lags_s_idx": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.integer,
#             ),
#             "lags_s_rel": ArraySchema(
#                 ndim=1,
#                 shape=(None,),
#                 dtype=np.floating,
#             ),

#             "too_small": ScalarSchema(bool),
#         }
#     ),
# )

# SEGMENT_CORR_FLUC = DataName(
#     key="segment_corr_fluc",
#     description="Fluctuations and correlations of wavelength and amplitude (segment-wise)",
#     schema=DictSchema(
#         fields={
#             "lambda_t": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
#             "mean_a_t": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),

#             "lambda_autocorr": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
#             "mean_a_autocorr": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),

#             "lambda_mean_a_crosscorr": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),

#             "amp_corr_dt_ds": ArraySchema(ndim=2, shape=(None, None), dtype=np.floating),

#             # ✅ NEW consistent lag definition
#             "lags_dt_idx": ArraySchema(ndim=1, shape=(None,), dtype=np.integer),
#             "lags_t_phys": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),
#             "lags_t_rel": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),

#             "lags_s_idx": ArraySchema(ndim=1, shape=(None,), dtype=np.integer),
#             "lags_s_rel": ArraySchema(ndim=1, shape=(None,), dtype=np.floating),

#             "too_small": ScalarSchema(bool),
#         }
#     ),
# )

# LOCAL_PHASE = DataName(
#     key="local_phase",
#     description="List of local phase results (phase, amplitude, aux) per segment",
#     schema=ListSchema(
#         element_schema=DictSchema(
#             fields={
#                 "phase": ArraySchema(
#                     ndim=2,
#                     shape=(None, None),
#                     dtype=np.floating,
#                 ),
#                 "amplitude": ArraySchema(
#                     ndim=2,
#                     shape=(None, None),
#                     dtype=np.floating,
#                 ),
#                 "aux": DictSchema(
#                     fields={
#                         "T": ScalarSchema(int),
#                         "wt": ScalarSchema(int),
#                         "ws": ScalarSchema(int),

#                         "too_small": ScalarSchema(bool),

#                         "amp_p01_s": ArraySchema(
#                             ndim=1,
#                             shape=(None,),
#                             dtype=np.floating,
#                         ),
#                         "msd_dphi_norm_s": ArraySchema(
#                             ndim=1,
#                             shape=(None,),
#                             dtype=np.floating,
#                         ),
#                     }
#                 ),
#             }
#         )
#     ),
# )

# PHASE_DEFECTS = DataName(
#     key="phase_defects",
#     description="Per segment: raw, merged and cleaned phase defects plus quality interval",
#     schema=ListSchema(
#         element_schema=DictSchema(
#             {
#                 "raw": ArraySchema(
#                     ndim=2,
#                     shape=(None, 3),
#                     dtype=np.int_,
#                 ),
#                 "merged": ArraySchema(
#                     ndim=2,
#                     shape=(None, 3),
#                     dtype=np.int_,
#                 ),
#                 "clean": ArraySchema(
#                     ndim=2,
#                     shape=(None, 3),
#                     dtype=np.int_,
#                 ),
#                 "good_s_interval": ArraySchema(
#                     ndim=1,
#                     shape=(2,),
#                     dtype=np.integer,
#                 ),
#                 "effective_Nt": ScalarSchema(int),
#                 "effective_T": ScalarSchema(float),
#             }
#         )
#     ),
# )


# PHASE_DEFECTS_SERIES = DataName(
#     key="phase_defects_series",
#     description="Phase defects for full time series (no segmentation)",
#     schema=DictSchema(
#         {
#             "raw": ArraySchema(
#                 ndim=2,
#                 shape=(None, 3),
#                 dtype=np.int_,
#             ),
#             "merged": ArraySchema(
#                 ndim=2,
#                 shape=(None, 3),
#                 dtype=np.int_,
#             ),
#             "clean": ArraySchema(
#                 ndim=2,
#                 shape=(None, 3),
#                 dtype=np.int_,
#             ),
#             "good_s_interval": ArraySchema(
#                 ndim=1,
#                 shape=(2,),
#                 dtype=np.integer,
#             ),
#             "effective_Nt": ScalarSchema(int),
#             "effective_T": ScalarSchema(float),
#         }
#     ),
# )

# ARCLENGTH = DataName(
#     key="arclength",
#     description="Arclength coordinate along the filament in mum",
#     schema=ArraySchema(
#         ndim=1,
#         shape=(None,),
#         dtype=np.floating,
#     ),
# )


# CORR_LENGTH = DataName(
#     key="corr_length",
#     description="relative corrleation length xi/L",
#     schema=ScalarSchema(float),
# )

# NON_ISO = DataName(
#     key="non_iso",
#     description="nonisochrony parameters",
#     schema=ScalarSchema(float),
# )

# ML_PHASE_DIFF = DataName(
#     key="ml_phase_diff",
#     description="maximum likelihood estimate for more complex noise model",
#     schema=DictSchema(
#         fields={
#             "tau": ScalarSchema(float),
#             "omega0": ScalarSchema(float),
#             "D_omega": ScalarSchema(float),
#             "Q": ScalarSchema(float),
#             "sigma_phi": ScalarSchema(float),
#         }
#     ),
# )


# HYDRO_CONTRIBUTION = DataName(
#     key="hydro_contribution",
#     description="Mean hydrodynamic vs elastic torque contributions and power dissipation",
#     schema=DictSchema(
#         fields={
#             "mean_R_free": ScalarSchema(float),
#             "mean_R_fixed": ScalarSchema(float),
#             "mean_R_chlamy": ScalarSchema(float),
#             "mean_power_free": ScalarSchema(float),
#             "mean_power_fixed": ScalarSchema(float),
#             "mean_power_chlamy": ScalarSchema(float),
#         }
#     )
# )

# OSCILLATORY = DataName(
#     key="oscillatory",
#     description="Whether the signal exhibits oscillatory behavior",
#     schema=ScalarSchema(bool),
# )
