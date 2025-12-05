import os
from pathlib import Path

###------------ Config folder structure -----------------------
# Default data dirs (None means "not set" until overridden)
# MATLAB_DATA_DIR_PHASESPACE = None
# MATLAB_DATA_DIR_EXTRACTION = None
# MATLAB_DATA_DIR_NEW_PARAMETER = None

SPDE_DATA_DIR_PHASESPACE = None
SPDE_DATA_DIR_EXTRACTION = None
SPDE_DATA_DIR_NEW_PARAMETER = None
SPDE_DATA_DIR_PARAMSCAN = None 

SHARMA_DATA_DIR = None

# __file__ points to utils_rdf_pkg/utils_rdf/config.py
REPO_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DATA_DIR = REPO_ROOT / "analysed_data"

OUTPUT_DATA_PHASESPACE_DIR=OUTPUT_DATA_DIR / "summary_phasespace"
OUTPUT_DATA_EXTRACTION_DIR=OUTPUT_DATA_DIR / "summary_extraction"
OUTPUT_DATA_NEW_PARAMETER_DIR=OUTPUT_DATA_DIR / "summary_new_parameter"
OUTPUT_DATA_PARAMSCAN_DIR=OUTPUT_DATA_DIR / "summary_paramscan"
OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR=OUTPUT_DATA_DIR / "special_simulation_results"
OUTPUT_DATA_WN_PHASESPACE_DIR = OUTPUT_DATA_DIR / "summary_wn_phasespace"

OUTPUT_DATA_CWN_PHASESPACE_DIR = OUTPUT_DATA_DIR / "summary_cwn_phasespace"
OUTPUT_DATA_CWN_OPEN_PHASESPACE_DIR = OUTPUT_DATA_DIR / "summary_cwn_open_phasespace"
OUTPUT_DATA_CWN_PERIODIC_PHASESPACE_DIR = OUTPUT_DATA_DIR / "summary_cwn_periodic_phasespace"

OUTPUT_DATA_SBI_DIR=OUTPUT_DATA_DIR / "sbi_results"
OUTPUT_DATA_SBI_S1_DIR=OUTPUT_DATA_SBI_DIR / "S1"
OUTPUT_DATA_SBI_S2_DIR=OUTPUT_DATA_SBI_DIR / "S2"
OUTPUT_DATA_SBI_S3_DIR=OUTPUT_DATA_SBI_DIR / "S3"

OUTPUT_DATA_SHARME_DIR=OUTPUT_DATA_DIR / "experimental_results"

OUTPUT_FIGURES_DIR = REPO_ROOT / "figures"

OUTPUT_DATA_FLUCTUATIONS = OUTPUT_DATA_DIR / 'fluctuations'
OUTPUT_DATA_FLUCTUATIONS_EXTRACTION = OUTPUT_DATA_FLUCTUATIONS / 'extraction'
OUTPUT_DATA_FLUCTUATIONS_NEW_PARAMETER = OUTPUT_DATA_FLUCTUATIONS / 'new_parameter'
OUTPUT_DATA_FLUCTUATIONS_EXPERIMENT = OUTPUT_DATA_FLUCTUATIONS / 'experiment'
PHASE_DEFECT_RATES_CSV = Path(OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR) / "phase_defect_rates_vs_relN.csv"
SPATIAL_CORRELATION_CSV = Path(OUTPUT_DATA_SPECIAL_SIM_RESULTS_DIR) / "spatial_correlation_length_vs_relN.csv"

###------------ Config Matlab simulation parameters -----------------------
DT_SIMULATION = 0.0002  #dt of a frame used to rescale frequency, could be chnaged
TYPICAL_NS_SIMULATION = 101
INITIAL_TRANSIENT_TIME_SIMULATION = 500 # frames to ignore at the start of each simulation 
REMOVE_HILBERT_SIMULATION = 1000 # frames need to be removed after hilbert transform due to edge effects
N_HARMONICS_GLOBAL_PHASE_SIMULATION = 40  # number of Fourier modes to use to normalise tangent angle speed
N_HARMONICS_LOCAL_PHASE = 25

### Config Sharma data
N_SLIDING_WINDOWS_VARIANCE=20
MIN_LENGTH_AFTER_TRIM=500  
MIN_LENGTH_Q=2000
REMOVE_HILBERT_EXPERIMENT=100
N_HARMONICS_GLOBAL_PHASE_EXPERIMENT = 25

### Local overrides 
try:
    from .config_local import *
except ImportError:
    pass
