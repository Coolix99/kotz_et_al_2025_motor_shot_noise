# Motor Shot Noise Explains Active Fluctuations in a Single Cilium

Source code used for the paper *Motor Shot Noise Explains Active Fluctuations in a Single Cilium*.  
Authors: **Maximilian Kotz, Veikko F. Geyer, and Benjamin M. Friedrich**

---

## Installation

The repository contains a `utils_rdf` subfolder with helper functions used throughout the project.  
To make these utilities available system-wide, install them locally by creating a symbolic link:

```bash
cd utils_rdf
pip install -e .
```
After installation, copy `libspde_capi.so` into `utils_rdf`.  
This file is generated when building the C code in `01_cpp_Sim`.

```bash
cp ../01_cpp_Sim/build/libspde_capi.so utils_rdf/
```

## Folder Structure

The code and data are organized in a stepwise manner, following the analysis pipeline:

- **01_cpp_Sim/**  
  Code used for most numerical simulations. This directory must be built.  
  Inside the build folder you will find:
  - `spde_run`, which is used to run the simulations  
  - `libspde_capi.so`, which must be copied to `utils_rdf`

- **02_analyse_data/**  
  Python scripts to analyze the data from the C++ program and experimental files.

- **03_SBI/**  
  Simulation-based inference to obtain alternative parameter sets.  
  The script `01_collect_S123.py` should be run in parallel with stages S1–S3.

- **04_plots/**  
  Python scripts for creating raw versions of the figures for the manuscript.

- **05_matlab/**  
  MATLAB files used for additional simulations.

- **utils_rdf_pkg/**  
  Collection of helper functions and reusable Python modules (installed locally as described above).  
  `Config.py` contains the folder structure, which may need to be adapted.

