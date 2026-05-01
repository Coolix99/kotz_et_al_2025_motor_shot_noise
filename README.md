# Motor Shot Noise Explains Active Fluctuations in a Single Cilium

Source code used for the paper *Motor Shot Noise Explains Active Fluctuations in a Single Cilium*.  
Authors: **Maximilian Kotz, Veikko F. Geyer, and Benjamin M. Friedrich**

---

## Installation

The repository contains a `cilia` subfolder with helper functions used throughout the project.  
To make these utilities available system-wide, install them locally by creating a symbolic link:

```bash
cd cilia
pip install -e .
```

## Folder Structure

The code and data are organized in a stepwise manner, following the analysis pipeline:

- **01_cpp_Sim/**  
  Code used for most numerical simulations. This directory must be built.  

- **02_analyse_data/**  
  Python scripts to analyze the data from the C++ program and experimental files.

- **03_SBI/**  
  Simulation-based inference to obtain alternative parameter sets.  

- **04_plots/**  
  Python script for creating raw versions of the figures for the manuscript.

- **06_additional_simulations/**  
  Simulation of 2d Cass model with Hydrodynamics.

- **cilia/**  
  Collection of helper functions and reusable Python modules (installed locally as described above). 
