# Molten Pool Flow Reconstruction using Physics-Informed Neural Networks

Code and example data for the paper:

> **Reconstruction of Molten Pool Flow Fields from High-Speed Video Using Physics-Informed Neural Networks**
> *Journal of Manufacturing Processes* (under review)

---

## Overview

This repository provides the processing pipeline used to reconstruct molten pool surface flow fields from high-speed video in wire arc additive manufacturing (WAAM). The method combines RAFT optical flow estimation with a physics-informed neural network (PINN) that enforces incompressible Navier-Stokes equations, recovering dense velocity fields, pressure distributions, and physical parameters (effective viscosity, Reynolds number, arc force) from sparse visual observations.

Three experimental conditions are covered: wire heights of 0, 2, and 4 mm, corresponding to quasi-steady and time-dependent pool dynamics.

---

## Pipeline

The scripts are numbered in execution order:

| Script | Description |
|---|---|
| `step0_create_mask.py` | Interactive tool to draw a polygon mask around the weld pool boundary |
| `step1_raft_extraction.py` | Extracts dense optical flow from the masked video using RAFT |
| `step2_calibration.py` | Interactive two-stage calibration: fits pool boundary axes and computes homography |
| `step3_apply_correction.py` | Applies homography correction and symmetrization to all RAFT flow fields |
| `step4_data_illustration.py` | Generates visualization figures for each processed frame |
| `step5_0mm_v5_no_A_prior.py` | PINN reconstruction for the quasi-steady 0 mm case (arc force learned freely) |
| `step5_2mm_v5_A_prior.py` | PINN reconstruction for the time-dependent 2 mm case (arc force amplitude transferred from 0 mm case as prior) |

The 4 mm case uses the same script as the 2 mm case with different input data.

---

## Example Videos

The `example videos/` folder contains short, compressed clips illustrating the high-speed video input for each wire height condition. These are provided for illustration only. The original full-resolution recordings are not included due to file size.

---

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- NumPy, SciPy, Matplotlib
- [RAFT](https://github.com/princeton-vl/RAFT) (for step 1)

---

## Repository

[https://github.com/YueCaoUKY/Molten-Pool-Flow-Detection-via-PINN](https://github.com/YueCaoUKY/Molten-Pool-Flow-Detection-via-PINN)

---

## Citation

If you use this code, please cite the paper once published.
