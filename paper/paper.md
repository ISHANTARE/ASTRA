---
title: 'ASTRA-Core: A High-Performance Open-Source Python Engine for Space Traffic Risk Analysis and Astrodynamics'
tags:
  - Python
  - astrodynamics
  - orbital mechanics
  - space situational awareness
  - satellite conjunction analysis
  - space debris
authors:
  - name: Ishan Tare
    orcid: 0009-0003-1818-0388
    affiliation: 1
affiliations:
  - name: School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India
    index: 1
date: 24 March 2026
bibliography: paper.bib
---

# Summary

The near-Earth space environment is increasingly congested with satellites and debris, creating growing collision risk. ASTRA-Core (Autonomous Space Traffic Risk Analyzer) is an open-source Python library providing a complete, pip-installable computational pipeline for space traffic management. It ingests Two-Line Element (TLE) catalog data, propagates full orbital catalogs, detects conjunction events via a spatially indexed screening pass, computes Time of Closest Approach (TCA) to millisecond precision, derives a formal Probability of Collision ($P_c$), and supports finite-burn collision avoidance maneuver (CAM) simulation.

ASTRA-Core is available as `pip install astra-core-engine` with source code at [https://github.com/ISHANTARE/ASTRA](https://github.com/ISHANTARE/ASTRA).

# Statement of Need

Kessler and Cour-Palais [-@kessler1978collision] established the theoretical basis for collision-induced debris cascades, a scenario now increasingly relevant as megaconstellations expand the on-orbit population beyond eight thousand active payloads tracked by the U.S. Space Surveillance Network. An operational-grade Space Situational Awareness (SSA) pipeline requires: (1) TLE ingestion and validation [@hoots1980models], (2) high-fidelity trajectory propagation accounting for perturbative forces, (3) efficient conjunction screening across large catalogs, and (4) formal $P_c$ computation to inform maneuver decisions.

Researchers and small satellite operators additionally need the ability to simulate proposed collision avoidance maneuvers and to modify the underlying physics models for research reproducibility. Existing open-source tools address individual parts of this pipeline but leave critical gaps. ASTRA-Core provides a unified, high-fidelity, and fully open implementation targeting aerospace research groups, university students, and small satellite operators who cannot access proprietary operations software.

# State of the Field

`sgp4` [@rhoads2020sgp4] provides a fast Python SGP4 propagator but offers no conjunction analysis, covariance handling, or maneuver modeling. `poliastro` [@rodriguez2022poliastro] covers orbital mechanics and mission design but lacks a conjunction pipeline and live space weather integration. Commercial tools such as AGI STK offer full operational capability but are closed-source and inaccessible without costly licenses.

ASTRA-Core's contribution is unifying high-fidelity numerical propagation, large-scale conjunction screening, probabilistic $P_c$ quantification, and finite-burn maneuver modeling into a single open, pip-installable package — a compact and reproducible implementation of the full SSA analytical chain [@vallado2013fundamentals; @alfano2005review].

# Software Design

**Propagation.** `astra.propagator` implements Cowell's direct integration using `scipy.integrate.solve_ivp` (Dormand-Prince RK8(7)), chosen for its adaptive step control at numerically stiff perigee passes. SciPy's adaptive error controller automatically tightens the step size at discontinuities, ensuring truncation error remains within specified tolerances throughout [@montenbruck2000satellite]. The force model includes two-body gravity, $J_2$–$J_4$ zonal harmonics (WGS-84), empirical atmospheric drag (NRLMSISE-00 parameterized by F10.7/Ap from CelesTrak [@celestrak2026]), and Solar/Lunar third-body perturbations from JPL DE421 via Skyfield [@park2021jpl]. Analytical fallbacks are available for offline operation.

**Maneuver modeling.** The propagation timeline is segmented at engine ignition and cutoff boundaries. Coast arcs integrate a 6-DOF state $[r, v]$; powered arcs integrate a 7-DOF state $[r, v, m]$ with mass depleted by $\dot{m} = -F / (I_{sp} \cdot g_0)$ (Tsiolkovsky equation). This segmented approach ensures the integrator never steps across a force-model discontinuity, eliminating truncation error at burn transitions. Thrust direction is updated dynamically at every sub-step in VNB (Velocity-Normal-Binormal) or RTN (Radial-Transverse-Normal) reference frames.

**HPC acceleration.** The inner force loop is JIT-compiled with Numba (`@njit(fastmath=True, cache=True)`). Sun/Moon ephemerides are pre-fitted to 25-node Chebyshev polynomial splines and evaluated inside the Numba kernel, eliminating Python I/O overhead inside the integrator loop.

**Conjunction screening.** A `scipy.spatial.cKDTree` spatial index rebuilt at each timestep reduces candidate screening to $O(N \log N)$ [@vallado2013fundamentals], avoiding the $O(N^2)$ naive all-pairs comparison. Survivors are refined via cubic spline interpolation to locate TCA analytically. $P_c$ is computed via 6D Monte Carlo (default $N=10{,}000$ samples) on the 6×6 state covariance propagated to TCA using a finite-difference State Transition Matrix [@alfano2005review].

# Research Impact Statement

ASTRA-Core is published on PyPI (`astra-core-engine`) and archived with a permanent DOI via Zenodo [@tare2026astra], enabling formal academic citation. The library is actively used by its author for Space Traffic Management research at VIT. Its self-contained dependency tree (NumPy, SciPy, Skyfield, Numba, Plotly) and fully reproducible example scripts allow direct adoption by university research groups without any proprietary installations. The repository's examples directory contains end-to-end conjunction analysis workflows executable against live CelesTrak catalog data within minutes.

# AI Usage Disclosure

The Antigravity AI assistant (Google DeepMind) was used for drafting documentation, structuring this paper, and scaffolding boilerplate code. All mathematical implementations, design decisions, and force-model derivations were authored and validated by the human author, Ishan Tare. No AI-generated content was accepted without explicit review and correction.

# Acknowledgements

The author thanks the maintainers of Skyfield, SciPy, NumPy, and Numba for their foundational open-source libraries, and CelesTrak (T. S. Kelso) and JPL for freely accessible orbital catalog data and planetary ephemerides.

# References
