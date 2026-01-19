---
layout: default
title: Analysis Pipeline
parent: Developer Guides
nav_order: 4
---

# Developer Guide: Analysis Pipeline

[< Back: Refinement Loop](refinement.md)

This section explains the post-processing scripts used to verify model quality.

## 1. Density Evolution (`viz_density.py`)

**Location**: `scripts/analysis/viz_density.py`

Visualizes how the model's learned distribution $P(x)$ changes over AL iterations.

**Logic**:
*   **Input**: `runs/al_loop_...` directory.
*   **Process**:
    *   Iterates through each AL checkpoint.
    *   Generates a batch of positions (Diffusion Sampling).
    *   Calculates Phi/Psi backbone torsion angles.
    *   Bins them into a 2D histogram (Ramachandran plot).
*   **Metric**: Computes KL Divergence between the Generated Histogram and the Ground Truth (MD) Histogram.
*   **Output**: `evolution_density.png` (Grid of plots).

## 2. Refinement Funnel (`viz_funnel.py`)

**Location**: `scripts/analysis/viz_funnel.py`

Visualizes the "funneling" effect of the refinement loop.

**Logic**:
*   **Input**: `runs/refinement_...` directory containing `initial_structures.pt` and `refined_structures.pt`.
*   **Process**:
    *   Compute Phi/Psi for `initial` (Diffusion output).
    *   Compute Phi/Psi for `refined` (Force Field output).
    *   Plot vectors/arrows connecting Initial -> Refined points on the Ramachandran plane.
*   **Insight**: Shows if the force field correctly "snaps" rough guesses into the nearest physical basin (Alpha/Beta/C7eq).

## 3. Cluster Analysis (`analyze_clusters.py`)

**Location**: `scripts/analysis/analyze_clusters.py`

Quantifies population ratios.

**Logic**:
*   Definitions of basins (e.g., Alpha is $\phi \in [-100, -50], \psi \in [-60, -30]$).
*   Counts percentage of samples falling into each defined region.
*   Compares against Boltzmann weights from ground truth MD.
