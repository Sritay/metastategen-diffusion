---
layout: default
title: Home
nav_order: 1
has_children: false
---

# MetaStateGen Diffusion



---

Welcome to the **MetaStateGen Diffusion** project documentation.

> [!IMPORTANT]
> **Project Status**: This research software is currently under active development. Features, APIs, and documentation are undergoing frequent updates and are subject to change.

## Project Goal
This project builds a Machine Learning model capable of predicting metastable states for generic molecular systems. Our framework readily accepts single structures or full trajectories for any system with a valid input topology. 
We demonstrate the model's capabilities using **Alanine Dipeptide** as a benchmark for evaluating generation of metastable dynamics, as well as **Lignin** as a complex flexible system.

We combine **Geometric Deep Learning (EGNN)** for capturing molecular symmetries with **Active Learning** to iteratively explore and learn the conformational landscape efficiently.

<div align="center">
  <div style="display: flex; justify-content: center; gap: 20px;">
    <div style="text-align: center;">
      <img src="assets/movie.gif" alt="Alanine Dipeptide" width="300">
      <br>
      <em>Alanine Dipeptide</em>
    </div>
    <div style="text-align: center;">
      <img src="assets/lignin_diverse.gif" alt="Lignin Molecule" width="300">
      <br>
      <em>Lignin Molecule</em>
    </div>
  </div>
  <br>
  <em><strong>Figure 1: Generation in Action.</strong> The diffusion model generates diverse structures for both rigid peptide backbones and flexible aromatic polymers (generated from a single frame; see <a href="usage/training.html#low-data-training">Low Data Training</a>).</em>
</div>

---

## Introduction

### Motivation
Molecular Dynamics (MD) simulations are computationally expensive, particularly for sampling rare events and metastable states. This project addresses the challenge of exploring diverse, low-energy conformational basins without the prohibitive cost of long-timescale simulations.

### Methodology
1.  **EGNN-based Denoising Diffusion**: An E(3)-equivariant Graph Neural Network (EGNN) learns to reverse a diffusion process, approximating the **underlying conformational distribution** of the heavy-atom backbone.
2.  **Active Learning**: An iterative acquisition strategy uses ensemble uncertainty to guide the model towards unexplored regions of the energy landscape, efficiently covering metastable basins.
3.  **Refinement**: A physics-based **Pairwise Energy Surrogate** relaxes the generated backbones via Langevin dynamics to specific local minima, ensuring physical validity and correct geometry.
4.  **Generalization**: Support for arbitrary topologies via graph-based inference.

### Future Directions
*   **Oracle Expansion**: Integration with AIMD/CP2K for ground truth evaluation on larger systems.

---

## 📚 Documentation Contents

### [1. Active Learning Loop](active_learning.md)
*   **Ensemble Training**: Diversity through random seeds.
*   **Acquisition**: Using uncertainty (variance) to find new states.
*   **Oracle**: Snapping hallucinated candidates to valid ground truth.

### [2. Refinement Loop](refinement.md)
*   **Reconstruction**: Converting 10-atom backbones to 22-atom all-atom structures.
*   **Geometric Correction**: Using bond constraints and warm-up.
*   **Energy Relaxation**: Using the Pairwise Force Field.



### [3. Usage Guide](usage.md)

*   **Installation**: Setup and Data Download.
*   **Verification**: Fast sanity checks.
*   **Inference**: Using pretrained models.
*   **[Lignin Generation Example](usage/lignin_generation.md)**: conformer generation for flexible molecules.
*   **Training**: Active Learning and Dev workflows.
*   **Analysis**: Visualization and metrics.
*   **Configuration**: Full reference.

### [4. Developer Guides](developers.md)

*   **[Architecture & Theory](developers/architecture.md)**: Logic, Mathematical Foundations, and Dataset Specifications.
*   **Data Preprocessing**: Script logic for `get_mdshare_data` and `setup_al_split`.
*   **Active Learning Loop**: Deep dive into `run_active_learning` and ensemble training.
*   **Refinement Loop**: Internals of `sample_refined` and reconstruction algorithms.
*   **Analysis Pipeline**: Implementation details for `viz_density`, `viz_funnel`, and cluster analysis.

## Key Features

*   **Geometric Deep Learning**: Uses E(3)-equivariant Graph Neural Networks (EGNN) to learn 3D molecular distributions.
*   **Active Learning**: Iteratively explores the energy landscape using ensemble uncertainty to discover new metastable states.
*   **Arbitrary Molecule Support**: Automated topology inference for peptides, polymers (e.g., Lignin), and small molecules.
*   **Ring Constraints**: Explicitly enforces planarity and rigidity for cyclic systems (e.g., benzene) during generation.
*   **Low Data Regime**: Capable of training from sparse data (even **single structures**) using physics-based thermal augmentation.
*   **Flexible Inputs**: Works with **PDB-only** (single frame), or trajectories in **NPZ, LAMMPS, GRO, and XYZ** formats. Minimum requirement is just a single PDB file (must contain `CONECT` records).



[Next: Active Learning >](active_learning.md)
