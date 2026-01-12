# MetaStateGen Diffusion

Welcome to the **MetaStateGen Diffusion** project documentation.

## Project Goal
This project aims to generate physically valid, low-energy metastable states of peptides (focusing on Alanine Dipeptide) using a novel **Active Learning + Diffusion** approach.

We combine **Geometric Deep Learning (EGNN)** for capturing molecular symmetries with **Active Learning** to iteratively explore the conformational landscape.

---

## 📚 Documentation Contents

### [1. Methodology](methodology.md)
*   **Active Learning Loop**: How we iteratively train the ensemble.
*   **Refinement Loop**: How we convert backbones to full atoms.
*   **Scripts**: Detailed breakdown of `run_al_loop.py` and `sample_refined.py`.

### [2. Implementation Details](implementation.md)
*   **Model Architecture**: EGNN configurations and Hyperparameters.
*   **Datasets**: MDShare vs Timewarp.
*   **Design Decisions**: Why RBF? Why Constraints?

### [3. Usage Guide](usage.md)
*   **Installation**: Getting started.
*   **Running Training**: How to launch SLURM jobs.
*   **Running Sampling**: Generating your own structures.

---

## Quick Overview

### The Problem
Molecular Dynamics (MD) is expensive. We want to generate diverse, low-energy molecular states without running simulation for weeks.

### The Solution
1.  **Backbone Diffusion**: A diffusion model learns to generate the heavy-atom backbone roughly.
2.  **Active Learning**: We don't just train once. We train, generate, find uncertainty, label it, and retrain. This pushes the model into new regions.
3.  **Refinement**: We use a fast surrogate model (Pairwise Force Field) to relax the rough backbones into perfect physical structures.

---

[Next: Methodology >](methodology.md)
