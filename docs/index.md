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
This project builds a Machine Learning model capable of predicting metastable states for molecular systems. 
We currently focus on **Alanine Dipeptide** as a standard benchmark for evaluating generative models of metastable dynamics.

We combine **Geometric Deep Learning (EGNN)** for capturing molecular symmetries with **Active Learning** to iteratively explore and learn the conformational landscape efficiently.

<div align="center">
  <img src="assets/movie.gif" alt="Generated Structures" width="60%">
  <br>
  <em><strong>Figure 1: Generation in Action.</strong> The diffusion model generates diverse 10-atom backbones which are then reconstructed and refined into full all-atom structures.</em>
</div>

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

### [3. Implementation Details](implementation.md)
*   **Model Architecture**: EGNN configurations, Chirality, and Pairwise RBFs.
*   **Datasets**: MDShare vs Timewarp.

### [4. Usage Guide](usage.md)

*   **Installation**: Setup and Data Download.
*   **Verification**: Fast sanity checks.
*   **Workflows**: Training, Active Learning, and Refinement.
*   **Analysis**: Visualization and metrics.
*   **Configuration**: Full reference.

### [5. Developer Guides](developers_al_loop.md)

*   **[AL Loop Internals](developers_al_loop.md)**: Logic flow of the Active Learning loop.
*   **[Refinement Internals](developers_refinement.md)**: Deep dive into the sampling and reconstruction pipeline.

---

## Quick Overview

### The Problem
Molecular Dynamics (MD) is expensive. We want to generate diverse, low-energy molecular states without running simulation for weeks.

### The Solution
1.  **Backbone Diffusion**: A diffusion model learns to generate the heavy-atom backbone roughly.
2.  **Active Learning**: We don't just train once. We train, generate, find uncertainty, label it, and retrain. This pushes the model into new regions.
3.  **Refinement**: We use a fast surrogate model (Pairwise Force Field) to relax the rough backbones into perfect physical structures.

### Future Directions
*   **Oracle Expansion**: Integration with AIMD/CP2K for ground truth energy evaluation on larger systems.
*   **Generalization**: Extending the pipeline to multi-molecule systems and larger peptides.

---

[Next: Active Learning >](active_learning.md)
