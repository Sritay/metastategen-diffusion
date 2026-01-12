# MetaStateGen Diffusion

Welcome to the **MetaStateGen Diffusion** project documentation.

## Project Goal
This project aims to generate physically valid, low-energy metastable states of peptides (focusing on Alanine Dipeptide) using a novel **Active Learning + Diffusion** approach.

We combine:
1.  **Geometric Deep Learning (EGNN)** for capturing molecular symmetries.
2.  **Denoising Diffusion Probabilistic Models (DDPM)** for generative sampling.
3.  **Active Learning Loops** to iteratively explore the conformational landscape.
4.  **Physics-Guided Refinement** to relax generated structures into true energy minima.

## Key Features
-   **Active Learning**: The model learns from its own uncertainty, querying an "Oracle" (Ground Truth MD) to label high-uncertainty regions.
-   **Two-Stage Generation**:
    1.  **Diffusion (Backbone)**: Generates the coarse 10-atom backbone structure.
    2.  **Refinement (All-Atom)**: Reconstructs the 22-atom system and relaxes it using a learned Pairwise Force Field.
-   **Chirality Awareness**: explicit geometric features to distinguish between enantiomers (L-Ala vs D-Ala).

## Navigate
-   [Methodology](./methodology.md): How the Active Learning and Refinement loops work.
-   [Implementation](./implementation.md): Details on Model Architecture, Datasets, and constraints.
-   [Usage](./usage.md): Instructions for running training and sampling.
