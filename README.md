# MetaStateGen-Diffusion

Metastable state generation with equivariant diffusion for molecular dynamics.
**[📘 Full Documentation Website](https://sritay.github.io/metastategen-diffusion/)**

> [!IMPORTANT]
> **Project Status**: This research software is currently under active development. Features, APIs, and documentation are undergoing frequent updates and are subject to change.

## Overview

This project aims to generate phsycially valid, low-energy metastable states of peptides (focusing on Alanine Dipeptide) using a novel **Active Learning + Diffusion** approach.

<div align="center">
  <img src="docs/assets/movie.gif" alt="Generated Structures" width="60%">
  <br>
  <em><strong>Figure 1: Generation in Action.</strong> The diffusion model generates diverse 10-atom backbones which are then reconstructed and refined into full all-atom structures.</em>
</div>

We combine **Geometric Deep Learning (EGNN)** for capturing molecular symmetries with **Active Learning** to iteratively explore the conformational landscape. It uses E(3)-equivariant networks and a two-stage active learning pipeline:

1.  **Loop A (Active Learning):** Trains a committee of diffusion models on `mdshare` data to identify uncertainty and query an Oracle.
2.  **Loop B (Refinement):** Uses a pairwise force surrogate (trained on TimeWarp data) to refine generated structures via Langevin dynamics, ensuring physical plausibility.

## Installation
Please refer to the **[Installation Guide](https://sritay.github.io/metastategen-diffusion/usage/installation.html)** on our documentation website for detailed setup instructions, including environment creation and data preparation.

## Usage

Please refer to the **[Usage Guide](https://sritay.github.io/metastategen-diffusion/usage.html)** on our documentation website for detailed training and sampling instructions.

## Structure

- `src/metastategen/`: Core python package (Models, Data, Oracles).
- `scripts/`: CLI scripts for training and sampling.
- `configs/`: YAML configuration files.
- `slurm/`: Example SLURM submission scripts.

## License

MIT
