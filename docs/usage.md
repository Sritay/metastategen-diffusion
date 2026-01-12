---
layout: default
title: Usage Guide
nav_order: 5
math: mathjax
---

# Usage Guide

[< Back: Implementation](implementation.md) | [Back to Home >](index.md)

---

## Prerequisites
*   **OS**: Linux (tested on RHEL/CentOS via HPC) or macOS.
*   **Hardware**: Capable CPU or standard consumer GPU.
    *   *Note*: The project was developed and tested on **Archer2 HPC** using **AMD MI210 GPUs** with **ROCm 5.6.0**. Use of HPC is optional as the model is lightweight (~1M parameters).
*   **Software**: Python 3.8+, PyTorch 2.0+.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Sritay/metastategen-diffusion.git
    cd metastategen-diffusion
    ```

2.  **Environment Setup**
    Create a fresh virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Package**
    Install the package in editable mode to enable the `msgen` CLI:
    ```bash
    pip install -e .
    ```
    *Note: For HPC environments (ROCm), ensure you load appropriate modules (see `slurm/pip-install.sh`) or install the PyTorch ROCm version manually.*

---

## Running the Code

You can run the project using either the **CLI** (`msgen`) or the **Scripts** (for existing SLURM workflows).

### Option A: Using the CLI (`msgen`)

The `msgen` command is the main entry point for local development and new workflows.

**1. Training Diffusion Model**
```bash
msgen train --config configs/ala2_default.yaml
```

**2. Active Learning Loop**
Runs the full AL loop (Ensemble Training -> Acquisition -> Oracle -> Retraining).
```bash
msgen al --config configs/ala2_al.yaml
```

**3. Sampling & Refinement**
Generates structures using a trained diffusion model and refines them with the pairwise force field.
```bash
msgen sample \
    --diff-ckpt runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt \
    --force-ckpt runs/energy_pairwise/best_model.pt \
    --out-dir runs/my_refined_test \
    --n-samples 1000
```

**4. Reporting**
Generates Ramachandran and Free Energy plots from dihedrals data.
```bash
msgen report --dihedrals data/processed/ala2/dihedrals.npz --outdir reports/my_report
```

---

### Option B: Python Scripts (SLURM / HPC)

For compatibility with existing `slurm/*.sh` scripts, the original python scripts in the `scripts/` directory are maintained as wrappers. These function identically to the CLI commands.

**1. Active Learning Loop**
```bash
# SLURM
sbatch slurm/92_train_loop_24.sh

# Python
python scripts/run_al_loop.py --config configs/ala2_al_24_hpc.yaml
```

**2. Refinement Loop**
```bash
# SLURM
sbatch slurm/93_refine_loop_23_fixed.sh

# Python
python scripts/sample_refined.py --diff-ckpt ... --force-ckpt ...
```

---

## Configuration

Configuration is handled via YAML files in `configs/`.

*   **`configs/ala2_default.yaml`**: Standard diffusion training.
*   **`configs/ala2_al.yaml`**: Active Learning configuration (oracle, acquisition, ensemble size).
*   **`configs/ala2_energy.yaml`**: Pairwise energy model training.

---

[< Back: Implementation](implementation.md) | [Back to Home >](index.md)
