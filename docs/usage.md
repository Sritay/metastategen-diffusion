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
    We recommended using `venv` or `conda` with **Python 3.10+**.

    ```bash
    # Create environment
    python -m venv .venv
    source .venv/bin/activate

    # Upgrade pip (Required for pyproject.toml support)
    pip install --upgrade pip

    # Install Package (Editable mode)
    # This automatically installs dependencies (numpy, torch, etc.) defined in pyproject.toml
    pip install -e .
    ```

    *> [!NOTE]*
    *> For **HPC / ROCm** environments, you may need to install a specific PyTorch version *before* running the above command to avoid overwriting it with the PyPI default. See `slurm/pip-install.sh` for reference.*

---

## Running the Code

The primary way to interact with the project is via the `msgen` Command Line Interface (CLI).

### 1. Active Learning Loop (Loop 24)
To run the latest Active Learning iteration (Loop 24):

**Via CLI (Local/Debug):**
```bash
msgen al --config configs/ala2_al_24_hpc.yaml
```

**Via SLURM (HPC):**
```bash
sbatch slurm/92_train_loop_24.sh
```

### 2. Refinement Loop (Loop 23)
To run the latest Refinement process (Loop 23 with fixed constraints):

**Via CLI (Local/Debug):**
```bash
msgen sample \
    --diff-ckpt runs/day11_al_23_hpc/members/m000/checkpoints/curr_ckpt.pt \
    --force-ckpt runs/energy_pairwise/best_model.pt \
    --out-dir runs/loop_b_refinement_23_fixed \
    --n-samples 1000 \
    --batch-size 100 \
    --warmup-steps 1000 \
    --refinement-steps 50000 \
    --keep-percent 0.01 
```

**Via SLURM (HPC):**
```bash
sbatch slurm/93_refine_loop_23_fixed.sh
```

## Analysis & Visualization

We provide a suite of scripts in `scripts/analysis/` to verify the active learning and refinement results.

### 1. AL Density Evolution
visualize how the model's generated distribution covers the Ramachandran landscape over iterations.

```bash
python scripts/analysis/viz_density.py \
    --run runs/day11_al_23_hpc \
    --iters 0,5,10,15,20 \
    --outdir analysis_output/al_density
```
*Outputs: `evolution_density.png` (Backbone P(x)), `evolution_data.csv`*

### 2. Cluster Analysis
Identify and count populations in the metastable basins (Alpha, Beta, C7eq, etc.).

```bash
python scripts/analysis/analyze_clusters.py \
    --data analysis_output/al_density/evolution_data.csv \
    --outdir analysis_output/clusters
```
*Outputs: `cluster_analysis.png` and summary CSVs.*

### 3. Refinement Funnel Plot
Visualize the "snapping" of coarse backbone structures into physical minima.

```bash
python scripts/analysis/viz_funnel.py \
    --run runs/loop_b_refinement_23_fixed \
    --outdir analysis_output/refinement
```
*Outputs: `funnel_plot.png` (Initial vs Refined overlay).*

---

## Configuration Reference

Key parameters in `configs/*.yaml`:

### Active Learning (`ala2_al_24_hpc.yaml`)
*   `active_learning.n_iters`: Number of AL iterations (Default: 20).
*   `active_learning.n_acquire`: Candidates to label per iteration (Default: 500).
*   `active_learning.acquisition_strategy`: Strategy to select candidates (e.g., `uncertainty`).
*   `model.rbf_cutoff`: Cutoff distance for EGNN edges (Default: 10.0).
*   `train.finetune_epochs`: Epochs to retrain per iteration (Default: 20).

### Diffusion (`ala2_default.yaml`)
*   `diffusion.T`: Total diffusion steps (Default: 1000).
*   `diffusion.schedule`: Noise schedule (e.g., `cosine`).

## CLI Reference (`msgen sample`)

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--diff-ckpt` | Path to trained Diffusion checkpoint | Required |
| `--force-ckpt` | Path to trained Force/Energy checkpoint | Required |
| `--refinement-steps` | Number of Langevin dynamics steps | 2000 |
| `--warmup-steps` | Steps with bond constraints enabled | 1000 |
| `--step-size` | Langevin step size | 1e-5 |
| `--keep-percent` | Fraction of lowest energy structures to keep | 1.0 (100%) |

---

[< Back: Implementation](implementation.md) | [Back to Home >](index.md)
