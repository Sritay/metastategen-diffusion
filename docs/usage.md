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

    # Install dependencies (ROCm/CUDA version may vary)
    # See slurm/pip-install.sh for exact versions used on HPC
    pip install "numpy<2" scipy matplotlib pyyaml tqdm rich mdshare pandas wandb
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
    
    # Install Package (Editable mode)
    pip install -e .
    ```

---

## Running the Code

### 1. Active Learning Loop (Loop 24)
To run the latest Active Learning iteration (Loop 24):

**Via SLURM (HPC):**
```bash
sbatch slurm/92_train_loop_24.sh
```

**Via Python (Local/Debug):**
```bash
python scripts/run_al_loop.py --config configs/ala2_al_24_hpc.yaml
```

### 2. Refinement Loop (Loop 23)
To run the latest Refinement process (Loop 23 with fixed constraints):

**Via SLURM (HPC):**
```bash
sbatch slurm/93_refine_loop_23_fixed.sh
```

**Via Python (Local/Debug):**
```bash
python scripts/sample_refined.py \
    --diff-ckpt runs/day11_al_23_hpc/members/m000/checkpoints/curr_ckpt.pt \
    --force-ckpt runs/energy_pairwise/best_model.pt \
    --out-dir runs/loop_b_refinement_23_fixed \
    --n-samples 1000
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



---

## Configuration

Configuration is handled via YAML files in `configs/`.

*   **`configs/ala2_default.yaml`**: Standard diffusion training.
*   **`configs/ala2_al.yaml`**: Active Learning configuration (oracle, acquisition, ensemble size).
*   **`configs/ala2_energy.yaml`**: Pairwise energy model training.

---

[< Back: Implementation](implementation.md) | [Back to Home >](index.md)
