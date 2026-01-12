# Usage Guide

## Prerequisites
-   Linux environment with SLURM scheduler (for HPC execution).
-   Python 3.10+
-   CUDA-enabled GPU.

## Installation
Dependencies instructions are generally handled via the provided `pip-install.sh` or `conda` environment setup.

## Running the Loops

### 1. Active Learning (Training)
To start or continue an Active Learning loop, use the scripts in `slurm/`.
**Example**: Running Loop 24
```bash
sbatch slurm/92_train_loop_24.sh
```
*   **What it does**: Launches the AL script `scripts/run_al_loop.py` with `configs/ala2_al_24_hpc.yaml`.

### 2. Refinement (Sampling)
To refine generated structures using the trained models.
**Example**: Running Refinement for Loop 23
```bash
sbatch slurm/93_refine_loop_23_fixed.sh
```
*   **What it does**: Launches `scripts/sample_refined.py`.
*   **Key Flags**:
    *   `--diff-ckpt`: Path to the diffusion model checkpoint.
    *   `--force-ckpt`: Path to the pairwise force model.
    *   `--n-samples`: Number of structures to generate.
    *   `--keep-percent`: Top % of structures to keep after energy filtering.

## Output
Results are typically saved in the `runs/` directory:
-   `runs/loop_X_.../refined_results.pt`: Contains `initial_positions` (Diffusion) and `refined_positions` (Final).
