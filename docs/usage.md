# Usage Guide

[< Back to Implementation](implementation.md) | [Back to Home >](index.md)

---

## Prerequisites
*   **OS**: Linux (tested on RHEL/CentOS via HPC).
*   **Hardware**: CUDA GPU (A100/H100 recommended for training).
*   **Software**: Python 3.10+, PyTorch 2.0+.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Sritay/metastategen-diffusion.git
    cd metastategen-diffusion
    ```

2.  **Install Dependencies**
    Use the provided script to set up a `venv` or `conda` env.
    ```bash
    bash slurm/pip-install.sh
    ```

---

## Running the Code

### 1. Active Learning Loop
The AL loop is managed by `run_al_loop.py`. The standard way to run this on a cluster is via SLURM.

**Command**:
```bash
sbatch slurm/92_train_loop_24.sh
```

**What happens**:
1.  The script invokes `scripts/run_al_loop.py`.
2.  It loads the config `configs/ala2_al_24_hpc.yaml`.
3.  It cycles through generic -> sample -> label -> train loops.
4.  Logs are written to `runs/al_loop_.../`.

### 2. Refinement Loop
After generating backbones, refine them using the Pairwise model.

**Command**:
```bash
sbatch slurm/93_refine_loop_23_fixed.sh
```

**Key Arguments**:
*   `--diff-ckpt`: Path to the trained diffusion model (from AL loop).
*   `--force-ckpt`: Path to the force surrogate model (usually trained separately).
*   `--refinement-steps`: How long to relax (e.g., 50,000).

**Output**:
*   Check `runs/loop_b_refinement_.../` for `refined_results.pt`.

---

[< Back to Implementation](implementation.md) | [Back to Home >](index.md)
