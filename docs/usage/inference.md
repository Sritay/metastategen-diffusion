---
layout: default
title: Inference (Sampling)
parent: Usage Guide
nav_order: 2
---

# Inference Workflows

[< Back: Verification](verification.md) | [Next: Training >](training.md)

---

This section covers how to use **Pretrained Models** to sample and refine molecular structures. This assumes you have downloaded the checkpoints to `pretrained_models/`.

## 1. Sampling & Refinement

The `msgen sample` command runs the full generation pipeline:
1.  **Diffusion**: Generates coarse backbone structures using the pretrained EGNN.
2.  **Reconstruction**: Reconstructs full-atom geometry.
3.  **Refinement**: Optimizes the structure using the pretrained Force Field.

### Basic Usage

#### 1. MLIP Refinement (Force Field)
Standard mode for proteins/peptides using a trained potential.

```bash
msgen sample \
    --diff-ckpt pretrained_models/diffusion_model.pt \
    --force-ckpt pretrained_models/force_field.pt \
    --out-dir runs/test_mlip \
    --n-samples 100 \
    --refinement-steps 2000
```

#### 2. Geometric Refinement (Lignin/Topology)
Physics-free mode using topological constraints. Ideal for Lignin.
See [Lignin Generation Example](lignin_generation.md) for details.

```bash
msgen sample \
    --diff-ckpt pretrained_models/diffusion_model.pt \
    --topology data/miscanthus/L0.pdb \
    --connectivity data/miscanthus/L0.psf \
    --out-dir runs/test_geometric \
    --refinement-mode geometric \
    --n-samples 100
```

#### 3. RDKit Refinement (UFF Relaxation)
Deterministically reconstructs hydrogens and performs a full structural relaxation using the Universal Force Field (UFF) implemented in RDKit.

```bash
msgen sample \
    --diff-ckpt pretrained_models/diffusion_model.pt \
    --topology data/raw/alanine-dipeptide-nowater.pdb \
    --out-dir runs/test_rdkit \
    --refinement-mode rdkit \
    --n-samples 100
```

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--diff-ckpt` | Required | Path to the trained Diffusion Model checkpoint (`.pt`). |
| `--force-ckpt` | Optional | Path to the trained Pairwise Energy Model (Required if mode=`mlip`). |
| `--refinement-mode` | `mlip` | Refinement Strategy: `mlip` (Force Field), `geometric` (Clash Removal), or `rdkit` (UFF Relaxation). |
| `--n-samples` | 100 | Number of structures to generate. |
| `--batch-size` | 100 | Batch size for inference (adjust based on GPU memory). |
| `--refinement-steps` | 2000 | Number of gradient descent steps. |
| `--keep-percent` | 1.0 | Fraction of lowest-energy samples to keep (e.g., `0.01` keeps top 1%). |
| `--output-formats` | `['pdb', 'lammps_data']` | List of formats to save: `pdb`, `gro`, `xyz`, `lammps_dump`, `lammps_data`. |

### Output

The script creates a directory (e.g. `runs/test_sampling`) containing:
*   `refined_results.pt`: A PyTorch file containing the final `refined_positions` (Batch, 22, 3) and `initial_positions`.
*   `checkpoint_batch_*.pt`: Intermediate checkpoints.

---

[< Back: Verification](verification.md) | [Next: Training >](training.md)
