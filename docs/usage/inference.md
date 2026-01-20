---
layout: default
title: Inference (Sampling)
parent: Usage Guide
nav_order: 2
---

# Inference Workflows

This section covers how to use **Pretrained Models** to sample and refine molecular structures. This assumes you have downloaded the checkpoints to `pretrained_models/`.

## 1. Sampling & Refinement

The `msgen sample` command runs the full generation pipeline:
1.  **Diffusion**: Generates coarse backbone structures using the pretrained EGNN.
2.  **Reconstruction**: Reconstructs full-atom geometry.
3.  **Refinement**: Optimizes the structure using the pretrained Force Field.

### Basic Usage

```bash
msgen sample \
    --diff-ckpt pretrained_models/diffusion_model.pt \
    --force-ckpt pretrained_models/force_field.pt \
    --out-dir runs/test_sampling \
    --n-samples 100 \
    --refinement-steps 2000
```

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--diff-ckpt` | Required | Path to the trained Diffusion Model checkpoint (`.pt`). |
| `--force-ckpt` | Required | Path to the trained Pairwise Energy Model checkpoint (`.pt`). |
| `--n-samples` | 100 | Number of structures to generate. |
| `--batch-size` | 100 | Batch size for inference (adjust based on GPU memory). |
| `--refinement-steps` | 2000 | Number of gradient descent steps for energy minimization. Use `50000` for high-quality refinement. |
| `--keep-percent` | 1.0 | Fraction of lowest-energy samples to keep (e.g., `0.01` keeps top 1%). |

### Output

The script creates a directory (e.g. `runs/test_sampling`) containing:
*   `refined_results.pt`: A PyTorch file containing the final `refined_positions` (Batch, 22, 3) and `initial_positions`.
*   `checkpoint_batch_*.pt`: Intermediate checkpoints.
