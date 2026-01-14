---
layout: default
title: Configuration
parent: Usage Guide
nav_order: 4
---

# Configuration Reference

Key parameters in `configs/*.yaml`:

## Active Learning (`ala2_al_24_hpc.yaml`)
*   `active_learning.n_iters`: Number of AL iterations (Default: 20).
*   `active_learning.n_acquire`: Candidates to label per iteration (Default: 500).
*   `active_learning.acquisition_strategy`: Strategy to select candidates (e.g., `uncertainty`).
*   `model.rbf_cutoff`: Cutoff distance for EGNN edges (Default: 10.0).
*   `train.finetune_epochs`: Epochs to retrain per iteration (Default: 20).

## Diffusion (`ala2_default.yaml`)
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

## Custom Data Layouts

If your seed data or checkpoints are in non-standard locations, you can override them via CLI or Config:

### Active Learning Overrides
```bash
msgen al --config configs/your_config.yaml \
    "data.seed_path=/path/to/custom/seed.pt" \
    "data.pool_path=/path/to/custom/pool.pt"
```

### Refinement Overrides
Simply point the CLI arguments to your specific files:
```bash
msgen sample \
    --diff-ckpt /path/to/my/model.pt \
    --force-ckpt /path/to/my/force.pt ...
```
