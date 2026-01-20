---
layout: default
title: Training (Dev)
parent: Usage Guide
nav_order: 3
---

# Training Workflows

[< Back: Inference](inference.md) | [Next: Analysis >](analysis.md)

---

This section covers how to train the models from scratch, including the Active Learning loop. This is intended for developers or researchers reproducing the results.

## 1. Active Learning Loop

The Active Learning loop trains an ensemble of diffusion models iteratively.

### Configuration
Training is controlled by a YAML configuration file. See `configs/ala2_al.yaml` for a reference.

### Running via CLI
```bash
msgen al --config configs/ala2_al_24_hpc.yaml
```

### Running via SLURM (HPC)
For long-running jobs on a cluster:
```bash
sbatch slurm/92_train_loop_24.sh
```

## 2. Training Single Models

You can also train individual components separately.

### Diffusion Model
```bash
msgen train --config configs/ala2_default.yaml
```

### Force Field Surrogate
(See `scripts/train_pairwise.py` - currently not exposed via `msgen` CLI but available in scripts).

---

[< Back: Inference](inference.md) | [Next: Analysis >](analysis.md)
