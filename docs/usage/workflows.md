---
layout: default
title: Workflows
parent: Usage Guide
nav_order: 2
---

# Core Workflows

The primary way to interact with the project is via the `msgen` Command Line Interface (CLI).

## 1. Active Learning Loop (Loop 24)
To run the latest Active Learning iteration (Loop 24):

### Via CLI (Local/Debug)
```bash
msgen al --config configs/ala2_al_24_hpc.yaml
```

### Via SLURM (HPC)
```bash
sbatch slurm/92_train_loop_24.sh
```

## 2. Refinement Loop (Loop 23)
To run the latest Refinement process (Loop 23 with fixed constraints):

### Via CLI (Local/Debug)
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

### Via SLURM (HPC)
```bash
sbatch slurm/93_refine_loop_23_fixed.sh
```
