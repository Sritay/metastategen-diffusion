---
layout: default
title: Configuration
parent: Usage Guide
nav_order: 5
---

# Configuration Reference

[< Back: Analysis](analysis.md) | [Next: Developer Guides >](../developers.md)

---

MetaStateGen uses YAML configuration files to control Training and Active Learning. This page details every available parameter, organized by section.

## 1. Top-Level Sections

A standard config file (`configs/*.yaml`) is divided into these blocks:

| Section | Purpose |
| :--- | :--- |
| **`data`** | Dataset paths, batch sizes, and preprocessing options. |
| **`model`** | Architecture hyperparameters (e.g., EGNN depth). |
| **`diffusion`** | Noise schedule and timestep settings. |
| **`train`** | Optimization settings (Learning Rate, Epochs). |
| **`ensemble`** | (AL Only) Ensemble size and seed management. |
| **`active_learning`** | (AL Only) Iteration loop and acquisition parameters. |

---

## 2. Parameter Reference

### `data` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed_path` | str | Required | Path to the initial labeled dataset (`.pt`). |
| `val_path` | str | Required | Path to the validation dataset (`.pt`). |
| `pool_path` | str | Required | Path to the unlabeled pool (`.pt`). |
| `batch_size` | int | 256 | Batch size for training. |
| `num_workers` | int | 4 | DataLoader workers (use 0 for debugging). |
| `scale_factor` | float | 1.0 | Global scaling factor for coordinates (e.g. `7.6`). |

### `model` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `type` | str | "egnn" | Model architecture. |
| `n_layers` | int | 6 | Number of EGNN layers. |
| `hidden_dim` | int | 128 | Feature dimension size ($h$). |
| `time_emb_dim` | int | 128 | Dimension for time embeddings. |
| `use_rbf` | bool | true | Use Radial Basis Functions for edge features. |
| `rbf_cutoff` | float | 10.0 | Interaction radius (Angstroms likely, verify units). |
| `use_chiral_features` | bool | true | Inject chiral volume signal? (Critical for Ala2). |

### `diffusion` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `T` | int | 1000 | Number of diffusion timesteps. |
| `schedule` | str | "cosine" | Noise schedule type (`cosine`, `linear`). |
| `beta_start` | float | 1e-4 | Start noise variance. |
| `beta_end` | float | 0.02 | End noise variance. |
| `recenter_every_step` | bool | true | Force Center-of-Mass to origin after every step? |

### `train` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed` | int | 42 | Random seed for reproducibility. |
| `epochs` | int | 500 | Training epochs (or initial cold-start epochs for AL). |
| `lr` | float | 3e-4 | Learning rate (Adam). |
| `grad_clip` | float | 1.0 | Gradient clipping norm. |
| `rot_aug` | bool | true | Apply random SO(3) rotations during training? |

### `ensemble` Section (Active Learning)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `members` | int | 3 | Number of independent models in the ensemble. |
| `seeds` | list[int] | auto | Custom seeds for each member (e.g. `[111, 222, 333]`). |

### `active_learning` Section (Active Learning)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_iters` | int | 20 | Number of AL iterations. |
| `n_acquire` | int | 500 | Number of samples to label per iteration. |
| `n_candidates` | int | 5000 | Number of candidates to generate for screening. |
| `acquisition_strategy` | str | "uncertainty" | Strategy: `uncertainty` or `random`. |
| `finetune_epochs` | int | 20 | Epochs to train after each acquisition. |
| `oracle_device` | str | "cpu" | Device for Oracle NN-search (use CPU to save VRAM). |

---

## 3. CLI Argument Reference

The `msgen sample` command works differently. Instead of a YAML file, it uses direct CLI arguments for runtime flexibility.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--diff-ckpt` | Path to trained Diffusion checkpoint | **Required** |
| `--force-ckpt` | Path to trained Force/Energy checkpoint | **Required** |
| `--out-dir` | Output directory for results | **Required** |
| `--n-samples` | Number of structures to generate | 1000 |
| `--batch-size` | Batch size for sampling | 100 |
| `--refinement-steps` | Steps of Langevin dynamics | 2000 |
| `--warmup-steps` | Steps with bond constraints enabled | 1000 |
| `--step-size` | Langevin step size ($\eta$) | 1e-5 |
| `--keep-percent` | Fraction of lowest energy structures to keep | 1.0 (100%) |

---

[< Back: Analysis](analysis.md) | [Next: Developer Guides >](../developers.md)
