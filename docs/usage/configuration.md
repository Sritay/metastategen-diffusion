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

This section controls data loading and preprocessing.

#### Base Parameters (Path & Topology)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `topo_path` | str | **Required** | Path to PDB/PSF for topology, connectivity, and graph inference. |
| `traj_path` | str | Optional | Path to Trajectory (NPZ, LAMMPS, GRO, XYZ, or PDB). If omitted, `topo_path` is used as a single-frame trajectory. |
| `npz_path` | str | Legacy | Alias for `traj_path` (backward compatibility). |
| `pdb_path` | str | Legacy | Alias for `topo_path` (backward compatibility). |
| `scale_factor` | float | 1.0 | Global scaling factor for coordinates (e.g. 0.1 to convert Angstrom to nm). |
| `batch_size` | int | 256 | Batch size for training/sampling. |
| `num_workers` | int | 4 | Number of DataLoader workers. |

#### Low Data Augmentation
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `augment_low_data` | bool | true | Enable thermal augmentation if dataset is small. |
| `min_aug_frames` | int | 100 | Minimum frames required to skip augmentation. |
| `aug_n_copies` | int | 1000 | Target dataset size after augmentation. |
| `aug_noise_scale` | float | 0.05 | Standard deviation of Gaussian noise added during augmentation. |

#### Pre-Splits (Advanced / AL Resume)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed_path` | str | Optional | Path to initial labeled training set (`.pt`). |
| `val_path` | str | Optional | Path to validation set (`.pt`). |
| `pool_path` | str | Optional | Path to unlabeled pool set (`.pt`). |
| `meta_path` | str | Optional | Path to metadata file (mean/std statistics). |

### `model` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `type` | str | "egnn" | Model architecture identifier. |
| `n_atom_types` | int | Auto | Number of atom types. Inferred from data if possible. |
| `n_layers` | int | 6 | Depth of the EGNN (number of layers). |
| `hidden_dim` | int | 128 | Hidden feature dimension ($h$). |
| `time_emb_dim` | int | 128 | Dimension of sinusoidal time embeddings. |
| `use_rbf` | bool | true | Use Radial Basis Functions for edge features. |
| `rbf_cutoff` | float | 10.0 | Cutoff distance for graph edges (in scaled units). |
| `use_chiral_features` | bool | true | Inject chiral volume signal (critical for enantiomers). |

### `diffusion` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `T` | int | 1000 | Total number of diffusion timesteps. |
| `schedule` | str | "cosine" | Beta schedule type: `cosine` or `linear`. |
| `beta_start` | float | 1e-4 | Starting noise variance $\beta_1$. |
| `beta_end` | float | 0.02 | Ending noise variance $\beta_T$. |
| `recenter_every_step` | bool | true | Center molecule at origin after every denoising step. |

### `train` Section

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `project` | str | - | Project name for logging (WandB/Tags). |
| `exp_id` | str | - | Unique experiment identifier. |
| `out_dir` | str | - | Directory to save checkpoints and logs. |
| `seed` | int | 42 | Global random seed. |
| `epochs` | int | 500 | Total training epochs. |
| `save_every` | int | - | Frequency (in epochs) to save checkpoints. |
| `lr` | float | 3e-4 | Learning rate (Adam optimizer). |
| `grad_clip` | float | 1.0 | Gradient clipping norm ($max\_norm$). |
| `rot_aug` | bool | true | Apply random SO(3) rotations to training data. |

### `active_learning` Section (AL Only)

Controls the iterative loop and acquisition strategy.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_iters` | int | 20 | Number of AL iterations. |
| `n_acquire` | int | 500 | Samples to label per iteration. |
| `n_candidates` | int | 5000 | Candidate pool size generated for screening. |
| `acquisition_strategy` | str | "uncertainty" | Strategy: `uncertainty` (variance) or `random`. |
| `finetune_epochs` | int | 20 | Epochs to train after each acquisition step. |
| `condition_strategy` | str | "uniform" | distribution for selecting conditioning variable (e.g. Time). |
| `oracle_device` | str | "cpu" | Device for Oracle inference (CPU saves VRAM). |
| `eval_samples` | int | 1000 | Number of samples for generation metrics. |
| `rmsd_thresh` | float | 0.15 | RMSD threshold for basin clustering metrics. |

---

## 3. CLI Argument Reference

The `msgen sample` command works differently. Instead of a YAML file, it uses direct CLI arguments for runtime flexibility.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--diff-ckpt` | Path to trained Diffusion checkpoint | **Required** |
| `--force-ckpt` | Path to trained Force/Energy checkpoint | **Required** |
| `--topology` | Path to PDB file for reconstruction | **Required** |
| `--out-dir` | Output directory for results | **Required** |
| `--n-samples` | Number of structures to generate | 1000 |
| `--batch-size` | Batch size for sampling | 100 |
| `--refinement-steps` | Steps of Langevin dynamics | 2000 |
| `--warmup-steps` | Steps with bond constraints enabled | 1000 |
| `--step-size` | Langevin step size ($\eta$) | 1e-5 |
| `--keep-percent` | Fraction of lowest energy structures to keep | 1.0 (100%) |

---

[< Back: Analysis](analysis.md) | [Next: Developer Guides >](../developers.md)
