---
layout: default
title: Training (Dev)
parent: Usage Guide
nav_order: 3
---

# Training Workflows

[< Back: Inference](inference.md) | [Next: Analysis >](analysis.md)

---

This section covers the two main training modes available in MetaStateGen: **Standard Training** (for fixed datasets) and **Active Learning** (for iterative discovery).

---

## 1. Standard Training (Single Model)

Use this mode to train a single diffusion model on a fixed dataset (e.g., an existing MD trajectory). This is useful for establishing baselines or training on data where active learning is not required.

### Features
*   **Generalized Input**: Supports any molecular system via `.npz` (positions) and `.pdb` (topology) files.
*   **Topology Inference**: Automatically detects atom types, bonds, and chiral centers from the PDB.
*   **Single Model**: Trains one EGNN-Diffusion model (no ensemble overhead).

### Running via CLI
```bash
msgen train --config configs/my_training_config.yaml
```

### Configuration Example
Create a YAML config (e.g., `configs/train_aacg.yaml`):

```yaml
train:
  seed: 42
  out_dir: runs/train_aacg
  epochs: 100
  save_every: 10
  lr: 3e-4

data:
  # Universal Data Loader inputs
  npz_path: data/timewarp/AACG/AACG-traj-arrays.npz
  pdb_path: data/timewarp/AACG/AACG-traj-state0.pdb
  scale_factor: 1.0
  batch_size: 64

model:
  n_layers: 4
  hidden_dim: 128
  time_emb_dim: 32
diffusion:
  T: 500
  beta_start: 1e-4
  beta_end: 0.02
```

---

## 2. Active Learning Loop

Use this mode to iteratively discover new metastable states. The loop trains an **ensemble** of models, generates candidate structures, estimates uncertainty, and queries an oracle (ground truth pool) to acquire new training data.

### Features
*   **Ensemble Training**: Trains multiple models (seeds) to estimate epistemic uncertainty.
*   **Auto-Splitting**: Automatically splits a raw dataset into Seed (initial), Pool (oracle), and Validation sets.
*   **Iterative Loop**: Sampling $\to$ Acquisition $\to$ Oracle $\to$ Retraining.

### Running via CLI
```bash
msgen al --config configs/ala2_al.yaml
```

### Configuration Example
The config requires an `active_learning` block and an `ensemble` block:

```yaml
active_learning:
  exp_id: al_experiment
  n_iters: 10
  n_candidates: 2000
  n_acquire: 100
  acquisition_strategy: uncertainty
  # Auto-split parameters
  initial_seed_size: 100
  val_size: 2000

ensemble:
  members: 4  # Number of models in ensemble

data:
  npz_path: data/processed/ala2.npz
  pdb_path: data/processed/ala2.pdb
  # Paths where splits will be saved/loaded
  seed_path: runs/al_experiment/splits/seed.pt
  pool_path: runs/al_experiment/splits/pool.pt
  val_path: runs/al_experiment/splits/val.pt

# ... (standard train/model/diffusion blocks same as above)
```

---

## 3. Training Auxiliary Models

### Force Field Surrogate
Train the pairwise energy model using the generalized trainer:
```bash
python scripts/train_pairwise.py --data-source data/my_molecule.npz --out_dir runs/my_energy_model
```

---

[< Back: Inference](inference.md) | [Next: Analysis >](analysis.md)
