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
  topo_path: data/timewarp/AACG/AACG-traj-state0.pdb
  traj_path: data/timewarp/AACG/AACG-traj-arrays.npz # Optional (can be LAMMPS/GRO/XYZ/PDB)
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

### Low Data Training
For systems with very limited data (e.g., a single PDB frame of a lignin polymer), MetaStateGen can automatically augment the data with thermal noise and rotations.
You only need to provide the topology file; it will be used as the single-frame trajectory automatically.

```yaml
data:
  # Single frame input (PDB-only)
  topo_path: data/lignin/L0.pdb
  # traj_path: Omitted (Implies use topo_path as 1-frame trajectory)
  
  # Optional: Separate topology file (if PDB lacks bonds)
  # topology_path: data/lignin/L0.psf
  
  # Augmentation Controls
  augment_low_data: true
  min_aug_frames: 100
  aug_n_copies: 1000      # Replicate 1 frame -> 1000 frames
  aug_noise_scale: 0.05   # Add 0.05 Angstrom thermal jitter
```

#### How Low Data Augmentation Works
When `augment_low_data: true` is enabled and the input dataset is smaller than `min_aug_frames`:
1.  **Replication**: The available frames are duplicated to reach `aug_n_copies` (e.g., 1000).
2.  **Thermal Jitter**: Independent Gaussian noise ($\sigma=$ `aug_noise_scale`) is added to each copy. This simulates thermal fluctuations around the equilibrium structure.
3.  **Rotation**: During training, `train.rot_aug: true` applies random SO(3) rotations to every batch, ensuring the model learns strict rotational invariance even from a single static input.

---

## 2. Active Learning Loop

Use this mode to iteratively discover new metastable states. The loop trains an **ensemble** of models, generates candidate structures, estimates uncertainty, and queries an oracle (ground truth pool) to acquire new training data.

### Features
*   **Ensemble Training**: Trains multiple models (seeds) to estimate epistemic uncertainty.
*   **Auto-Splitting**: Automatically splits a raw dataset (PDB, NPZ, or LAMMPS) into Seed (initial), Pool (oracle), and Validation sets.
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
  # Input for Auto-Splitting (if splits don't exist)
  topo_path: data/processed/ala2.pdb
  traj_path: data/processed/ala2.npz
  
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
