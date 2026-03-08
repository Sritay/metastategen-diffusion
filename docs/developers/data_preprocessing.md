---
layout: default
title: Data Preprocessing
parent: Developer Guides
nav_order: 1
---

# Developer Guide: Data Preprocessing

[< Back: Developer Guide](../developers.md) | [Next: Active Learning >](al_loop.md)

This section explains how data is prepared for training in MetaStateGen. The package was designed for arbitrary molecular systems, utilizing a generalized data loading pathway, but it also retains legacy scripts for benchmarking on the Alanine Dipeptide MDShare dataset.

## 1. Generalized Data Loading (Manager)

**Location**: `src/metastategen/data/manager.py`

The modern `load_training_data` function allows the training and Active Learning loops to bypass manual preprocessing scripts for standard use cases. It supports loading arbitrary topologies and datasets.

### Logic
1.  **Input**: Accepts `traj_path` (NPZ, PDB, LAMMPS, GRO, XYZ) and `topo_path` (PDB or PSF).
2.  **Fallback**: If `traj_path` is missing, it loads `topo_path` as a single-frame trajectory (useful for Lignin/Single-Structure runs or highly limited data scenarios).
3.  **AL Integration**: `active_learning.py` uses this to load raw data and performs an **in-memory split** (Seed/Val/Pool) if the split files `.pt` don't exist yet. This significantly simplifies the user workflow by removing the need for manual dataset shard creation.

---

## 2. Legacy Alanine Dipeptide Benchmarks

For testing, benchmarking, and historical reproducibility, we include scripts to process the standard Alanine Dipeptide simulation data from `mdshare`.

### 2a. Data Download (`get_mdshare_data.py`)

**Location**: `scripts/get_mdshare_data.py`

Fetches raw trajectories from the `mdshare` repository.

#### Inputs & Outputs

| Type | Description |
| :--- | :--- |
| **Source** | `mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-positions.npz')` |
| **Output** | `data/raw/ala2/ala2.npz` |

#### Logic
1.  **Check**: If `data/raw` exists.
2.  **Fetch**: Calls `mdshare` to download the 250,000 frame numpy array.
3.  **Verify**: Checks array shape `(250000, 10, 3)`.

### 2b. Positions Selection (`preprocess_positions.py`)

**Location**: `scripts/preprocess_positions.py`

Filters and subsamples the raw MD trajectory.

#### Arguments
*   `--data`: Path to raw `.npz` file.
*   `--out`: Output `.pt` file.
*   `--stride`: Subsample frequency (default: 1).
*   `--max-frames`: Truncate to N frames.
*   `--fraction`: Randomly select float% (0.0 to 1.0) of data.

#### Logic (Function: `select_indices`)
1.  **Load**: Reads numpy array.
2.  **Conversion**: Converts to `torch.Tensor`.
3.  **Slicing**: 
    1.  Stride: `indices = indices[::stride]`
    2.  Truncate: `indices = indices[:max_frames]`
    3.  Subsample: `indices = np.random.choice(..., replace=False)`
4.  **Save**: Writes tensor to disk.

#### Output Format
*   **File**: `ala2_positions.pt`
*   **Content**: `torch.Tensor` of shape `(N, num_atoms, 3)`.

### 2c. Sharding & Splitting (`setup_al_split.py`)

**Location**: `scripts/setup_al_split.py`

Creates the Seed, Pool, and Validation sets for Active Learning benchmarking.

#### Arguments
*   `--data-dir`: Directory containing processed shards.
*   `--seed-size`: Number of frames for initial training (e.g., 100).
*   `--val-size`: Number of frames for validation (e.g., 2000).
*   `--seed-traj`: Trajectory ID to pull seed data from (0, 1, or 2).
*   `--val-traj`: Trajectory ID to pull val data from.

#### Logic
1.  **Load**: Aggregates all `.pt` shards in `data-dir`.
2.  **Filter**: Separates indices by `traj_id` (0, 1, 2 correspond to the 3 independent MD simulations).
3.  **Split**:
    *   **Seed Set**: Takes the first `seed_size` frames from `seed_traj`.
    *   **Validation Set**: Takes `val_size` random frames from `val_traj` (excluding any overlap).
    *   **Pool Set**: All remaining frames.
4.  **Save**: Dumps three separate dictionaries.

#### Output Files (`.pt` Dictionaries)
Each output file contains a dictionary with keys:
*   `positions`: `(N, num_atoms, 3)`
*   `atom_types`: `(num_atoms,)` (N, C, O, etc.)
*   `traj_id`: `(N,)`
*   `phi_psi`: `(N, 2)` (optional)

| File | Role | Typical Size |
| :--- | :--- | :--- |
| `al_seed.pt` | Initial Training Data | 100 - 5000 |
| `al_val.pt` | Metrics & Early Stopping | 2000 - 20000 |
| `al_pool_ref.pt` | Oracle Search Space | ~200,000 |
