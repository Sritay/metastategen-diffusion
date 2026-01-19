---
layout: default
title: Data Preprocessing
parent: Developer Guides
nav_order: 1
---

# Developer Guide: Data Preprocessing

[< Back: Developer Guide](../developers.md) | [Next: Active Learning >](al_loop.md)

This section explains the scripts responsible for preparing MD data for training.

## 1. Data Download (`get_mdshare_data.py`)

**Location**: `scripts/get_mdshare_data.py`

Fetches raw trajectories from the `mdshare` repository.
*   **Input**: None (Downloads from URL).
*   **Output**: `data/raw/ala2/ala2.npz` (Positions).

## 2. Positions Selection (`preprocess_positions.py`)

**Location**: `scripts/preprocess_positions.py`

Filters and subsamples the raw MD trajectory.

**Core Function**: `select_indices`
*   **Input**: Raw trajectory array `(N, n_atoms, 3)`.
*   **Logic**:
    *   Slices by stride (e.g., every 10th frame).
    *   Filters by max frames (e.g., first 100k).
    *   Optionally selects a random subset.
*   **Output**: `data/processed/ala2/ala2_positions.pt`.

## 3. Sharding & Splitting (`setup_al_split.py`)

**Location**: `scripts/setup_al_split.py`

Creates the Seed, Pool, and Validation sets for Active Learning.

**Key Logic**:
*   **Seed**: A small, diverse subset (initially labeled) used to cold-start the AL loop.
*   **Pool**: The large, unlabeled reservoir of frames that the Oracle (simulated) can label.
*   **Validation**: Held-out data for computing NLL/RMSE metrics.
*   **Sharding**: Splits large tensors into smaller `.pt` chunks for efficient dataloading.

**Output Structure**:
*   `train_seed.pt`
*   `pool_ref.pt`
*   `val_ref.pt`
