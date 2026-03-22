---
layout: default
title: Installation
parent: Usage Guide
nav_order: 1
---

# Installation

[< Back: Usage Index](../usage.md) | [Next: Verification >](verification.md)

---

## 1. Clone the Repository
```bash
git clone https://github.com/Sritay/metastategen-diffusion.git
cd metastategen-diffusion
```

## 2. Environment Setup
It is recommended to use `venv` or `conda` with **Python 3.10+**.

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (Required for pyproject.toml support)
pip install --upgrade pip

# Install Package (Editable mode)
# This automatically installs dependencies (numpy, torch, mdtraj, etc.) defined in pyproject.toml
pip install -e .
```

> [!NOTE]
> For **HPC / ROCm** environments, you may need to install a specific PyTorch version *before* running the above command to avoid overwriting it with the PyPI default. See `slurm/pip-install.sh` for reference.

## 3. Download Benchmark Data (Optional)
While MetaStateGen supports parsing your own arbitrary data (PDBs, LAMMPs, GRO, NPZ), the project includes the Alanine Dipeptide dataset as a standard benchmark. Run the provided script to fetch and process it:

```bash
python scripts/get_mdshare_data.py
python scripts/preprocess_positions.py
python scripts/setup_al_split.py
```
This sequence:
1. Downloads raw MD data to `data/raw`.
2. Processes it into unified PyTorch shards in `data/processed/ala2`.
3. Generates the Seed, Pool, and Validation splits required for Active Learning benchmarks.

---

[< Back: Usage Index](../usage.md) | [Next: Verification >](verification.md)
