---
layout: default
title: Installation
parent: Usage Guide
nav_order: 1
---

# Installation

## 1. Clone the Repository
```bash
git clone https://github.com/Sritay/metastategen-diffusion.git
cd metastategen-diffusion
```

## 2. Environment Setup
We recommended using `venv` or `conda` with **Python 3.10+**.

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (Required for pyproject.toml support)
pip install --upgrade pip

# Install Package (Editable mode)
# This automatically installs dependencies (numpy, torch, etc.) defined in pyproject.toml
pip install -e .
```

> [!NOTE]
> For **HPC / ROCm** environments, you may need to install a specific PyTorch version *before* running the above command to avoid overwriting it with the PyPI default. See `slurm/pip-install.sh` for reference.
