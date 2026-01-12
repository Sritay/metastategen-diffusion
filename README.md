# MetaStateGen-Diffusion

Metastable state generation with equivariant diffusion for molecular dynamics.

## Overview

This project implements a "Physics-Compliant" diffusion model for generating metastable states of Alanine Dipeptide. It uses E(3)-equivariant networks and a two-stage active learning pipeline:

1.  **Loop A (Active Learning):** Trains a committee of diffusion models on `mdshare` data to identify uncertainty and query an Oracle.
2.  **Loop B (Refinement):** Uses a pairwise force surrogate (trained on TimeWarp data) to refine generated structures via Langevin dynamics, ensuring physical plausibility.

## Installation

1.  **Environment Setup:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    ```

    *Note: For ROCm AMD GPU support, please see `slurm/pip-install.sh` or install PyTorch with the appropriate `--index-url`.*

2.  **Data:**
    Data is expected in `data/` directory.
    - `mdshare` data: `data/processed/ala2/`
    - `timewarp` data: `data/timewarp/`

## Usage

### Training Diffusion
```bash
python scripts/train_diffusion.py --config configs/ala2_default.yaml
```

### Sampling & Refinement
```bash
python scripts/sample_refined.py --diff-ckpt runs/best_model.pt --force-ckpt runs/energy_model.pt
```

## Structure

- `src/metastategen/`: Core python package (Models, Data, Oracles).
- `scripts/`: CLI scripts for training and sampling.
- `configs/`: YAML configuration files.
- `slurm/`: Example SLURM submission scripts.

## License

MIT
