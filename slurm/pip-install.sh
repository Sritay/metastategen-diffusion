#!/bin/bash
set -euo pipefail

# Ensure pip tooling is current
python -m pip install --upgrade pip setuptools wheel

# Core scientific stack (keep minimal; numpy<2 for torch 2.2 builds)
python -m pip install "numpy<2" scipy matplotlib pyyaml tqdm rich mdshare

# ROCm 5.6 wheels for PyTorch 2.2 (as in your guide snippet)
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 torchtext==0.17.0 \
  --index-url https://download.pytorch.org/whl/rocm5.6

# Install this repo editable (so `msgen` is available)
python -m pip install -e .

