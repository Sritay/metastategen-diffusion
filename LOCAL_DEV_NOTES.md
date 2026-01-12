# Local Development Notes

## GPU Support

### macOS (Metal/MPS)
The local environment (PyTorch 2.1+) supports Metal Performance Shaders (MPS) for GPU acceleration on Mac (M1/M2/M3 or AMD Radeon).

However, **do not enable this by default in the main scripts** (`scripts/train_energy.py`, etc.), as the project runs on a cluster with AMD MI250 GPUs which expect standard `cuda` device semantics (via ROCm).

If you need to run locally with GPU acceleration for testing:
1.  Temporarily modify `scripts/train_energy.py`:
    ```python
    # Change:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # To:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ```
2.  Or use the CPU (default fallback).

### Cluster (AMD MI250)
The cluster environment likely uses ROCm. PyTorch on ROCm reports `torch.cuda.is_available() == True`, so the existing code works without modification.

### Local python env
Use .venv in repo root for local development. Activate it with:
source .venv/bin/activate or if it doesnt work discover how to yourself just use .venv