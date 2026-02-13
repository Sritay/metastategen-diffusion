---
layout: default
title: Lignin Generation Example
parent: Usage Guide
nav_order: 3
---

# Lignin Conformer Generation (L0 Example)

[< Back: Inference](inference.md)

This guide demonstrates how to generate valid conformers for complex, flexible molecules like **Lignin (L0)**. Unlike small rigid molecules, lignin requires special handling for its flexible glycosidic linkages and ring systems.

## Key Features Used

*   **Robust Local Frame Reconstruction**: The `align_and_reconstruct` module now uses a robust frame definition that prevents hydrogen collapse even when heavy-atom geometry is distorted.
*   **Geometric Refinement**: A "physics-free" refinement mode that resolves steric clashes and enforces bond lengths/angles using topology constraints, without requiring a trained force field.
*   **Topology-Aware Constraints**: Automatically infers ring rigidity, 1-3 angle constraints, and geminal H-H pairs from the input topology (PSF/PDB).

---

## Step 1: Prepare Input Data

You need:
1.  **Coordinates**: A PDB file defining the reference structure (e.g., `data/miscanthus/L0.pdb`).
2.  **Topology**: A PSF (Charmm) or GRO (Gromacs) file defining bonds and atom types (e.g., `data/miscanthus/L0.psf`).
3.  **Checkpoint**: A trained diffusion model checkpoint (e.g., `runs/verify_l0_geo/checkpoints/final.pt`).

> **Note**: For this example, we assume `scale_factor: 7.0` was used during training to scale bond lengths (~0.15 nm) to diffusion scale (~1.0).

---

## Step 2: Run Sampling

Run the `msgen sample` command. The critical flags for valid lignin generation are `--refinement-mode geometric` and `--connectivity`.

```bash
# Example Command
msgen sample \
  --diff-config configs/verify_l0_local.yaml \
  --diff-ckpt runs/verify_l0_geo/checkpoints/final.pt \
  --topology data/miscanthus/L0.pdb \
  --connectivity data/miscanthus/L0.psf \
  --refinement-mode geometric \
  --n-samples 100 \
  --batch-size 10 \
  --out-dir runs/l0_generation_example \
  --output-formats pdb
```

### Breakdown of Arguments

*   `--refinement-mode geometric`: Enables the clash-removal refinement loop. This is essential for preventing "ghost" atoms that overlap.
*   `--connectivity data/miscanthus/L0.psf`: Provides the bond graph. The pipeline uses this to:
    *   Identify rings (3-8 members) and enforce planarity.
    *   Identify 1-3 angle pairs (heavy atoms and H atoms) to enforce geometry.
    *   Protect geminal Hydrogens from collapsing during refinement.
*   `--topology data/miscanthus/L0.pdb`: Provides the reference template for reconstruction.

---

## Step 3: Output Analysis

The results will be saved in `runs/l0_generation_example/`.

*   **`refined.pdb`**: The final structures. Includes `CONECT` records so visualization tools like VMD draw bonds correctly without guessing.
*   **`initial.pdb`**: The raw output from the diffusion model (before refinement).

### Visualization
Open `refined.pdb` in VMD. You should see physically plausible structures with:
*   Correct ring puckering/planarity.
*   No overlapping atoms (clashes resolved).
*   Correct tetrahedral geometry at CH2/CH3 centers.

---

[< Back: Inference](inference.md)
