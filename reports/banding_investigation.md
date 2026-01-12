# Investigation Report: Banding and Planar Collapse in Generated Structures

## Executive Summary
The "banding" artifacts observed in the generated Ramachandran plots (diagonal lines $\psi = \phi \pm 180$) and the cluster at $(0,0)$ are caused by **Planar Collapse** of the generated molecules.
This is a direct result of the **EGNN Architecture** being invariant to reflections (O(3) invariant features), which prevents it from distinguishing between L-Alanine and D-Alanine. As a result, the diffusion model predicts the *average* of the two enantiomers, which is a non-physical planar structure.

## Evidence

### 1. Visual Geometry (Banding)
The observed bands lie on the lines:
$$ \psi = \phi + 180 $$
$$ \psi = \phi - 180 $$
These lines correspond geometrically to conformations where the backbone is perfectly flat (all atoms in a plane). The cluster at $(0,0)$ corresponds to the fully extended/eclipsed planar state.

### 2. Quantitative Chirality Analysis
We computed the Chiral Volume ($V = (N-C_\alpha) \cdot ((C-C_\alpha) \times (C_\beta-C_\alpha))$) for both the Seed Data and the Generated Samples.

| Metric | Seed Data (Ground Truth) | Generated Samples (Loop 12, Iter 20) |
| :--- | :--- | :--- |
| **Mean Chiral Volume** | **0.0025** (Consistent L-Ala) | **0.0001** ($\approx 0$) |
| **Std Dev** | 0.0002 | 0.0051 |
| **L-Fraction** | **100%** | **50%** (Racemic Noise) |
| **Planar Fraction** | 0% | **100%** |

### 3. Root Cause: Reflection Equivariance
The standard EGNN (E(n) Equivariant Graph Neural Network) uses **distances** ($d_{ij} = ||x_i - x_j||$) as its primary edge feature.
*   **O(3) Invariance:** Distances are invariant to rotation *and* reflection. The distance matrix of L-Alanine is identical to that of D-Alanine.
*   **Ambiguity:** When the model sees a noisy structure, it extracts distance features. If these features cannot distinguish L from D, the model must predict a score (force) that is valid for *both*.
*   **Averaging:** The optimal prediction (minimizing MSE) for a bimodal distribution (L and D) where inputs are ambiguous is the **mean**. The geometric mean of L-Ala and D-Ala is the **Planar** structure.

## Conclusion
The "diffraction pattern" is effectively an interference pattern between the L and D modes, collapsing the probability density onto the symmetry plane.
The issue is **not** in the seed data (which is 100% L-Ala).
The issue is **in the architecture**: The current EGNN implementation lacks chiral features (e.g., torsion angles or scalar triple products) necessary to break reflection symmetry.

## Recommendations
To fix this, the model needs **Chiral Features**:
1.  **Input Features:** explicitly provide torsion angles or signed volumes as node/edge attributes.
2.  **Network Modification:** Use an SO(3)-equivariant architecture (like e3nn or modified EGNN) that is sensitive to chirality.
