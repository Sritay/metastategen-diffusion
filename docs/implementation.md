# Implementation Details

## Model Architectures

### 1. Backbone Diffusion Model
-   **Backbone**: E(n) Equivariant Graph Neural Network (EGNN).
-   **Type**: Denoising Diffusion Probabilistic Model (DDPM).
-   **Input**: 10-atom backbone point cloud $x \in \mathbb{R}^{10 \times 3}$.
-   **Conditioning**: One-hot atom types (C, N, O).
-   **Symmetry**: Rotation and Translation invariant by design (SE(3) symmetry).

### 2. Pairwise Surrogate Model
-   **Type**: Graph-based Force Field.
-   **Input**: Full 22-atom coordinates.
-   **Mechanism**:
    1.  Computes all pairwise distances $d_{ij}$.
    2.  Expands distances using **Gaussian RBF** (Radial Basis Functions) to capture local interactions.
    3.  Passes RBF features through an MLP to predict scalar energy $E$.
    4.  Forces $F = -\nabla_x E$ are computed via autograd.
-   **Training**: Minimizes Mean Squared Error (MSE) on both Energy and Forces against the **Timewarp** dataset.

---

## Key Design Constraints

### Geometric Constraints
During the refinement process, the diffusion output can sometimes be unphysical (e.g., broken bonds). To remedy this, we apply **Hard Constraints**:
-   **Bond Lengths**: The N-CA (1.46 Å) and CA-C (1.51 Å) bonds are iteratively projected to their equilibrium lengths during the refinement steps.
-   **Reason**: Prevents the "explosion" of structures where the force model might otherwise push atoms infinitely far apart in vacuum.

### Chirality Handling
Standard EGNNs are reflection invariant (O(3)), meaning they cannot distinguish between enantiomers (L-Alanine vs D-Alanine).
-   **Problem**: The model might generate D-Alanine mixed with L-Alanine, or "collapse" into planar transition states.
-   **Solution**: We compute explicit **Chiral Features** (e.g., Scalar Triple Products of the N-CA-CB-C neighbors) and inject them into the model or use them as auxiliary supervisory signals to enforce the correct L-isomerism.
