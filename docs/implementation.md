# Implementation Details

[< Back to Methodology](methodology.md) | [Next: Usage >](usage.md)

This section provides a deep technical dive into the model architectures, data structures, and constraints used in MetaStateGen.

---

## Datasets

### 1. MDShare (Backbone Only)
*   **Role**: Training data for the Active Learning (Diffusion) loop.
*   **Structure**: 10-atom point clouds of Alanine Dipeptide.
*   **Atom Selection**: N, CA, C, O, CB (Sidechain Carbon), and associated backbone atoms. Hydrogens are implicitly ignored.
*   **Source**: A 250,000-frame Molecular Dynamics trajectory provided by the MDShare project.
*   **Preprocessing**:
    *   **Centering**: All molecules are translated so their Center of Mass (COM) is at the origin.
    *   **Scaling**: Coordinates are typically scaled (e.g., divided by a scale factor) to be roughly $\sim O(1)$ for the neural network.
    *   **Aligning**: (Optional) Rotational alignment is NOT performed because the EGNN is E(3)-equivariant (it handles rotation naturally).

### 2. Timewarp (All-Atom)
*   **Role**: Training data for the Pairwise Surrogate/Refinement loop.
*   **Structure**: Full 22-atom system including all Hydrogens.
*   **Rich Labels**: Unlike MDShare, this dataset includes **Forces** ($F_i \in \mathbb{R}^3$) and **Potential Energy** ($E \in \mathbb{R}$) for every frame.
*   **Necessity**: Diffusion models can learn distributions $p(x)$ from positions alone. However, to *refine* a structure, we need to minimize energy $\nabla_x E$. This requires learning the energy landscape, for which force labels are crucial supervision.

---

## Model Architectures

### Backbone Diffusion Model: E(n) Equivariant Graph Neural Network (EGNN)
The diffusion model learns to predict the noise $\epsilon$ added to a structure. We use an EGNN because it guarantees that if the input molecule rotates, the predicted noise vectors rotate exactly the same way (Equivariance).

*   **Class**: `metastategen.models.egnn.EGNN`
*   **Graph Structure**: Fully connected graph (all atoms attend to all other atoms).
*   **Layers**: 6 Layers.
*   **Hidden Dimension**: 128 or 256 channels.
*   **Update Mechanism**:
    1.  **Edge Update**: $m_{ij} = \phi_e(h_i, h_j, \|r_i - r_j\|^2, \text{emb}(t))$
    2.  **Coordinate Update**: $r_i^{l+1} = r_i^l + \sum_{j} (r_i - r_j) \phi_x(m_{ij})$
    3.  **Node Update**: $h_i^{l+1} = \phi_h(h_i, \sum_j m_{ij})$
*   **Conditioning**:
    *   **Timestep $t$**: Sinusoidal positional embeddings injected into the edge MLP.
    *   **Atom Types**: One-hot encodings (C, N, O) injected into the initial node features $h_i^0$.
*   **Chirality Fix**:
    *   Standard EGNNs are **O(3)** equivariant (invariant to reflection/mirroring). This is bad for chiral molecules like Alanine, where L-Ala and D-Ala are distinct.
    *   **Solution**: We explicitly compute **Chiral Volume** features (scalar triple products $V = \vec{r}_{N-CA} \cdot (\vec{r}_{CB-CA} \times \vec{r}_{C-CA})$) and inject them as edge features. This breaks the reflection symmetry, allowing the model to distinguish enantiomers.

### Pairwise Surrogate Model (Energy Force Field)
This model acts as a differentiable "calculator" that predicts Energy given positions.

*   **Class**: `metastategen.models.pairwise.PairwiseEnergyModel`
*   **Core Logic**: Energy is a sum of pairwise interactions.
*   **Input Pre-processing**:
    1.  Compute all pairwise distances $d_{ij}$ ($N(N-1)/2$ pairs).
    2.  **RBF Expansion**: We expand each scalar distance into a high-dimensional vector using Gaussian Radial Basis Functions (RBF).
        *   Why? A single scalar $d_{ij}$ is hard to learn from. RBFs allow the network to learn different "bins" of interaction strength (e.g., strong repulsion at short range, attraction at medium range).
        *   Parameters: $N_{RBF}=32$ centers spread from 0Å to 10Å.
*   **Architecture**:
    *   The RBF features for all pairs are flattened and passed through a **3-Layer MLP**.
    *   Output: Single scalar $E$.
*   **Force Derivation**:
    *   To get forces, we do not output a vector. We compute $F = -\nabla_x E$ using PyTorch `autograd`.
    *   **Advantage**: This ensures the vector field is "conservative" (curl-free). Restoring forces always point towards lower energy.

---

[< Back to Methodology](methodology.md) | [Next: Usage >](usage.md)
