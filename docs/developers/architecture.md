---
layout: default
title: Architecture & Theory
parent: Developer Guides
nav_order: 0
math: mathjax
---

# Architecture & Theory

[< Back: Developer Guides](../developers.md) | [Next: Usage Guide](../usage.md)

---

This section provides a deep technical dive into the model architectures, data structures, and constraints used in MetaStateGen.

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

### 1. Backbone Diffusion Model: E(n) Equivariant Graph Neural Network (EGNN)

The diffusion model learns to reverse a noise process. Specifically, it predicts the noise $\epsilon$ added to a structure at timestep $t$. We utilize an **EGNN** because of its variance properties: if the input molecule rotates, the predicted noise vectors rotate exactly the same way (**Equivariance**), while the internal scalar features remain unchanged (**Invariance**).

**Class**: `metastategen.models.egnn.EGNN`

#### Mathematical Notation

| Symbol | Meaning | Dimensions |
| :--- | :--- | :--- |
| $h_i^l$ | Node Feature (scalar) for atom $i$ at layer $l$ | $\mathbb{R}^{D}$ (e.g., $D=128$) |
| $\vec{r}_i^l$ | Coordinate Vector for atom $i$ at layer $l$ | $\mathbb{R}^3$ |
| $m_{ij}$ | Message / Interaction vector between atoms $i, j$ | $\mathbb{R}^{D}$ |
| $\phi_e, \phi_h, \phi_x$ | Neural Networks (MLPs) for Edges, Nodes, and Coord updates | $\text{MLP}: \mathbb{R}^K \to \mathbb{R}^D$ |
| $\text{emb}(t)$ | Sinusoidal Time Embedding (Transformer-style) | $\mathbb{R}^{D}$ |

#### Update Mechanism (Layer $l \to l+1$)

The EGNN updates both the scalar features $h$ and vector coordinates $\vec{r}$ equivariantly:

1.  **Edge Update**: We verify pairwise interactions based on distance and atom states.
    $$ m_{ij} = \phi_e \left( h_i^l, h_j^l, \lVert \vec{r}_i^l - \vec{r}_j^l \rVert^2, \text{emb}(t) \right) $$
2.  **Coordinate Update**: Positions are nudged based on the message. The sum of differences ensures translation invariance.
    $$ \vec{r}_i^{l+1} = \vec{r}_i^l + \sum_{j \neq i} (\vec{r}_i^l - \vec{r}_j^l) \cdot \phi_x(m_{ij}) $$
3.  **Node Update**: Aggregating messages to update atomic state.
    $$ h_i^{l+1} = \phi_h \left( h_i^l, \sum_{j \neq i} m_{ij} \right) $$

#### Architecture Specifics
*   **Depth**: 6 Layers.
*   **Hidden Dimension ($D$)**: 128 channels.
*   **Activation**: SiLU (Swish).
*   **Conditioning**:
    *   **Timestep $t$**: Injected into every edge operation via $\text{emb}(t)$.
    *   **Atom Types**: One-hot encodings (C, N, O) form the initial node features $h_i^0$.

#### Chirality Fix
Standard EGNNs are **O(3)** equivariant (invariant to reflection/mirroring). This is problematic for chiral molecules like Alanine, where L-Ala and D-Ala are distinct enantiomers with different energies.

**Solution**: We explicitly compute **Chiral Volume** terms (scalar triple products) and inject them as additional scalar features into the Edge MLP $\phi_e$.

$$ V = \vec{r}_{N-CA} \cdot (\vec{r}_{CB-CA} \times \vec{r}_{C-CA}) $$

By conditioning on $V$, the model breaks reflection symmetry ($V \to -V$ under mirror), allowing it to distinguish L vs D forms.

---

### 2. Pairwise Surrogate Model (Energy Force Field)

This model acts as a differentiable "Energy Calculator" $E(x)$. It is trained to mimic the ground-truth Differential DFT/Force-Field energy.

**Class**: `metastategen.models.pairwise.PairwiseEnergyModel`

#### Core Logic
The total energy is modeled as a sum of learnable pairwise potentials. Unlike the EGNN, this is a simpler, translation/rotation invariant architecture tailored for stability.

$$ E_{total} = \sum_{i < j} \Psi_{\theta}(d_{ij}) $$

#### Input Pre-processing
1.  **Distances**: Compute all pairwise distances $d_{ij}$ ($N(N-1)/2$ pairs). For 22 atoms, this is ~231 pairs.
2.  **RBF Expansion**: We explicitly expand scalar distances into a high-dimensional vector using **Gaussian Radial Basis Functions (RBF)** to capturing short/medium/long range effects distinctly.
    $$ \text{RBF}_k(d) = \exp \left( - \frac{(d - \mu_k)^2}{\sigma^2} \right) $$
    *   **Parameters**: $N_{RBF}=32$ centers ($\mu_k$) spread linearly from 0Å to 10Å.

#### Network Architecture (The Potential $\Psi_\theta$)
*   **Input**: RBF vector of size 32.
*   **MLP**: A 3-Layer Dense Network applied to *each pair independently*.
    *   Layer 1: Linear(32 $\to$ 64) + SiLU
    *   Layer 2: Linear(64 $\to$ 64) + SiLU
    *   Layer 3: Linear(64 $\to$ 1) (No activation, outputs Energy contribution)
*   **Aggregation**: Summing the outputs of all pairs yields the total scalar Energy $E$.

#### Conservative Forces via Autograd
We do **not** train a separate network to predict forces. Instead, we compute the gradient of the predicted energy with respect to atomic input coordinates:

$$ \vec{F}_i = - \nabla_{\vec{r}_i} E_{total}(\vec{r}) $$

**Advantage**: This guarantees the vector field is **Conservative** (Curl-free, $\nabla \times F = 0$). This is physically required for a stationary potential energy surface and ensures that "rolling down hill" (Refinement) always leads to a valid minimum without entering infinite energy loops.

---

[< Back: Refinement Loop](../refinement.md) | [Next: Usage >](../usage.md)
