# Methodology

The MetaStateGen approach divides the metastable state generation into two distinct loops: the **Active Learning (AL) Loop** and the **Refinement Loop**.

## 1. Active Learning Loop (Backbone Generation)
The goal of the Active Learning loop is to train a generative model that can sample diverse backbone conformations, even those that are rare in the initial training data.

### Process
1.  **Train Ensemble**: An ensemble of EGNN-based Diffusion models is trained on the current "labeled" dataset of backbone structures.
2.  **Generate Samples**: The ensemble generates varying candidate structures.
3.  **Oracle Evaluation**: An "Oracle" (Ground Truth MD simulation or Reference Potential) evaluates the candidates.
4.  **Acquisition**: High-uncertainty or novel candidates are identified (e.g., using variation across the ensemble or distance from known basins).
5.  **Update Pool**: These new candidates are labeled and added to the training pool.
6.  **Loop**: The process repeats, progressively exploring the energy landscape.

### Data: MDShare
-   **Content**: 10-atom backbone coordinates (N, CA, C, O, etc.).
-   **Features**: No forces or energies provided.
-   **Usage**: Solely for training the geometric diffusion model to learn the distribution $p(x_{backbone})$.

---

## 2. Refinement Loop (All-Atom Reconstruction)
The diffusion model outputs coarse backbone approximations. The Refinement Loop is responsible for converting these into physically valid, generic low-energy states.

### Process
1.  **Reconstruction**: The 10-atom backbone is mapped to a full 22-atom Alanine Dipeptide template. Side chain atoms are initialized via template matching.
2.  **Warm-up**: A short burst of gradient descent using the learned Pairwise Force Field. This fixes gross steric clashes from the reconstruction.
3.  **Filtering**: Candidates are ranked by the Pairwise Energy Model. Only the lowest energy fraction (e.g., top 1%) are kept.
4.  **Main Refinement**: Extended Langevin dynamics or Gradient Descent simulations relax the structures into the nearest local minima.

### Data: Timewarp
-   **Content**: Full 22-atom coordinates.
-   **Features**: Includes ground truth **Potential Energies** and **Atomic Forces**.
-   **Usage**: Training the Pairwise Surrogate Model.

### Why separate loops?
-   **Efficiency**: Diffusion on 10 atoms is faster and easier to train than on 22 atoms.
-   **Data Availability**: We often have abundant coarse data but expensive/limited labeled force data.
-   **precision**: The surrogate model, trained on forces, provides the local precision that score-based diffusion models might miss.
