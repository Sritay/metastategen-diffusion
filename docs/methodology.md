# Methodology

The MetaStateGen approach divides the metastable state generation into two distinct, coupled loops.

---

## 1. Active Learning Loop (Backbone Generation)

**Primary Script**: [`scripts/run_al_loop.py`](../scripts/run_al_loop.py)

The Active Learning (AL) loop is the exploration engine. It iteratively improves a diffusion model by actively seeking out "uncertain" or novel structures.

### Step 0: Ensemble Initialization & Training
Before the loop begins, we initialize an **Ensemble** of $M$ probabilistic models (typically $M=4$).
*   **Function**: `_train_member`
*   **Process**: Each member is an EGNN-based Diffusion Model. They are trained independently on the initial seed dataset (MDShare data, 10-atom backbone).
*   **Diversity**: Diversity is induced by random initialization seeds and random dataloader shuffling. This ensures that in regions of dense data, models agree, while in unexplored regions, their predictions diverge.
*   **Training Objective**: Standard Denoising Diffusion objective: $\mathbb{E}_{t, x_0, \epsilon} [ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 ]$.

### Step 1: Candidate Generation (Consensus Sampling)
*   **Function**: `_sample_candidates` calls `_consensus_ddpm` or `_consensus_ddim`.
*   **Logic**: We generate a large pool of candidates (e.g., $N=1000$) to explore the landscape.
*   **Consensus Mechanism**: instead of generating from a single model, we use the **Mean** of the predicted noise from all ensemble members to guide the trajectory: $\bar{\epsilon}_\theta(x_t, t) = \frac{1}{M} \sum_{i=1}^M \epsilon_{\theta_i}(x_t, t)$. This stabilizes generation.
*   **Uncertainty Estimation**: Simultaneously, we compute the **Variance** of the noise predictions: $\mathbb{V}[\epsilon] = \frac{1}{M-1} \sum (\epsilon_i - \bar{\epsilon})^2$. This variance is summed over the diffusion trajectory to produce a single scalar "Uncertainty Score" for each generated structure.

### Step 2: Acquisition (Active Selection)
*   **Function**: `select_acquisition` (from `metastategen.active_learning.acquisition`)
*   **Strategy**: Maximize Uncertainty. We select the top $k$ candidates (e.g., $k=200$) with the highest accumulated uncertainty scores.
*   **Hypothesis**: High variance implies the models disagree, meaning they have not seen data in this region. Acquiring ground truth here yields the highest Information Gain.
*   **Code Reference**: `torch.topk(scores, k=k)` identifies the indices of the most novel structures.

### Step 3: Oracle Labeling
*   **Class**: `DatasetOracle` (from `metastategen.oracles.dataset_oracle`)
*   **Role**: The Oracle acts as the "Ground Truth". In a real wet-lab setting, this would be an experiment. In our simulation, it is a lookup into a massive validation dataset (MDShare).
*   **Query**: The Oracle takes the generated candidate positions $x_{gen}$.
*   **Response**: It performs a nearest-neighbor search (using `torch.cdist`) against the 250,000-frame pool to find the physically valid structure closest to the generated candidate. This "snaps" the generated hallucination to a real physical state.
*   **Visualization**:

![Acquisition Strategy](assets/acquisition_strategy.png)
*Figure 1: The AL loop acquiring new points (Red) in unvisited regions (White/Empty space) compared to the initial seed (Black) and previous iterations (Green).*

### Step 4: Retraining
*   **Process**: The newly labeled Oracle structures are added to the training set (`ALDataManager`).
*   **Fine-tuning**: The ensemble members are fine-tuned for a few epochs on this augmented dataset, incorporating the new knowledge.

---

## 2. Refinement Loop (All-Atom Reconstruction)

**Primary Script**: [`scripts/sample_refined.py`](../scripts/sample_refined.py)

The Refinement Loop converts the coarse, generated backbones into physically valid, generic low-energy states using a learned Pairwise Force Field.

### Step 1: Template alignment & Reconstruction
*   **Function**: `align_and_reconstruct` (from `metastategen.reconstruct`)
*   **Input**: 10-atom backbone from Diffusion ($X_{gen}$).
*   **Algorithm**: Kabsch Algorithm.
    1.  We take a reference Ideal Alanine Dipeptide template (22 atoms).
    2.  We extract its 10 backbone atoms.
    3.  We compute the optimal Rotation $R$ and Translation $T$ to align the template backbone to $X_{gen}$.
    4.  We apply $(R, T)$ to the **full** 22-atom template.
    5.  **Critical Step**: We overwrite the backbone positions with $X_{gen}$ to preserve the diffusion model's generated conformation, while keeping side-chains attached rigidly.

### Step 2: Warm-up Phase (Geometric Correction)
*   **Goal**: The rigid reconstruction creates "Frankenstein" molecules where side-chains might clash sterically or bonds might be slightly stretched.
*   **Process**: 1000 steps of gradient descent using the **Pairwise Force Model**.
*   **Bond Constraints**:
    *   **Function**: `constrain_bonds_22`
    *   **Logic**: The force model is soft. To prevent atoms from drifting into vacuum or collapsing, we explicitly project the N-CA and CA-C bonds to fixed physical lengths (1.46Å, 1.51Å) after every gradient step. This is a "Shake"-like algorithm implemented via iterative coordinate correction.

### Step 3: Energy Filtering
*   **Model**: `PairwiseEnergyModel`
*   **Logic**: We predict the potential energy $E(x)$ of all warmed-up candidates.
*   **Selection**: We keep only the top $1\%$ of structures (lowest Energy). This filters out kinetically trapped states or geometric disasters that the warm-up could not fix.

### Step 4: Main Refinement (Relaxation)
*   **Process**: The surviving candidates undergo a deep relaxation (50,000 steps).
*   **Physics**: This mimics an energy minimization or low-temperature molecular dynamics simulation. The structure slides down the Potential Energy Surface (PES) predicted by the surrogate model into the nearest local metastable basin.
