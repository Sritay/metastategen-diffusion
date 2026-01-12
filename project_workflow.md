# Project Workflow

## Overview
This project consists of two distinct but connected loops:

- **Loop A — Training & Active Learning**
  Uses an ensemble (“committee”) of diffusion models to measure uncertainty and
  select informative data for labeling.
  
- **Loop B — Generation & Physical Refinement**
  Uses a *single* trained diffusion model to generate molecular structures,
  followed by refinement using a learned force surrogate and optional physics
  oracle validation.

---

## Loop A: Training + Active Learning

**Goal:** Efficiently improve the diffusion model using uncertainty-driven data acquisition.

1️⃣ **Initial Labeled Subset**
- Begin with ~5–10% labeled MD frames.

2️⃣ **Train a Committee of Diffusion Models**
- 3–5 independently trained models.
- Models learn structural distribution.

3️⃣ **Uncertainty Estimation**
- Evaluate remaining unlabeled frames.
- Compute committee disagreement (uncertainty).

4️⃣ **Query Oracle**
- In the demo: “oracle” = hidden labels from existing dataset.
- In a real system: oracle could be MD simulation or expert input.

5️⃣ **Expand Training Pool**
- Add queried data to labeled set.

6️⃣ **Retrain / Update Models**
- Iterate and produce learning curves (coverage vs. #queries).

**Outcome:**  
A stronger, data-efficient diffusion model with demonstrated active-learning benefits.

---

## Loop B: Sampling + Guided Physical Refinement

**Goal:** Generate scientifically credible molecular structures.

1️⃣ **Final Generator Model**
- Use a *single trained diffusion model*
  - best performing model
  - or retrained model using all selected AL data

2️⃣ **Sample Candidate Structures**
- Generate conformations from the diffusion model.

3️⃣ **Filtering (Top 1%)**
- Generate a large pool (e.g., 10,000 samples).
- Compute Force-Surrogate Energy for all.
- Keep only the lowest energy ~1% (e.g., 100) for expensive refinement.
- *Note:* This significantly improves yield of metastable states.

4️⃣ **Force-Surrogate Refinement**
- Apply short Langevin / gradient refinement using learned force predictor.
- Push samples toward lower-energy basins.

5️⃣ **Optional Physics Oracle Validation**
- Use LAMMPS or similar backend for minimization / verification.

6️⃣ **Evaluation**
- Compare φ/ψ densities
- Compare free-energy landscapes
- Evaluate basin coverage / clustering
- **Visualization (Movie Generation):**
  - Since filtering breaks the 1-to-1 index, use **RMSD Matching** to pair each refined structure with its closest "initial" parent from the generated pool.
  - Create interleaved PDB (Frame N: Raw, Frame N+1: Refined).

**Outcome:**  
Physically plausible, diverse, and validated molecular structures.

---

## Key Clarifications

- **Committee is ONLY used in Loop A**  
  (uncertainty + active learning).

- **Loop B does NOT use uncertainty**  
  It uses a single trained generator + refinement.

- **Refined samples are NOT added back to training**  
  to avoid surrogate-induced bias.

---

## Deliverables
- Trained diffusion models
- Active learning performance plots
- Generated structure sets
- Refined structure sets
- φ/ψ density comparisons
- Free-energy comparisons
- Basin clustering analysis
- CLI + Slurm workflows
- Short demo + report

