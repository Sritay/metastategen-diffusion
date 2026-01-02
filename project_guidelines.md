# MetaStateGen-Diffusion: Project Guidelines & Agent Instructions

## 1. Project Vision & Physics Justification
**Goal:** Develop a flagship "SOTA ML for MD" demo project. The primary purpose is to be desirable to academic and industry employers by generating metastable states of Molecular Dynamics (MD) configurations.

### 1.1 Why this is not a "Toy" Project
*   **Benchmark:** We use **Alanine Dipeptide (Ala2)** via `mdshare`. This is the standard "unit test" for free energy methods.
*   **Validation:** We evaluate properly using the **Free Energy Surface (FES)** over backbone dihedrals ($\phi, \psi$). This proves understanding of statistical mechanics, unlike simple RMSD metrics.
*   **Methodology:** We move beyond "density matching" to **"Physics-Compliant Sampling"**:
    *   **E(3)-Equivariance:** Models must respect rotational symmetry.
    *   **Force Surrogate:** We train a force predictor ($F = -\nabla V$) to refine generative samples, fixing steric clashes and driving structures to local minima.
    *   **Oracle-Agnostic:** We avoid hard dependencies on OpenMM/LAMMPS for the core loop, making the tool adaptable to any energy calculator.

---

## 2. strict Technical Constraints for Agents
All agents working on this project **MUST** adhere to the following constraints.

### 2.1 Hardware & Environment
*   **Platform:** **ROCm AMD GPUs**.
*   **Job Scheduler:** **SLURM**.
*   **Environment Source of Truth:** Read `slurm/20_train_gpu.sh` (or similar existing scripts) for the exact `#SBATCH` headers, module loads, and Python environment setup. **Do not guess generic CUDA commands.**

### 2.2 Data Strategy: "The Two-Dataset Pattern"
We strictly separate the sources for generic sampling vs. force training due to data availability limitations.
1.  **Positions & Dihedrals (Week 1 / Generation):**
    *   **Source:** `mdshare`.
    *   **Files:** `alanine-dipeptide-3x250ns-backbone-dihedrals.npz`, `alanine-dipeptide-3x250ns-heavy-atom-positions.npz`.
    *   *Purpose:* Training the diffusion model and validating the Free Energy Surface.
2.  **Forces (Week 2 / Refinement):**
    *   **Source:** `mdshare` *does not* provide the forces we need by default. We use the **TimeWarp** dataset which is already available locally in `data/timewarp`.
    *   **Units:** Verified as **Nanometers** (consistent with MDShare).
    *   **Atom Count Strategy (CRITICAL UPDATE):**
        *   **Diffusion Model:** Operates on **10 heavy atoms** (mdshare data).
        *   **Pairwise Surrogate:** Operates on **All 22 atoms** (Timewarp data).
        *   **Reason:** Forces on heavy atoms are physically dependent on Hydrogen positions. Training on 10 atoms creates "noised" forces and instability.
        *   **Bridge:** To refine a diffusion sample, we must **reconstruct Hydrogens** (e.g., using `pdbfixer` or geometric generic placement) before passing it to the Pairwise Surrogate.
    *   **Refinement Strategy:** Use `scripts/process_timewarp.py` with `force_heavy_only=False`. We train the surrogate on the full physical system.

### 2.3 Dependency Constraints
*   **Core Logic:** `torch`, `numpy`, `mdshare`.
*   **Forbidden in Core:** `openmm`, `lammps`.
    *   *Reason:* Installation complexity on various clusters.
    *   *Alternative:* Use a **"Dataset Oracle"** (Nearest Neighbor in a hidden pool) to simulate expensive physical relaxations during Active Learning experiments.
*   **Visuals:** `matplotlib` for all plots.

---

## 3. The 14-Day Implementation Plan

### Week 1: Real Benchmark + Real Evaluation
*   **Focus:** Geometry Generation & FES Validation.
*   **Done:**
    *   Install `mdshare` and get `.npz` data.
    *   Reference plots: Ramachandran density, FES $F(\phi, \psi)$.
    *   Diffusion model on heavy atoms (E(3)-aware).
    *   Metric: Basin clustering vs Reference.

### Week 2: "Almost Adoptable" (Forces & Refinement)
*   **Focus:** Physics compliance & Active Learning.
*   **Current Active Tasks:**
    1.  **Pairwise Surrogate:** Train an MLP to predict pairwise forces.
    2.  **Guided Refinement:** Use the surrogate to run Langevin dynamics on diffusion samples ($x_{t+1} = x_t + \eta \nabla E + \dots$).
    3.  **Active Learning Loop:**
        *   Start with a "Weak Learner" (tiny % of data).
        *   Generator $\to$ Uncertainty $\to$ Query Oracle $\to$ Retrain.
        *   Use the "Dataset Oracle" to mock the lab labeling process.
    4.  **LAMMPS Integration (Bonus):** Demonstrate "Oracle Agnostic" capability by plugging in a LAMMPS backend for a materials system (separate from Ala2).

---

## 4. Directory Structure Standard
Maintain the standard project tree:
```text
metastategen-diffusion/
├── configs/           # YAML configs (use hydra-like structure or simple yaml)
├── data/
│   ├── raw/           # Downloads
│   └── processed/     # .pt tensors (split, aligned)
├── reports/           # PNG plots and markdown reports
├── runs/              # Experiment logs, checkpoints, samples
├── scripts/           # CLI python scripts (train, sample, data_prep)
├── slurm/             # .sh submission scripts (ROCm specific)
└── src/
    └── metastategen/  # Python package
        ├── models/    # diffusion.py, egnn.py, force.py
        ├── data/      # dataset classes
        └── oracles/   # oracle interfaces
```
