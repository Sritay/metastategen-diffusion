# Agent Instructions: Day 10 - Force Surrogate & Refinement Implementation

## Context
We are implementing the "Physics Track" of the `metastategen-diffusion` project.
**Current State:** We have a working Diffusion model (baseline & ensemble) and Active Learning loop.
**Missing State:** We lack the "Force Surrogate" (Energy Model) and "Guided Refinement" (Langevin Dynamics) components originally planned.
**Constraint:** The user runs on **ROCm AMD GPUs** via SLURM. STRICTLY follow the headers/environment setup from existing scripts in `slurm/`.

## Objective
Implement the missing Force Surrogate pipeline to refine diffusion samples.
1.  **Data:** Download/Process ALA2 forces.
2.  **Model:** Train an EGNN to predict forces (Regression).
3.  **Sampling:** Implement `LangevinRefiner` to polish diffusion samples using the Force Surrogate.
4.  **Automation:** Provide SLURM scripts for training and sampling.

---

## Task 1: Data Acquisition (`scripts/get_mdshare_data.py`)
Modify `scripts/get_mdshare_data.py` to download the heavy-atom forces.
* **Source:** `mdshare`. Look for `alanine-dipeptide-3x250ns-heavy-atom-forces.npz` (or similar standard ALA2 force dataset available in mdshare).
* **Processing:**
    * Ensure strict alignment with the existing position dataset (`heavy-atom-positions`).
    * Create a new processed artifact: `data/processed/ala2/al_forces_ref.pt`.
    * **Crucial:** Ensure the split (train/val) uses the exact same indices as the position data to avoid data leakage (check `data/processed/ala2/meta.pt` or `setup_al_split.py` logic).

## Task 2: Force Model Implementation (`src/metastategen/models/`)
We will reuse the existing EGNN backbone but strictly for regression.
1.  **Modify/Verify `src/metastategen/models/egnn.py`**:
    * Ensure the `EGNN` class can return raw vector outputs (forces) without diffusion time-embeddings if needed (or create a `ForceEGNN` wrapper class in a new file `src/metastategen/models/force.py` that inherits from or utilizes `EGNN`).
    * **Input:** Positions $x$. (No time $t$).
    * **Output:** Predicted Forces $\hat{F}$ (Vector per atom).
    * **Loss:** MSE between $\hat{F}$ and Ground Truth $F$.

## Task 3: Training Script (`scripts/train_force.py`)
Create a new training script `scripts/train_force.py`.
* **Template:** Base this off `scripts/train_diffusion.py` but strip out the variance preserving SDE logic.
* **Logic:** Standard supervised regression loop.
* **Logging:** Log `Train Force MSE` and `Val Force MSE`.
* **Checkpointing:** Save to `runs/day10_force/checkpoints/`.

## Task 4: Refinement Sampler (`scripts/sample_refined.py`)
Create `scripts/sample_refined.py`.
* **Logic:**
    1.  Load the **Diffusion Model** (from `runs/day5_ensemble_k3` or `day2_baseline`).
    2.  Load the **Force Surrogate** (from `runs/day10_force`).
    3.  **Step A (Propose):** Generate standard samples using the Diffusion model (reuse `src/metastategen/models/diffusion.py` logic).
    4.  **Step B (Refine):** Run $K$ steps of Overdamped Langevin Dynamics using the **Force Surrogate**:
        $$x_{t+1} = x_t + \eta \nabla E(x_t) + \sqrt{2\eta\beta^{-1}} z_t$$
        (Where $\nabla E$ is the predicted force).
* **Config:** Allow configuring `refinement_steps` (e.g., 100) and `step_size` (eta).
* **Output:** Save samples to `runs/day10_force/samples/refined_samples.pt`.

## Task 5: SLURM Integration (`slurm/`)
Create two new SLURM scripts. **CRITICAL:** Read `slurm/20_train_gpu.sh` first. Copy its `#SBATCH` headers (partition, rocm modules, python environment) exactly.
1.  `slurm/60_train_force.sh`: Runs `scripts/train_force.py`.
2.  `slurm/61_sample_refined.sh`: Runs `scripts/sample_refined.py`.

## Compatibility Checklist
* **Names:** Use `ForceEGNN` or similar distinct naming if creating a new class.
* **Config:** Create `configs/ala2_force.yaml` for the surrogate training.
* **Environment:** Do NOT introduce `openmm` or `lammps` dependencies. Use `torch` only.
* **Paths:** Use relative paths consistent with the existing `dir_structure.txt`.

## Output Requirements
Generate the full code for:
1.  `scripts/get_mdshare_data.py` (Update)
2.  `src/metastategen/models/force.py` (New)
3.  `scripts/train_force.py` (New)
4.  `scripts/sample_refined.py` (New)
5.  `slurm/60_train_force.sh` (New)
6.  `slurm/61_sample_refined.sh` (New)
7.  `configs/ala2_force.yaml` (New)
