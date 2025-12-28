# Agent Instructions: Building the Physical Bridge (Day 11+)

**Goal:** Create a **NEW** script `scripts/run_physical_loop.py` that connects the Diffusion Committee to a Physical Oracle (Human/Simulator) via the Refinement Bridge.

**Critical Constraint:** Do **NOT** modify or delete `scripts/run_al_loop.py`. That script is optimzed for the "Proxy Task" (training the backbone generator against the dataset-oracle). The new script is for the "Real World" task.

## The Workflow Concept
1.  **Generate:** The Diffusion Ensemble creates a large pool of 10-atom backbones.
2.  **Filter/Query:** Select candidates with **High Uncertainty** (Committee Disagreement).
3.  **The Bridge (Refinement):**
    *   **Align:** Map the uncertain 10-atom backbones to the 22-atom template (using `src.metastategen.reconstruct`).
    *   **Refine:** Minimize the energy of these 22-atom structures using the **Force + Energy Surrogate** (Langevin Dynamics works on Forces, but Energy can be used for filtering).
    *   *Why?* A physical oracle (MD engine) requires relaxed structures. Passing raw backbones will result in instant rejection (high energy/clashes).
4.  **Oracle Call:**
    *   Pass the *refined* 22-atom structures to the Oracle.
    *   (Future: The Oracle will calculate stability via OpenMM or ask a Human Expert).
5.  **Feedback:** Add the validated data back to the training pool.

## Implementation Guide for `scripts/run_physical_loop.py`

### 1. Setup & Imports
*   Copy the skeleton from `run_al_loop.py` but **rename** it.
*   Import the Reconstruction Bridge and Energy Model:
    ```python
    from metastategen.reconstruct import align_and_reconstruct
    from metastategen.models.energy import EnergyEGNN
    ```

### 2. Load the Surrogate
*   Load the trained **Force + Energy Surrogate** checkpoint (from `runs/force_surrogate...` or `runs/energy_model...`).
*   **Model Class:** Use `EnergyEGNN`. It returns `(Energy, Force)` tuples.
*   Ensure it is in `eval()` mode to save memory.

### 3. Implement "The Bridge" Logic
Inside the active learning loop, *after* selecting uncertain indices (`sel_idx`) but *before* `oracle.query()`:

```python
# 1. Get Uncertain 10-atom Candidates
uncertain_backbones = samples[sel_idx] # [K, 10, 3]

# 2. Reconstruct (The Bridge Step A)
# Load 22-atom template once outside loop
refined_candidates = align_and_reconstruct(uncertain_backbones, template_22, heavy_indices)

# 3. Refine (The Bridge Step B)
# Run short Langevin dynamics using Surrogate
# See scripts/sample_refined.py for exact parameters (eta, steps, noise)
refined_candidates = run_langevin_refinement(refined_candidates, force_model, steps=100)

# 4. Query Physical Oracle
labels = physical_oracle.query(refined_candidates)
```

### 4. Configuration
*   Create a new config file `configs/ala2_physical.yaml`.
*   Include paths for both the Diffusion checkpoints (if fine-tuning) and the Force Surrogate checkpoint.
