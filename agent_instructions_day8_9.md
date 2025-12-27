# AGENT INSTRUCTIONS: MetaStateGen-Diffusion (Days 8-9)

## 1. Role & Persona
You are a Senior ML Research Engineer.
* **Focus:** Automating the scientific discovery loop. You care about **closed-loop orchestration**: Model → Sampling → Uncertainty → Oracle → Retraining.
* **Mindset:** "Simulate first." Since we don't want to compile complex MD engines (OpenMM/LAMMPS) inside the container just yet, we will **simulate** the expensive experiment using a "Dataset Oracle."

## 2. Project State (Context)
* **Completed (Days 1-7):**
    * End-to-end Diffusion training & sampling.
    * Ensemble construction ($K$ models) for uncertainty estimation.
    * Basin clustering and Triage metrics.
* **Current Goal (Days 8-9):** **Active Learning (AL) Loop.**
    * **Critical Pivot:** We cannot use the "Day 2" model (trained on Traj 0+1) for the AL demo because it already "knows" too much. To demonstrate *discovery*, we must start fresh with a **"Weak Learner"** trained on a tiny fraction of the data.

## 3. The "Dataset Oracle" Strategy
To avoid heavy MD dependencies for now, we will **mock** the physical simulation:
1.  **The "Universe" (Hidden):** The full high-resolution dataset (Traj 0, 1, 2).
2.  **The "Experiment":**
    * The model generates a sample $x_{gen}$.
    * **Oracle Query:** The oracle takes $x_{gen}$, finds the **nearest neighbor** $x_{ref}$ in the "Hidden Universe", and returns $x_{ref}$.
    * *Physics interpretation:* Simulates relaxing a noisy structure to a true equilibrium state.

## 4. Immediate Task List (Days 8-9)

### A. Data Setup: The "Discovery" Split (CRITICAL NEW TASK)
* **`scripts/setup_al_split.py`**:
    * Loads the full mdshare data.
    * Creates a strict partition to simulate a "data-poor" starting point:
        * **Initial Seed (`data/processed/ala2/al_seed.pt`):** Small subset (e.g., first 2% of Traj 0, or ~5k frames).
        * **Oracle Pool (`data/processed/ala2/al_pool_ref.pt`):** The rest of the data (Traj 0 [remainder] + Traj 1 + Traj 2). This is what the Oracle searches.
        * **Validation (`data/processed/ala2/al_val.pt`):** A held-out subset of Traj 2 to measure generalization.
    * *Why?* If we don't do this, the model starts with low uncertainty, and the AL curves will be flat.

### B. The Oracle Module
* **`src/metastategen/oracles/base.py`**: Abstract base class `Oracle` with method `query(positions: Tensor) -> Tensor`.
* **`src/metastategen/oracles/dataset_oracle.py`**:
    * Loads `al_pool_ref.pt`.
    * **Memory Safety:** Implements `query` using **batched** distance calculations.
        * *Pitfall Avoidance:* Do NOT compute a $[N_{gen}, N_{pool}]$ matrix (e.g., 2k $\times$ 750k) in one go. Chunk $N_{gen}$ into batches of ~100 to avoid OOM on GPU.
    * Returns the nearest neighbors (the "relaxed" structures).

### C. Active Learning Logic & Data Management
* **`src/metastategen/active_learning/acquisition.py`**:
    * Implement `random` and `uncertainty` (variance-based) selection.
* **`src/metastategen/data/manager.py`**:
    * **Dataset Aggregation:** Utility to merge the `Initial Seed` + `Acquired Data`.
    * *Pitfall Avoidance:* Ensure the DataLoader mixes these effectively. Do not just finetune on the *new* data (catastrophic forgetting). The retraining step must see the **cumulative** dataset.

### D. The AL Loop Script
* **`scripts/run_al_loop.py`**: The "Flagship" orchestration.
    * **Phase 0 (Cold Start):** Train a fresh ensemble on `al_seed.pt` (do not load Day 2 weights).
    * **Loop (Iter 1..N):**
        1.  **Sample:** Generate candidates using current ensemble.
        2.  **Score:** Compute uncertainty.
        3.  **Acquire:** Select top-$k$ candidates.
        4.  **Label (Oracle):** Get "relaxed" structures from the Pool.
        5.  **Update:** Append to the Cumulative Dataset.
        6.  **Retrain:** Finetune the ensemble on the Cumulative Dataset (warm-start allowed from previous iter).
        7.  **Evaluate:** Compute Basin Coverage / KL on `al_val.pt`.
    * **Output:** `al_metrics.csv` (Coverage vs. Oracle Calls).

### E. Configs & Slurm
* **`configs/ala2_al.yaml`**:
    * Define `initial_seed_size` and `oracle_pool_source`.
    * Set `training.finetune_epochs` (keep it short, e.g., 5-10 epochs per AL iter).
* **`slurm/50_al_loop.slurm`**: Long-running job (or chained dependencies) on GPU.

## 5. Required File Structure Additions

```text
metastategen-diffusion/
├── configs/
│   └── ala2_al.yaml             <-- NEW (AL specific config)
├── src/
│   └── metastategen/
│       ├── active_learning/
│       │   ├── __init__.py
│       │   └── acquisition.py   <-- NEW
│       ├── oracles/
│       │   ├── __init__.py
│       │   ├── base.py          <-- NEW
│       │   └── dataset_oracle.py <-- NEW (Batched implementation)
│       └── data/
│           └── manager.py       <-- NEW (Cumulative dataset logic)
├── scripts/
│   ├── setup_al_split.py        <-- NEW (The "Discovery" partition)
│   └── run_al_loop.py           <-- NEW (Main loop)
└── slurm/
    └── 50_al_loop.slurm         <-- NEW


6. Execution Guidelines
Start Small: For the first run, use a very small seed (e.g., 100 frames) and a short loop (3 iterations) to verify the pipeline works.

Metrics: The key success indicator is the rate of basin discovery. Uncertainty sampling should find the "unseen" basins of Traj 2 faster than random sampling.

Determinism: Log the indices of the "Acquired" frames so we can debug exactly which structures the model "asked for."
