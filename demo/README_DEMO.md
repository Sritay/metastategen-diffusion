# Flagship Demo Artifacts

This folder contains the assets and scripts for the flagship demonstration of the Active Learning (Loop A) and Refinement (Loop B) pipeline.

## 1. Evolution Visualization
**File:** `evolution_plot.png`
**Description:** Visualizes the progression of generated structures across Active Learning iterations (Iter 0 to Iter 3).
- **Iter 0:** Noisy, unstructured (Random/Cold Start).
- **Iter 3:** Clear "Alpha Basin" formation (Phi ~ -75, Psi ~ -40).

## 2. Refinement Funnel
**File:** `funnel_plot.png` (Generating...)
**Description:** Visualizes the effect of the "Loop B" Force Refinement.
- **Blue Points:** Raw samples from the Diffusion Generator (Loop A).
- **Red Points:** The same samples after Energy Minimization (Loop B).
- **Interpretation:** The "funnel" effect demonstrates that the generator lands in the basin of attraction, and refinement successfully minimizes structures to the valid metastable state.

## 3. Control Experiment (Random vs Active Learning)
**Status:** Running in background.
**Log:** `control_experiment.log`
**Output Dir:** `../runs/day8_9_control`
**Goal:** Demonstrate that "Random Querying" (Control) fails to discover the basin as efficiently as "Uncertainty Sampling" (Active Learning).

## Scripts
- `viz_evolution.py`: Generates the evolution plot from existing AL data.
- `viz_funnel.py`: Generates the funnel plot by running generation + refinement.
- `run_control.sh`: Runs the Control Experiment (Active Learning Loop with `strategy=random`).
