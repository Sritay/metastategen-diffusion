# Flagship Demo Artifacts

This folder contains the assets and scripts for the flagship demonstration of the Active Learning (Loop A) and Refinement (Loop B) pipeline.

## 1. Density Evolution
**Files:** `evolution_density.pdf` / `evolution_density.png`
**Description:** Visualizes the progression of the generative model's distribution across 10 Active Learning iterations (Iter 0, 3, 6, 10).
- **Log Scale Hexbin:** Shows the density of generated structures in Phi-Psi space.
- **Overlays:** **Ground Truth Filled Contours** (Grey topographical map) show the target validation density.
- **Goal:** Demonstrate the model evolving from a random/collapsed initial state to fully covering the validation density (no mode collapse).
- **Key Insight:** Iterations show the "filling in" of the conformational space.

## 2. Acquisition Strategy ("Making AL Visible")
**Files:** `acquisition_strategy.pdf` / `acquisition_strategy.png`
**Description:** Visualizes the "Active Learning Loop" in action.
- **Black Points (Seed):** The initial random training set (fixed).
- **Green Points (History):** Points acquired in *previous* iterations (cumulative Model knowledge).
- **Red Points (New Batch):** The new batch selected *in this iteration*.
- **Overlays:** Validation Set Contours (Ground Truth).
- **Goal:** Watch the "Green" region grow to fill gaps, while "Red" points attack the remaining white space (high uncertainty).

## 3. Uncertainty Landscape
**Files:** `uncertainty_map.pdf` / `uncertainty_map.png`
**Description:** Maps the model's predictive uncertainty ($Var[\epsilon_\theta]$) across the conformational space.
- **Heatmap:** Brighter colors = Higher model uncertainty.
- **Overlays:** Validation Set Contours (Ground Truth).
- **Goal:** Show that uncertainty is high in unexplored regions (driving acquisition) and decreases in explored regions over time.
- **Key Insight:** A late spike in uncertainty (e.g., Iter 9) indicates the model discovering a completely new region of phase space.

## 4. Learning Curves
**Files:** `learning_curves.pdf` / `learning_curves.png`
**Description:** Quantitative metrics of success.
- **Basin Discovery:** Fraction of known metastable states found (1.0 = All found).
- **KL Divergence:** How closely the generated distribution matches the validation ground truth (lower is better).

## 5. Validation Density
**File:** `val_density_loop5.png` (Reference only)
**Description:** The "Ground Truth" or "Answer Key".
- Shows the distribution of the Validation Set used for evaluating the model.
- Essential for confirming that the validation target itself is valid (covers multiple basins) and not trivial.

## 6. Refinement Funnel
**File:** `funnel_plot.png` (Generating...)
**Description:** Visualizes the effect of the "Loop B" Force Refinement.
- **Blue Points:** Raw samples from the Diffusion Generator (Loop A).
- **Red Points:** The same samples after Energy Minimization (Loop B).
- **Interpretation:** The "funnel" effect demonstrates that the generator lands in the basin of attraction, and refinement successfully minimizes structures to the valid metastable state.

## 7. Cluster Analysis (Automated Basin Discovery)
**Files:** `cluster_analysis.pdf` / `cluster_analysis.png`
**Description:** Uses DBSCAN clustering to automatically identify and label the dense regions of the generated distribution.
- **Overlays:** Validation Set Contours (Ground Truth).
- **Algorithm:** DBSCAN on circular Phi/Psi coordinates.
- **Key finding:** Can track the "Center of Mass" of generation shifting from barriers (Iter 6) to stable basins (Iter 10).
- **Top Clusters found (Iter 10):**
    - **#1:** (-170, 4) -> **Beta/C5**
    - **#3:** (-41, 145) -> **Beta/PPII**
    - **#4:** (4, -163) -> **Beta** 
- **Physical Interpretation:**
    - **Iter 0:** Random/Unphysical.
    - **Iter 6:** Exploring High-Energy Barriers (uncertainty driven).
    - **Iter 10:** Settled into valid metastable basins.

## Scripts
- `viz_density.py`: Generates density evolution plots.
- `viz_acquired.py`: Generates acquisition strategy plots.
- `viz_uncertainty_map.py`: Generates uncertainty heatmaps.
- `viz_metrics.py`: Generates scalar metric plots.
- `viz_funnel.py`: Generates the funnel plot.
- `analyze_clusters.py`: Performs DBSCAN clustering and prints basin centers.
