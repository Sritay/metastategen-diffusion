---
layout: default
title: Analysis & Visualization
parent: Usage Guide
nav_order: 4
---

# Analysis & Visualization

[< Back: Training](training.md) | [Next: Configuration >](configuration.md)

---

We provide a suite of scripts in `scripts/analysis/` to verify the active learning and refinement results.

## 1. AL Density Evolution
visualize how the model's generated distribution covers the Ramachandran landscape over iterations.

```bash
python scripts/analysis/viz_density.py \
    --run runs/<experiment_name> \
    --iters 0,5,10,15,20 \
    --outdir analysis_output/al_density
```
*Note: Replace `<experiment_name>` with your actual AL run directory (e.g., `tiny_al_test`).*
*Outputs: `evolution_density.png` (Backbone P(x)), `evolution_data.csv`*

## 2. Cluster Analysis
Identify and count populations in the metastable basins (Alpha, Beta, C7eq, etc.).

```bash
python scripts/analysis/analyze_clusters.py \
    --data analysis_output/al_density/evolution_data.csv \
    --outdir analysis_output/clusters
```
*Outputs: `cluster_analysis.png` and summary CSVs.*

## 3. Refinement Funnel Plot
Visualize the "snapping" of coarse backbone structures into physical minima.

```bash
python scripts/analysis/viz_funnel.py \
    --run runs/<refinement_experiment_name> \
    --outdir analysis_output/refinement
```
*Note: Replace `<refinement_experiment_name>` with your sampling output directory (e.g., `tiny_sample_test`).*
*Outputs: `funnel_plot.png` (Initial vs Refined overlay).*

---

[< Back: Training](training.md) | [Next: Configuration >](configuration.md)
