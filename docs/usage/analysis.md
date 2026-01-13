---
layout: default
title: Analysis & Visualization
parent: Usage Guide
nav_order: 3
---

# Analysis & Visualization

We provide a suite of scripts in `scripts/analysis/` to verify the active learning and refinement results.

## 1. AL Density Evolution
visualize how the model's generated distribution covers the Ramachandran landscape over iterations.

```bash
python scripts/analysis/viz_density.py \
    --run runs/day11_al_23_hpc \
    --iters 0,5,10,15,20 \
    --outdir analysis_output/al_density
```
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
    --run runs/loop_b_refinement_23_fixed \
    --outdir analysis_output/refinement
```
*Outputs: `funnel_plot.png` (Initial vs Refined overlay).*
