---
layout: default
title: Verification (Small Runs)
parent: Usage Guide
nav_order: 5
---

# Verification (Small Runs)

To quickly verify that your installation is working correctly without waiting for full training loops, you can use these "tiny" configurations.

## 1. Tiny Training Run
Trains the diffusion model for 1 epoch on a small batch.

```bash
msgen train --config configs/ala2_tiny_train.yaml
```
*Expected Output*: Should run for ~10 seconds and save a checkpoint in `runs/date...`.

## 2. Tiny Active Learning Run
Runs 1 iteration of active learning with a minimal ensemble (2 members).

```bash
msgen al --config configs/ala2_tiny_al.yaml
```
*Expected Output*: Should cycle through candidates, acquisition, and retraining in <1 minute. Results in `runs/tiny_al_test`.

## 3. Tiny Refinement Run
Refines a small batch of 10 samples for 50 steps.

```bash
msgen sample \
    --diff-ckpt runs/tiny_al_test/members/m000/checkpoints/iter_00.pt \
    --force-ckpt models/pretrained/force_field.pt \
    --out-dir runs/tiny_sample_test \
    --n-samples 10 \
    --refinement-steps 50 \
    --warmup-steps 10 \
    --batch-size 10
```
*Note: This command requires the checkpoint from Step 2.*
