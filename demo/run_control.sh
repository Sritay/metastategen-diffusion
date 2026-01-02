#!/bin/bash
# Run the control experiment
PYTHONPATH=src python scripts/run_al_loop.py --config configs/ala2_control_lite.yaml > demo/control_experiment.log 2>&1
