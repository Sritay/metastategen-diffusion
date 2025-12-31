#!/bin/bash
# Helper script to install Day 10 dependencies on the cluster login node.
# Run this from the project root on the cluster:
#   source scripts/cluster_install_day10.sh

set -euo pipefail

echo "Loading modules..."
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python

# Auto-detect mypyenv userbase (copied from 20_train_gpu.sh)
echo "Detecting environment..."
PYBASE=''
# Check if mypyenv/python/* exists
if compgen -G "${PWD}/mypyenv/python/*" > /dev/null; then
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d "$d" && -d "$d/bin" ]]; then PYBASE="$d"; break; fi
    done
fi

if [[ -z "$PYBASE" ]]; then
    echo "ERROR: Could not find valid mypyenv/python/* directory with bin/. setup."
    echo "Please ensure you are in the project root and mypyenv is set up as before."
    return 1 2>/dev/null || exit 1
fi

export PYTHONUSERBASE="$PYBASE"
export PATH="$PYBASE/bin:$PATH"

echo "PYTHONUSERBASE set to: $PYTHONUSERBASE"
echo "Which pip: $(which pip)"

echo "Installing pandas and wandb..."
pip install pandas wandb

echo "Updating project install..."
pip install -e .

echo "Done! You can now submit SLURM jobs."
