#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=e760
#SBATCH --job-name=msgen-refine
#SBATCH --time=00:30:00
set -euo pipefail

# 1. Load Modules
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python

cd /work/e760/e760/sritay/metastategen-diffusion

# Bind logic
build_bind_list () {
  local raw="$1"
  local fixed=""
  IFS=',' read -ra arr <<< "${raw}"
  for b in "${arr[@]}"; do
    local src="${b%%:*}"
    if [[ "${src}" == "/opt/cray/xpmem" || "${src}" == "/opt/cray/modulefiles/xpmem" ]]; then continue; fi
    if [[ -e "${src}" ]]; then fixed+="${fixed:+,}${b}"; fi
  done
  if [[ -d /opt/xpmem ]]; then fixed+="${fixed:+,}/opt/xpmem"; fi
  printf "%s" "${fixed}"
}
CCPE_BIND_FIXED="$(build_bind_list "${CCPE_BIND_ARGS}")"

singularity exec --cleanenv \
  --bind "${CCPE_BIND_FIXED},${PWD}" \
  --env LD_LIBRARY_PATH=${CCPE_LD_LIBRARY_PATH} \
  --rocm \
  ${CCPE_IMAGE_FILE} \
  bash -lc "
    set -euo pipefail
    cd ${PWD}
    
    # Python setup
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then PYBASE=\"\$d\"; break; fi
    done
    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"
    
    if [[ -x \"/opt/cray/pe/python/3.11.5/bin/python3\" ]]; then
        PY_EXE=\"/opt/cray/pe/python/3.11.5/bin/python3\"
    elif command -v python3.11 >/dev/null; then
        PY_EXE=python3.11
    else
        PY_EXE=python3
    fi
    
    # Run Refinement
    # Warning: Ensure checkpoints exist. 
    # If day5 ensemble exists, use it. Otherwise day2.
    
    DIFF_CKPT='runs/day2_baseline/checkpoints/final.pt'
    FORCE_CKPT='runs/day10_force/checkpoints/latest.pt' # Check path in train_force.py (runs/force_surrogate_ala2)
    # Actually wait, train_force.py uses cfg['experiment_name'] which is 'force_surrogate_ala2' in my config.
    # So path is runs/force_surrogate_ala2/checkpoints/latest.pt
    # I should update FORCE_CKPT below.
    
    FORCE_CKPT='runs/force_surrogate_ala2/checkpoints/latest.pt'

    echo \"Running Refinement Sampling...\"
    \$PY_EXE scripts/sample_refined.py \
        --diff-ckpt \$DIFF_CKPT \
        --force-ckpt \$FORCE_CKPT \
        --refinement-steps 100 \
        --out-dir runs/day10_force/samples
  "
