#!/bin/bash
#SBATCH --job-name=install-day10
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --account=<YOUR_BUDGET_CODE>
#SBATCH --export=none
set -euo pipefail

# 1. Load Modules (Same as build script)
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python

cd /work/e760/e760/sritay/2_metastategen-diffusion

# Bind logic (Standard boilerplate)
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

# 2. Run Installation in Container
singularity exec --cleanenv \
  --bind "${CCPE_BIND_FIXED},${PWD}" \
  --env LD_LIBRARY_PATH=${CCPE_LD_LIBRARY_PATH} \
  --rocm \
  ${CCPE_IMAGE_FILE} \
  bash -lc "
    set -euo pipefail
    cd ${PWD}
    
    # Python Detection (Same as train_gpu.sh)
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then PYBASE=\"\$d\"; break; fi
    done
    
    if [[ -z \"\$PYBASE\" ]]; then
        echo \"ERROR: Could not find valid mypyenv/python/* directory.\"
        exit 1
    fi

    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"
    
    # Priority Python Selection
    if [[ -x \"/opt/cray/pe/python/3.11.5/bin/python3\" ]]; then
        PY_EXE=\"/opt/cray/pe/python/3.11.5/bin/python3\"
    elif command -v python3.11 >/dev/null; then
        PY_EXE=python3.11
    else
        PY_EXE=python3
    fi
    
    echo \"Using Python: \$PY_EXE\"
    echo \"User Base: \$PYTHONUSERBASE\"
    
    # Install Day 10 Dependencies
    echo \"Installing pandas and wandb...\"
    \$PY_EXE -m pip install pandas wandb
    
    # Re-install project in editable mode to be safe
    echo \"Updating project installation...\"
    \$PY_EXE -m pip install -e .
    
    echo \"Installation complete.\"
    \$PY_EXE -c 'import pandas; import wandb; print(\"Imports successful\")'
  "
