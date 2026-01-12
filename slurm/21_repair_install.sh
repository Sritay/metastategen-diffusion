#!/bin/bash
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --account=<YOUR_BUDGET_CODE>
#SBATCH --job-name=msgen-repair
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

set -euo pipefail

module use <SITE_SPECIFIC_MODULE_PATH>
module load ccpe/23.12/rocm/5.6.0

cd <YOUR_PROJECT_ROOT>

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
  ${CCPE_IMAGE_FILE} \
  bash -lc "
    set -euo pipefail
    cd ${PWD}
    
    # 1. Detect Python Userbase
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then PYBASE=\"\$d\"; break; fi
    done
    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"
    
    # 2. Identify correct python executable (Robus check)
    if [[ -x \"\$PYBASE/bin/python3.11\" ]]; then
        PY_EXE=python3.11
    elif [[ -x \"\$PYBASE/bin/python3.10\" ]]; then
        PY_EXE=python3.10
    else
        PY_EXE=python3
    fi
    
    echo \"Using python: \$(which \$PY_EXE)\"
    \$PY_EXE --version
    
    # 3. Install missing packages
    echo 'Installing missing PyYAML and Scipy...'
    \$PY_EXE -m pip install PyYAML scipy
    
    # 4. Verification
    echo 'Verifying imports...'
    \$PY_EXE -c 'import yaml; print(\"yaml OK\")'
    \$PY_EXE -c 'import scipy; print(\"scipy OK\")'
  "
