#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=<YOUR_BUDGET_CODE>
#SBATCH --job-name=msgen-train
#SBATCH --time=00:30:00
set -euo pipefail

# 1. Load Modules
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python  # Explicitly load Python 3.11+ module

cd /work/e760/e760/sritay/2_metastategen-diffusion

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
    
    # 2. Detect Python Userbase
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then PYBASE=\"\$d\"; break; fi
    done
    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"
    
    # 3. Aggressive Python Detection
    # Priority 1: Known working Cray Python path
    # Priority 2: python3.11 in PATH
    # Priority 3: Default python3
    if [[ -x \"/opt/cray/pe/python/3.11.5/bin/python3\" ]]; then
        PY_EXE=\"/opt/cray/pe/python/3.11.5/bin/python3\"
    elif command -v python3.11 >/dev/null; then
        PY_EXE=python3.11
    else
        PY_EXE=python3
    fi
    
    echo \"Using Python Executable: \$PY_EXE\"
    \$PY_EXE --version
    
    # 4. Construct PYTHONPATH
    # We use the detected executable to ask for its version string (e.g., '3.11')
    PYVER=\$(\$PY_EXE -c 'import sys; print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")')
    export PYTHONPATH=\"${PWD}:\$PYBASE/lib/python\${PYVER}/site-packages\"
    
    echo \"PYTHONPATH: \$PYTHONPATH\"
    echo \"Checking imports...\"
    \$PY_EXE -c 'import yaml; print(\"yaml import OK\")'

    # 5. Run Training
    \$PY_EXE scripts/train_diffusion.py --config configs/ala2_day2.yaml
  "
