#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=e760
#SBATCH --job-name=msgen-control
#SBATCH --time=12:00:00
set -euo pipefail

module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python

cd /work/e760/e760/sritay/2_metastategen-diffusion

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

    PYMAJOR=\$(\$PY_EXE -c 'import sys; print(sys.version_info[0])')
    PYMINOR=\$(\$PY_EXE -c 'import sys; print(sys.version_info[1])')
    PYVER=\$PYMAJOR.\$PYMINOR
    export PYTHONPATH=\"${PWD}:${PWD}/src:\$PYBASE/lib/python\${PYVER}/site-packages\"

    # Important: To ensure exact comparison with Loop 3, we should ideally use the EXACT SAME data split.
    # Loop 3 created 'data/processed/ala2/split_3'.
    # If that folder exists on the cluster, we can just point to it in the config (done).
    # If we regenerate it with the same seed, it should be identical.
    # To be safe, let's regenerate it here to ensure it exists for this job, using the same parameters as Loop 3.
    
    echo 'Ensuring Data Split Exists (Same params as Loop 3)...'
    \$PY_EXE scripts/setup_al_split.py \
      --data-dir data/processed/ala2/shards \
      --out-dir data/processed/ala2/split_3 \
      --seed-size 5000 \
      --val-size 2000 \
      --val-traj 1

    echo 'Running Control Experiment (Random Acquisition)...'
    \$PY_EXE scripts/run_al_loop.py --config configs/ala2_control.yaml
  "
