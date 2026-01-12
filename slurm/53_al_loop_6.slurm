#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=e760
#SBATCH --job-name=msgen-al-6
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

# Estimated runtime reasoning:
# Similar to Loop 5 but with scale fix.
# Should fit comfortably in 12h.

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

    # We can reuse the split from loop 5 if desired, or create a 'split_6'
    echo 'Creating new data split (AL Loop 6)...'
    \$PY_EXE scripts/setup_al_split.py \
      --data-dir data/processed/ala2/shards \
      --out-dir data/processed/ala2/split_6 \
      --seed-size 5000 \
      --val-size 2000 \
      --val-traj -1

    echo 'Running Active Learning Loop 6 (Scaled)...'
    # Important: Ensure config points to split_6 if you changed it above.
    # I will patch the config file to use split_6 in the next step or assume user wants to reuse split 5.
    # Better to stick to split 5 for now as the config I wrote uses split 5.
    # WAIT, I wrote the config to use split 5.
    # Let's adjust this script to use split 5 to match the config, 
    # OR update the config to use split 6.
    # Let's use split 6 for a clean slate. I will update the config in a moment.

    # RERUN SETUP FOR SPLIT 6
    # And run with updated config
    \$PY_EXE scripts/run_al_loop.py --config configs/ala2_al_6_hpc.yaml
  "
