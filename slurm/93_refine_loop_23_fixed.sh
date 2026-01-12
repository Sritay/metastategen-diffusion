#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=<YOUR_BUDGET_CODE>
#SBATCH --job-name=refine-23-fix
#SBATCH --time=12:00:00
#SBATCH --output=slurm-refine-23-fix-%j.out

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
    
    echo 'Running Loop 23 Refinement (With Fixed Bond Constraints)...'
    
    \$PY_EXE scripts/sample_refined.py \\
      --diff-config configs/ala2_al_23_hpc.yaml \\
      --diff-ckpt runs/day11_al_23_hpc/members/m000/checkpoints/iter_20.pt \\
      --force-ckpt runs/energy_pairwise/best_model.pt \\
      --out-dir runs/loop_b_refinement_23_fixed \\
      --n-samples 10000 \\
      --batch-size 1000 \\
      --warmup-steps 1000 \\
      --keep-percent 0.01 \\
      --refinement-steps 50000 \\
      --step-size 1e-7
  "
