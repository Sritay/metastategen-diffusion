#!/bin/bash
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --account=e760
#SBATCH --job-name=msgen-report
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

set -euo pipefail

module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0

cd /work/e760/e760/sritay/metastategen-diffusion

# Bind logic ...
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
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then PYBASE=\"\$d\"; break; fi
    done
    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"
    export PYTHONPATH=\"${PWD}\"

    echo 'Computing dihedrals for generated samples...'
    python3 scripts/compute_generated_dihedrals.py \
      --samples runs/day2_baseline/samples/samples.pt \
      --outdir runs/day2_baseline/reports

    echo 'Generating plots...'
    python3 scripts/report_day2.py \
      --gen-npz runs/day2_baseline/reports/generated_dihedrals.npz \
      --outdir runs/day2_baseline/reports
      
    echo 'Done. Check runs/day2_baseline/reports/'
  "
