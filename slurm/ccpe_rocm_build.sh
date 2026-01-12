#!/bin/bash
#SBATCH --job-name=ccpe-rocm-build
#SBATCH --ntasks=8
#SBATCH --time=00:50:00
#SBATCH --account=<YOUR_BUDGET_CODE>
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --export=none
#SBATCH --mem=10G

set -euo pipefail
export OMP_NUM_THREADS=1

# from ARCHER2 user guide
module use <SITE_SPECIFIC_MODULE_PATH>
module load ccpe/23.12/rocm/5.6.0

cd <YOUR_PROJECT_ROOT>

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo ""

build_bind_list () {
  local raw="$1"
  local fixed=""
  IFS=',' read -ra arr <<< "${raw}"

  for b in "${arr[@]}"; do
    local src="${b%%:*}"  # supports "src" and "src:dst"

    # explicitly drop broken Cray xpmem entries
    if [[ "${src}" == "/opt/cray/xpmem" || "${src}" == "/opt/cray/modulefiles/xpmem" ]]; then
      echo "Dropping Cray xpmem bind entry: ${b}" >&2
      continue
    fi

    if [[ -e "${src}" ]]; then
      fixed+="${fixed:+,}${b}"
    else
      echo "Skipping missing bind source: ${src}" >&2
    fi
  done

  # add real xpmem location if present
  if [[ -d /opt/xpmem ]]; then
    fixed+="${fixed:+,}/opt/xpmem"
    echo "Added XPMEM bind: /opt/xpmem" >&2
  else
    echo "Note: /opt/xpmem not found; continuing without XPMEM bind." >&2
  fi

  # IMPORTANT: only output the bind list on stdout
  printf "%s" "${fixed}"
}

echo "CCPE_BIND_ARGS (raw):"
echo "${CCPE_BIND_ARGS}" | tr ',' '\n'
echo ""

# Capture only the bind list (stdout). All logs go to stderr.
CCPE_BIND_FIXED="$(build_bind_list "${CCPE_BIND_ARGS}")"

echo ""
echo "CCPE_BIND_ARGS (fixed):"
echo "${CCPE_BIND_FIXED}" | tr ',' '\n'
echo ""

singularity exec --cleanenv \
  --bind "${CCPE_BIND_FIXED},${PWD}" \
  --env LD_LIBRARY_PATH=${CCPE_LD_LIBRARY_PATH} \
  ${CCPE_IMAGE_FILE} \
    ${CCPE_ROCM_BUILDER} ${PWD} mypyenv slurm/pip-install.sh

