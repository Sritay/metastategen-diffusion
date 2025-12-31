#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=e760
#SBATCH --job-name=msgen-report
#SBATCH --time=00:50:00
set -euo pipefail

EXP_ID="${EXP_ID:-day5_ensemble_k3}"
REPO_ROOT="/work/e760/e760/sritay/metastategen-diffusion"

BASELINE_SAMPLES="${BASELINE_SAMPLES:-runs/day2_baseline/samples/samples.pt}"
# Default "pooled" samples to consensus output to avoid missing naive runs.
POOLED_DIR="${POOLED_DIR:-runs/${EXP_ID}/samples/consensus}"
# Optional secondary consensus set for comparison (set to a path explicitly if needed).
CONSENSUS_DIR="${CONSENSUS_DIR:-}"

module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python

cd "${REPO_ROOT}"

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

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || echo "WARNING: rocm-smi returned non-zero exit code"
fi

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

    PYVER=\$(\$PY_EXE -c 'import sys; print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")')
    export PYTHONPATH=\"${PWD}:\$PYBASE/lib/python\${PYVER}/site-packages\"

    # Cluster basins
    \$PY_EXE scripts/cluster_samples.py --samples ${BASELINE_SAMPLES} --outdir runs/${EXP_ID}/reports/baseline_clusters
    \$PY_EXE scripts/cluster_samples.py --samples ${POOLED_DIR}/samples.pt --outdir runs/${EXP_ID}/reports/pooled_clusters

    CONSENSUS_ARG=\"\"
    if [[ -n \"${CONSENSUS_DIR}\" && -f \"${CONSENSUS_DIR}/samples.pt\" ]]; then
      \$PY_EXE scripts/cluster_samples.py --samples ${CONSENSUS_DIR}/samples.pt --outdir runs/${EXP_ID}/reports/consensus_clusters
      CONSENSUS_ARG=\"--ensemble-consensus-clusters runs/${EXP_ID}/reports/consensus_clusters/clusters.json\"
    fi

    # Uncertainty evaluation + triage
    \$PY_EXE scripts/eval_uncertainty.py \
      --uncertainty ${POOLED_DIR}/uncertainty.pt \
      --samples ${POOLED_DIR}/samples.pt \
      --out runs/${EXP_ID}/reports/uncertainty_bins.csv

    \$PY_EXE scripts/triage_and_report.py \
      --uncertainty ${POOLED_DIR}/uncertainty.pt \
      --samples ${POOLED_DIR}/samples.pt \
      --out runs/${EXP_ID}/reports/triage_curve.csv

    # Plots
    \$PY_EXE scripts/report_day5_7.py \
      --uncertainty-bins runs/${EXP_ID}/reports/uncertainty_bins.csv \
      --triage-curve runs/${EXP_ID}/reports/triage_curve.csv \
      --baseline-clusters runs/${EXP_ID}/reports/baseline_clusters/clusters.json \
      --ensemble-pooled-clusters runs/${EXP_ID}/reports/pooled_clusters/clusters.json \
      \${CONSENSUS_ARG} \
      --exp-id ${EXP_ID}
  "
