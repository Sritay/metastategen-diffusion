#!/bin/bash
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --account=<account_name>
#SBATCH --job-name=msgen-day1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=00:45:00

set -euo pipefail
export OMP_NUM_THREADS=1
export OMP_PLACES=cores

#from archer2 user guide
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0

cd /work/<...>/metastategen-diffusion

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo ""

build_bind_list () {
  local raw="$1"
  local fixed=""
  IFS=',' read -ra arr <<< "${raw}"

  for b in "${arr[@]}"; do
    local src="${b%%:*}"

    # Drop broken Cray xpmem entries (not present on these nodes)
    if [[ "${src}" == "/opt/cray/xpmem" || "${src}" == "/opt/cray/modulefiles/xpmem" ]]; then
      echo "Dropping Cray xpmem bind entry: ${b}" >&2
      continue
    fi

    # Keep only bind sources that exist on this node
    if [[ -e "${src}" ]]; then
      fixed+="${fixed:+,}${b}"
    else
      echo "Skipping missing bind source: ${src}" >&2
    fi
  done

  # Add actual xpmem location (present on ARCHER2)
  if [[ -d /opt/xpmem ]]; then
    fixed+="${fixed:+,}/opt/xpmem"
    echo "Added XPMEM bind: /opt/xpmem" >&2
  fi

  printf "%s" "${fixed}"
}

CCPE_BIND_FIXED="$(build_bind_list "${CCPE_BIND_ARGS}")"

echo ""
echo "CCPE_BIND_ARGS (fixed):"
echo "${CCPE_BIND_FIXED}" | tr ',' '\n'
echo ""

singularity exec --cleanenv \
  --bind "${CCPE_BIND_FIXED},${PWD}" \
  --env LD_LIBRARY_PATH=${CCPE_LD_LIBRARY_PATH} \
  ${CCPE_IMAGE_FILE} \
  bash -lc "
    set -euo pipefail
    cd ${PWD}

    # Detect CCPE-built python userbase directory (e.g., mypyenv/python/3.11.5)
    PYBASE=''
    for d in ${PWD}/mypyenv/python/*; do
      if [[ -d \"\$d\" && -d \"\$d/bin\" ]]; then
        if ls \"\$d/bin\"/pip3* >/dev/null 2>&1; then
          PYBASE=\"\$d\"
          break
        fi
      fi
    done
    if [[ -z \"\$PYBASE\" ]]; then
      echo 'ERROR: Could not find python userbase under mypyenv/python/* (expected 3.11.5 etc.)'
      ls -la ${PWD}/mypyenv/python || true
      exit 2
    fi

    # Use Cray Python inside container, but point it at the userbase
    export PYTHONUSERBASE=\"\$PYBASE\"
    export PATH=\"\$PYBASE/bin:\$PATH\"

    PYVER=\$(python3 -c 'import sys; print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")')

    # IMPORTANT: your package directory is at repo root: ${PWD}/metastategen (not src/)
    export PYTHONPATH=\"${PWD}:\$PYBASE/lib/python\${PYVER}/site-packages\"

    echo \"[day1] python3: \$(which python3)\"
    python3 -c 'import sys; print(sys.version)'
    python3 -c 'import numpy as np; print(\"numpy\", np.__version__)'
    python3 -c 'import torch; print(\"torch\", torch.__version__)'
    python3 -c 'import mdshare; print(\"mdshare\", getattr(mdshare, \"__version__\", \"(unknown)\"))' || true

    echo \"[day1] PYBASE=\$PYBASE\"
    echo \"[day1] PYTHONUSERBASE=\$PYTHONUSERBASE\"
    echo \"[day1] PYTHONPATH=\$PYTHONPATH\"

    # Sanity check: confirm imports from repo root work
    python3 -c 'import metastategen; import metastategen.eval; print(\"metastategen OK:\", metastategen.__file__)'

    echo '[day1] Step 1/3: download mdshare data'
    python3 scripts/get_mdshare_data.py --outdir data/raw --seed 0

    echo '[day1] Step 2/3: preprocess positions -> shards'
    python3 scripts/preprocess_positions.py \
      --positions-npz data/raw/alanine-dipeptide-3x250ns-heavy-atom-positions.npz \
      --dihedrals-npz data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz \
      --raw-dir data/raw \
      --outdir data/processed/ala2 \
      --stride 10 \
      --max-frames-per-traj 20000 \
      --shard-size 5000 \
      --seed 0

    echo '[day1] Step 3/3: reference plots (density + free energy)'
    python3 -m metastategen.cli report \
      --dihedrals data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz \
      --outdir reports/reference \
      --bins 180 \
      --kT 1.0 \
      --seed 0

    echo '[day1] Done.'
    echo 'Outputs:'
    echo '  reports/reference/reference_ramachandran_density.png'
    echo '  reports/reference/reference_free_energy.png'
    echo '  data/processed/ala2/shards/*.pt'
  "

