#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --qos=gpu-shd
#SBATCH --account=e760
#SBATCH --job-name=msgen-ens-train
#SBATCH --time=01:00:00

# TRACE MODE: Print every command (-x), exit on error (-e), 
# exit on undefined vars (-u), catch pipe errors (-o pipefail)
set -euxo pipefail

# --- ERROR TRAP ---
# If the script exits with an error, this function will run to tell you where.
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

echo "=== STARTING DEBUG SCRIPT ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# 1. Environment Setup
echo "--- Loading Modules ---"
module use /work/y07/shared/archer2-lmod/others/dev
module load ccpe/23.12/rocm/5.6.0
module load cray-python
echo "Modules loaded."

REPO_ROOT="/work/e760/e760/sritay/metastategen-diffusion"
echo "Changing directory to REPO_ROOT: ${REPO_ROOT}"
if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "CRITICAL ERROR: REPO_ROOT does not exist!"
    exit 1
fi
cd "${REPO_ROOT}"

# 2. Variable Verification
echo "--- Verifying CCPE Variables ---"
# Check if CCPE variables from the module are actually set
if [[ -z "${CCPE_IMAGE_FILE:-}" ]]; then
    echo "CRITICAL ERROR: CCPE_IMAGE_FILE is undefined. Module load failed?"
    exit 1
fi
if [[ ! -f "${CCPE_IMAGE_FILE}" ]]; then
    echo "CRITICAL ERROR: Singularity image not found at: ${CCPE_IMAGE_FILE}"
    exit 1
fi
echo "Image found: ${CCPE_IMAGE_FILE}"

if [[ -z "${CCPE_BIND_ARGS:-}" ]]; then
    echo "WARNING: CCPE_BIND_ARGS is empty. Binding might be incomplete."
fi

# 3. Bind Logic
echo "--- Building Bind List ---"
build_bind_list () {
  local raw="$1"
  local fixed=""
  IFS=',' read -ra arr <<< "${raw}"
  for b in "${arr[@]}"; do
    local src="${b%%:*}"
    # [cite_start]Skip XPMEM as per original script logic [cite: 3, 4]
    if [[ "${src}" == "/opt/cray/xpmem" || "${src}" == "/opt/cray/modulefiles/xpmem" ]]; then continue; fi
    if [[ -e "${src}" ]]; then fixed+="${fixed:+,}${b}"; fi
  done
  if [[ -d /opt/xpmem ]]; then fixed+="${fixed:+,}/opt/xpmem"; fi
  printf "%s" "${fixed}"
}
CCPE_BIND_FIXED="$(build_bind_list "${CCPE_BIND_ARGS}")"
echo "Bind list created: ${CCPE_BIND_FIXED}"

# 4. GPU & Singularity Check
echo "--- Checking GPU (rocm-smi) ---"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi || echo "WARNING: rocm-smi returned non-zero exit code"
else
    echo "WARNING: rocm-smi command not found."
fi
echo "--- GPU Check Complete ---"

echo "--- Checking Singularity Executable ---"
if ! command -v singularity >/dev/null 2>&1; then
    echo "CRITICAL ERROR: 'singularity' command not found in PATH."
    exit 1
fi
singularity --version

# 5. Execution
echo "--- Launching Container ---"

# We use a HEREDOC with 'EOF' (quoted) to prevent the host shell from
# trying to expand variables intended for the container.
# Variables like $PWD are passed via the bind arguments or --env.

singularity exec --cleanenv \
  --bind "${CCPE_BIND_FIXED},${PWD}" \
  --env LD_LIBRARY_PATH="${CCPE_LD_LIBRARY_PATH}" \
  --env SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}" \
  --rocm \
  "${CCPE_IMAGE_FILE}" \
  bash <<'EOF'
    # --- INSIDE CONTAINER ---
    set -euxo pipefail
    echo "Container start: $(date)"
    echo "Container User: $(whoami)"
    echo "Container PWD: $PWD"

    # Verify script existence inside container
    if [[ ! -f "scripts/train_ensemble.py" ]]; then
        echo "CRITICAL ERROR: scripts/train_ensemble.py not found inside container at $PWD"
        ls -l scripts/
        exit 1
    fi

    # Python Setup
    echo "Searching for Python..."
    PYBASE=''
    # Using explicit loop to avoid expansion issues
    for d in "$PWD/mypyenv/python/"*; do
      if [[ -d "$d" && -d "$d/bin" ]]; then 
        PYBASE="$d"
        echo "Found Python environment: $PYBASE"
        break
      fi
    done

    if [[ -z "$PYBASE" ]]; then
        echo "CRITICAL ERROR: Could not find valid python env in $PWD/mypyenv/python/"
        echo "Listing $PWD/mypyenv/:"
        ls -F "$PWD/mypyenv/" || echo "mypyenv not found"
        exit 1
    fi

    export PYTHONUSERBASE="$PYBASE"
    export PATH="$PYBASE/bin:$PATH"

    # Determine Python Executable
    if [[ -x "/opt/cray/pe/python/3.11.5/bin/python3" ]]; then
        PY_EXE="/opt/cray/pe/python/3.11.5/bin/python3"
    elif command -v python3.11 >/dev/null; then
        PY_EXE=python3.11
    else
        PY_EXE=python3
    fi
    echo "Using Python Executable: $PY_EXE"

    # Set PYTHONPATH
    PYVER=$("$PY_EXE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    export PYTHONPATH="$PWD:$PYBASE/lib/python$PYVER/site-packages"
    echo "PYTHONPATH is: $PYTHONPATH"

    # Handle Array Task ID
    MEMBER_ARG=""
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "Detected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
      MEMBER_ARG="--member-idx $SLURM_ARRAY_TASK_ID"
    else
      echo "No SLURM_ARRAY_TASK_ID detected. Running single task."
    fi

    echo "--- Executing Python Script ---"
    "$PY_EXE" scripts/train_ensemble.py \
      --config configs/ala2_ensemble.yaml \
      ${MEMBER_ARG}
    
    echo "--- Python Script Finished ---"
EOF

echo "=== DEBUG SCRIPT FINISHED SUCCESSFULLY ==="