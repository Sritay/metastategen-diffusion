#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Deterministic seed for day1
SEED="${SEED:-0}"

echo "[day1] Installing editable package (if not already installed)..."
python -m pip install -e . >/dev/null

echo "[day1] Step 1/3: Download mdshare Ala2 artifacts"
python scripts/get_mdshare_data.py --outdir data/raw --seed "${SEED}"

echo "[day1] Step 2/3: Preprocess heavy-atom positions into PyTorch shards"
python scripts/preprocess_positions.py \
  --positions-npz data/raw/alanine-dipeptide-3x250ns-heavy-atom-positions.npz \
  --dihedrals-npz data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz \
  --raw-dir data/raw \
  --outdir data/processed/ala2 \
  --stride "${STRIDE:-10}" \
  --max-frames-per-traj "${MAX_FRAMES_PER_TRAJ:-20000}" \
  --shard-size "${SHARD_SIZE:-5000}" \
  --seed "${SEED}"

echo "[day1] Step 3/3: Generate reference plots (density + free energy)"
msgen report \
  --dihedrals data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz \
  --outdir reports/reference \
  --bins "${BINS:-180}" \
  --kT 1.0 \
  --seed "${SEED}"

echo "[day1] Done. Outputs:"
echo "  - reports/reference/reference_ramachandran_density.png"
echo "  - reports/reference/reference_free_energy.png"
echo "  - data/processed/ala2/shards/*.pt"

