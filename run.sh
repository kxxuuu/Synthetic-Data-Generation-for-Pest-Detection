#!/usr/bin/env bash
set -euo pipefail

# Safe wrapper for lightweight local usage.
# Heavy stages (generate/train/eval) should run on compute nodes via sbatch/srun.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh split --data-dir <dir> --out-dir <dir> [split args...]
  bash run.sh train [train args...]
  bash run.sh eval [eval args...]
  bash run.sh generate <blender_bin> <background_dir> <rat_model> <mouse_model> <cockroach_model> <out_dir> <num_images>

Notes:
  - split: lightweight, safe on login node.
  - train/eval/generate: blocked on login node by default; use sbatch job.sbatch (recommended) or srun.
  - Set ALLOW_LOGIN_HEAVY=1 only for tiny smoke tests.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

stage="$1"
shift || true

is_heavy_stage() {
  [[ "$1" == "train" || "$1" == "eval" || "$1" == "generate" ]]
}

if is_heavy_stage "$stage" && [[ "${ALLOW_LOGIN_HEAVY:-0}" != "1" ]]; then
  echo "[SAFE-GUARD] '$stage' is treated as heavy and is blocked on login nodes."
  echo "Use: sbatch job.sbatch"
  echo "Or:  srun --pty <resources> bash"
  echo "If this is a tiny smoke test, re-run with ALLOW_LOGIN_HEAVY=1."
  exit 2
fi

case "$stage" in
  split)
    exec python scripts/split_detection_dataset.py "$@"
    ;;
  train)
    exec python scripts/train_detr.py "$@"
    ;;
  eval)
    exec python scripts/eval_detr.py "$@"
    ;;
  generate)
    exec bash scripts/run_generate_synthetic.sh "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
