#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# bash scripts/run_generate_synthetic.sh \
#   /path/to/blender \
#   data/raw/kitchen_backgrounds \
#   assets/models/rat.glb \
#   assets/models/mouse.glb \
#   assets/models/cockroach.glb \
#   data/generated/synth_v1 \
#   200

if [ "$#" -lt 7 ]; then
  echo "Usage: $0 <blender_bin> <background_dir> <rat_model> <mouse_model> <cockroach_model> <out_dir> <num_images>"
  exit 1
fi

BLENDER_BIN="$1"
BACKGROUND_DIR="$2"
RAT_MODEL="$3"
MOUSE_MODEL="$4"
COCKROACH_MODEL="$5"
OUT_DIR="$6"
NUM_IMAGES="$7"

"$BLENDER_BIN" --background \
  --python scripts/generate_synthetic_blender.py -- \
  --background-dir "$BACKGROUND_DIR" \
  --rat-model "$RAT_MODEL" \
  --mouse-model "$MOUSE_MODEL" \
  --cockroach-model "$COCKROACH_MODEL" \
  --out-dir "$OUT_DIR" \
  --num-images "$NUM_IMAGES"
