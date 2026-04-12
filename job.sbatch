#!/usr/bin/env bash
#SBATCH --job-name=pest-synth
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-common
#SBATCH --account=coursesf25
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

# Easy-to-edit DCC batch entrypoint.
# Default outputs are under /work/$USER.
# Stage options: split | train | eval | generate
# If your DCC project requires an account or different partition, submit with:
#   sbatch -A <ACCOUNT> -p <PARTITION> job.sbatch
STAGE="${STAGE:-train}"

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
WORK_ROOT="${WORK_ROOT:-/work/$USER/pest_synth}"
CONDA_ENV="${CONDA_ENV:-pest-synth}"

DATA_DIR="${DATA_DIR:-$WORK_ROOT/data/generated/synth_v1}"
SPLIT_OUT_DIR="${SPLIT_OUT_DIR:-$WORK_ROOT/data/splits/synth_v1}"
SPLIT_JSON="${SPLIT_JSON:-$SPLIT_OUT_DIR/split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/outputs/detr_synth_v1}"

# Generate stage vars (adjust paths as needed)
BLENDER_BIN="${BLENDER_BIN:-blender}"
BACKGROUND_DIR="${BACKGROUND_DIR:-$WORK_ROOT/data/raw/kitchen_backgrounds}"
RAT_MODEL="${RAT_MODEL:-$PROJECT_ROOT/assets/models/rat/rat_primary.glb}"
MOUSE_MODEL="${MOUSE_MODEL:-$PROJECT_ROOT/assets/models/mouse/mouse_primary.glb}"
COCKROACH_MODEL="${COCKROACH_MODEL:-$PROJECT_ROOT/assets/models/cockroach/cockroach_primary.glb}"
GENERATE_OUT_DIR="${GENERATE_OUT_DIR:-$WORK_ROOT/data/generated/synth_v1}"
NUM_IMAGES="${NUM_IMAGES:-200}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"

EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-$OUTPUT_DIR/eval_test.json}"
EVAL_CONF_THRESHOLD="${EVAL_CONF_THRESHOLD:-0.5}"
EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.5}"

mkdir -p "$WORK_ROOT" "$SPLIT_OUT_DIR" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

echo "STAGE=$STAGE"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WORK_ROOT=$WORK_ROOT"

case "$STAGE" in
  split)
    python scripts/split_detection_dataset.py \
      --data-dir "$DATA_DIR" \
      --out-dir "$SPLIT_OUT_DIR" \
      --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
      --seed 42
    ;;
  train)
    python scripts/train_detr.py \
      --data-dir "$DATA_DIR" \
      --split-json "$SPLIT_JSON" \
      --output-dir "$OUTPUT_DIR" \
      --epochs "$TRAIN_EPOCHS" \
      --batch-size "$TRAIN_BATCH_SIZE" \
      --lr "$TRAIN_LR" \
      --num-workers "$TRAIN_NUM_WORKERS"
    ;;
  eval)
    python scripts/eval_detr.py \
      --data-dir "$DATA_DIR" \
      --split-json "$SPLIT_JSON" \
      --checkpoint-dir "$OUTPUT_DIR/best" \
      --output-json "$EVAL_OUTPUT_JSON" \
      --confidence-threshold "$EVAL_CONF_THRESHOLD" \
      --iou-threshold "$EVAL_IOU_THRESHOLD"
    ;;
  generate)
    bash scripts/run_generate_synthetic.sh \
      "$BLENDER_BIN" \
      "$BACKGROUND_DIR" \
      "$RAT_MODEL" \
      "$MOUSE_MODEL" \
      "$COCKROACH_MODEL" \
      "$GENERATE_OUT_DIR" \
      "$NUM_IMAGES"
    ;;
  *)
    echo "Unknown STAGE='$STAGE' (expected: split|train|eval|generate)"
    exit 1
    ;;
esac
