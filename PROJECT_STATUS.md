# Project Status and Runbook

## 1) Current Status (as of March 13, 2026)

### Completed

- Project scope selected: **Synthetic Data Generation for Pest Detection**.
- Assignment requirements parsed and documented.
- Environment file added: `environment.yml` (`pest-synth`).
- Places365 background pipeline completed:
  - download script: `scripts/download_places365.py`
  - kitchen-only filter script: `scripts/collect_places365_kitchen.py`
  - collected backgrounds: `kitchen` + `restaurant_kitchen`
- 3D model assets organized under:
  - `assets/models/rat/`
  - `assets/models/mouse/`
  - `assets/models/cockroach/`
- Blender synthetic generator implemented:
  - script: `scripts/generate_synthetic_blender.py`
  - launcher: `scripts/run_generate_synthetic.sh`
  - output format: image + normalized bbox labels + metadata CSV
- Dataset split utility implemented:
  - `scripts/split_detection_dataset.py`
- DETR training and evaluation utilities implemented:
  - `scripts/train_detr.py`
  - `scripts/eval_detr.py`
- README and ignore rules updated for this workflow.

### Already produced locally

- Synthetic sample dataset exists:
  - `data/generated/synth_v1_smoke4/`
  - 200 images + 200 labels (balanced across 3 classes)
- Split file generated:
  - `data/splits/synth_v1_smoke4/split.json`
  - train/val/test = 140/30/30

### Not completed yet

- No confirmed trained checkpoint yet (`outputs/detr_smoke4/best` missing).
- No final DETR evaluation report yet (`eval_test.json` pending).
- No video-level synthetic generation workflow yet (current pipeline is image-level).
- No final report package (executive summary / FAQ / technical appendix) yet.

## 2) What To Do Next

1. Run DETR training to produce `outputs/detr_smoke4/best`.
2. Run evaluation and generate `outputs/detr_smoke4/eval_test.json`.
3. Inspect failure cases and adjust synthetic generation (scale/visibility/background realism).
4. Generate a larger dataset (`synth_v1`) and retrain.
5. Add video generation stage (30-60s) for assignment compliance.
6. Prepare write-up artifacts.

## 3) Initialization

```bash
conda env create -f environment.yml
conda activate pest-synth
```

Install Blender separately (example on macOS):

```bash
brew install --cask blender
which blender
```

## 4) Data Preparation

### 4.1 Download Places365 and keep kitchen backgrounds only

```bash
python scripts/download_places365.py \
  --root data/raw/places365 \
  --split train-standard \
  --small true

python scripts/collect_places365_kitchen.py \
  --places-root data/raw/places365 \
  --out-dir data/raw/kitchen_backgrounds \
  --mode symlink \
  --max-per-class 5000
```

### 4.2 Confirm 3D models exist

Required primary files:

- `assets/models/rat/rat_primary.glb`
- `assets/models/mouse/mouse_primary.glb`
- `assets/models/cockroach/cockroach_primary.glb`

## 5) Synthetic Data Generation

```bash
bash scripts/run_generate_synthetic.sh \
  blender \
  data/raw/kitchen_backgrounds \
  assets/models/rat/rat_primary.glb \
  assets/models/mouse/mouse_primary.glb \
  assets/models/cockroach/cockroach_primary.glb \
  data/generated/synth_v1 \
  200
```

Outputs:

- images: `data/generated/synth_v1/images/*.png`
- labels: `data/generated/synth_v1/labels/*.txt`
- metadata: `data/generated/synth_v1/metadata.csv`

## 6) DETR Training and Evaluation

### 6.1 Split dataset

```bash
python scripts/split_detection_dataset.py \
  --data-dir data/generated/synth_v1 \
  --out-dir data/splits/synth_v1 \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### 6.2 Train DETR

```bash
python scripts/train_detr.py \
  --data-dir data/generated/synth_v1 \
  --split-json data/splits/synth_v1/split.json \
  --output-dir outputs/detr_synth_v1 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4
```

### 6.3 Evaluate DETR

```bash
python scripts/eval_detr.py \
  --data-dir data/generated/synth_v1 \
  --split-json data/splits/synth_v1/split.json \
  --checkpoint-dir outputs/detr_synth_v1/best \
  --output-json outputs/detr_synth_v1/eval_test.json \
  --confidence-threshold 0.5 \
  --iou-threshold 0.5
```

Notes:

- Run training before evaluation.
- `eval_detr.py` now checks checkpoint directory validity explicitly.

## 7) Suggested Commit Scope for This Push

Include:

- scripts, docs, configs, manifests, README, `.gitignore`, `environment.yml`

Avoid including:

- generated data, model binaries, checkpoints, logs

Quick check:

```bash
git status --short
git status --short --ignored
```
