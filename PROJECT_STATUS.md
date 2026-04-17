# Project Status and Runbook

## 1) Current Status (as of April 17, 2026)

Project phase label:

- **Phase 1 baseline**: static-image synthetic data generation + DETR smoke training
- **Transition to Phase 2**: video-generation MVP now exists, flattened video-frame training has been validated, but full assignment-scale video generation, scene adaptation from one kitchen photo, and final target metrics still remain

### Completed

- Project scope selected: **Synthetic Data Generation for Pest Detection**.
- Assignment requirements parsed and documented.
- Environment file added: `environment.yml` (`pest-synth`).
- Windows environment setup verified locally:
  - Conda environment created successfully
  - PyTorch GPU stack working on Windows
  - Blender installed and synthetic generation tested
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
- Blender video MVP implemented:
  - script: `scripts/generate_synthetic_video_blender.py`
  - output format: per-clip frames + per-frame labels + clip metadata + manifest
- Video frame flattening utility implemented:
  - `scripts/flatten_video_frames_for_detection.py`
- Dataset split utility implemented:
  - `scripts/split_detection_dataset.py`
- DETR training and evaluation utilities implemented:
  - `scripts/train_detr.py`
  - `scripts/eval_detr.py`
- README and ignore rules updated for this workflow.

### Already produced locally

- Synthetic sample dataset exists:
  - `data/generated/image_smoke_v1/`
  - 60 images + 60 labels
- Split file generated:
  - `data/splits/image_smoke_v1/split.json`
  - train/val/test = 41/9/10
- Video smoke dataset exists:
  - `data/generated/video_clip_smoke_v1/`
  - 1 clip / 12 frames / per-frame labels
- Flattened video-frame detection dataset exists:
  - `data/generated/video_frame_smoke_v1/`
  - 12 flattened image-label pairs
- 3D primary assets confirmed:
  - `assets/models/rat/rat_primary.glb`
  - `assets/models/mouse/mouse_primary.glb`
  - `assets/models/cockroach/cockroach_primary.glb`
- DETR smoke training completed:
  - `outputs/detr_image_smoke_v1/best`
  - `outputs/detr_image_smoke_v1/train.log`
  - best validation loss observed at epoch 1
- DETR video-frame smoke training completed:
  - `outputs/detr_video_frame_smoke_v1/best`
  - `outputs/detr_video_frame_smoke_v1/train.log`
  - best validation loss observed at epoch 2

### Not completed yet

- No final DETR evaluation report yet (`outputs/detr_image_smoke_v1/eval_test.json` pending).
- No final video-frame evaluation report yet.
- No full 30-60 second video workflow yet.
- No multi-clip or assignment-scale video generation run yet.
- No single-photo kitchen layout adaptation yet.
- No evidence yet of meeting the final `TPR >= 80%` and `FPR < 5%` target on instructor-run test video data.
- No final report package (executive summary / FAQ / technical appendix) yet.

## 2) What To Do Next

1. Run evaluation for both the image smoke model and the video-frame smoke model.
2. Extend the video MVP from short clips to longer 30-60 second clips.
3. Generate multiple clips and test batch scaling.
4. Add optional MP4 assembly for generated clips.
5. Inspect failure cases and adjust synthetic generation (scale/visibility/background realism).
6. Add scene adaptation from a single kitchen image or document a justified baseline approximation.
7. Prepare write-up artifacts.

Reference docs:

- `REQUIREMENTS_GAP_ANALYSIS.md`
- `VIDEO_GENERATION_PLAN.md`

## 3) Initialization

```bash
conda env create -f environment.yml
conda activate pest-synth
```

Install Blender separately.

Windows example:

```powershell
winget install -e --id BlenderFoundation.Blender
```

macOS example:

```bash
brew install --cask blender
which blender
```

## 3.1) Fresh Clone Expectations

After a teammate runs `git pull` or clones the repository fresh, these items should **not** be expected to exist locally:

- `data/raw/places365/`
- `data/raw/kitchen_backgrounds/`
- `data/generated/`
- `data/splits/`
- `outputs/`
- downloaded 3D model binaries under `assets/models/*/*.glb`

These are intentionally treated as reproducible local artifacts rather than shared git assets.

Teammates should recreate them in this order:

1. environment
2. Blender
3. pest model files
4. kitchen backgrounds
5. generation
6. split
7. training

## 4) Data Preparation

### 4.1 Download Places365 and keep kitchen backgrounds only

```bash
python scripts/download_places365.py \
  --root data/raw/places365 \
  --split val \
  --small true

python scripts/collect_places365_kitchen.py \
  --places-root data/raw/places365 \
  --out-dir data/raw/kitchen_backgrounds \
  --mode copy \
  --max-per-class 500
```

Windows note:

- `--mode copy` is recommended.
- The `val` split has been tested locally on Windows.

### 4.2 Confirm 3D models exist

Required primary files:

- `assets/models/rat/rat_primary.glb`
- `assets/models/mouse/mouse_primary.glb`
- `assets/models/cockroach/cockroach_primary.glb`

## 5) Synthetic Data Generation

Windows-tested command:

```powershell
$blender = Join-Path $env:ProgramFiles "Blender Foundation\Blender 5.1\blender.exe"
& $blender --background --python scripts\generate_synthetic_blender.py -- --background-dir data\raw\kitchen_backgrounds --rat-model assets\models\rat\rat_primary.glb --mouse-model assets\models\mouse\mouse_primary.glb --cockroach-model assets\models\cockroach\cockroach_primary.glb --out-dir data\generated\image_smoke_v1 --num-images 60
```

Cross-platform / Bash-style command:

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
  --data-dir data/generated/image_smoke_v1 \
  --out-dir data/splits/image_smoke_v1 \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### 6.2 Train DETR

```bash
python scripts/train_detr.py \
  --data-dir data/generated/image_smoke_v1 \
  --split-json data/splits/image_smoke_v1/split.json \
  --output-dir outputs/detr_image_smoke_v1 \
  --epochs 5 \
  --batch-size 4 \
  --lr 1e-4
```

### 6.3 Evaluate DETR

```bash
python scripts/eval_detr.py \
  --data-dir data/generated/image_smoke_v1 \
  --split-json data/splits/image_smoke_v1/split.json \
  --checkpoint-dir outputs/detr_image_smoke_v1/best \
  --output-json outputs/detr_image_smoke_v1/eval_test.json \
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
