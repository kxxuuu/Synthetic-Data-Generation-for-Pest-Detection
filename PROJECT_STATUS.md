# Project Status and Runbook

## 1) Current Status (as of April 20, 2026)

Project phase label:

- **Phase 1 baseline**: static-image synthetic data generation + DETR smoke training
- **Phase 2 smoke baseline**: video generation now includes 30-second clips, negative sampling, flattened frame training, held-out video-frame evaluation, demo MP4 assembly, and report-figure generation, but assignment-scale generation, scene adaptation from one kitchen photo, and final target metrics still remain
- **Current recommended detector baseline**: Faster R-CNN with `positive_anchor` batching on `video_frame_smoke_v5`

### Close-Out Readiness

The repository is now ready for close-out around the current local synthetic baseline.

- The main local detector baseline is stable and documented.
- Demo MP4s, diagnostics, and report-ready PNG figures all exist locally.
- The highest-value remaining work is no longer basic debugging. It is final packaging, write-up, and optional scale-up beyond the current smoke runs.

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
- DETR qualitative diagnostics utility implemented:
  - `scripts/diagnose_detr_predictions.py`
- Torchvision Faster R-CNN baseline utilities implemented:
  - `scripts/train_fasterrcnn.py`
  - `scripts/eval_fasterrcnn.py`
- Torchvision Faster R-CNN diagnostics and data-audit utilities implemented:
  - `scripts/diagnose_fasterrcnn_predictions.py`
  - `scripts/audit_detection_dataset.py`
- Demo/video-report utilities implemented:
  - `scripts/assemble_video_from_frames.py`
  - `scripts/plot_baseline_report_figures.py`
- README and ignore rules updated for this workflow.

### Already produced locally

- Synthetic sample dataset exists:
  - [image_smoke_v1](data/generated/image_smoke_v1/)
  - 60 images + 60 labels
- Split file generated:
  - [split.json](data/splits/image_smoke_v1/split.json)
  - train/val/test = 41/9/10
- Video smoke dataset exists:
  - [video_clip_smoke_v1](data/generated/video_clip_smoke_v1/)
  - 1 clip / 12 frames / per-frame labels
- Flattened video-frame detection dataset exists:
  - [video_frame_smoke_v1](data/generated/video_frame_smoke_v1/)
  - 12 flattened image-label pairs
- Video smoke dataset v2 exists:
  - [video_clip_smoke_v2](data/generated/video_clip_smoke_v2/)
  - 3 clips / duration-driven frame counts / includes negative clip and negative frames
- Flattened video-frame detection dataset v2 exists:
  - [video_frame_smoke_v2](data/generated/video_frame_smoke_v2/)
  - 24 flattened image-label pairs
- Video smoke dataset v3 exists:
  - [video_clip_smoke_v3](data/generated/video_clip_smoke_v3/)
  - 6 clips / 30 seconds each / 720 rendered frames total
  - includes all three pest classes, one fully negative clip, and negative frames inside positive clips
- Flattened video-frame detection dataset v3 exists:
  - [video_frame_smoke_v3](data/generated/video_frame_smoke_v3/)
  - 720 flattened image-label pairs
- Split file generated for v3:
  - [split.json](data/splits/video_frame_smoke_v3/split.json)
  - train/val/test = 503/108/109
- 3D primary assets confirmed:
  - `assets/models/rat/rat_primary.glb`
  - `assets/models/mouse/mouse_primary.glb`
  - `assets/models/cockroach/cockroach_primary.glb`
- DETR smoke training completed:
  - [best](outputs/detr_image_smoke_v1/best)
  - [train.log](outputs/detr_image_smoke_v1/train.log)
  - best validation loss observed at epoch 1
- DETR video-frame smoke training completed:
  - [best](outputs/detr_video_frame_smoke_v1/best)
  - [train.log](outputs/detr_video_frame_smoke_v1/train.log)
  - best validation loss observed at epoch 2
- DETR video-frame smoke training v2 completed:
  - [best](outputs/detr_video_frame_smoke_v2/best)
  - [train.log](outputs/detr_video_frame_smoke_v2/train.log)
  - best validation loss observed at epoch 3 (`0.913048`)
- DETR video-frame smoke training v3 completed:
  - [best](outputs/detr_video_frame_smoke_v3/best)
  - [train.log](outputs/detr_video_frame_smoke_v3/train.log)
  - best validation loss observed at epoch 3 (`4.420991`)
- DETR video-frame smoke evaluation v3 completed:
  - [eval_test.json](outputs/detr_video_frame_smoke_v3/eval_test.json)
  - at confidence threshold `0.5`: recall `0.0`, frame FPR `0.0`, negative test frames `36`
  - lower-threshold diagnostic also saved at [eval_test_thr020.json](outputs/detr_video_frame_smoke_v3/eval_test_thr020.json)
  - recall remained `0.0` at confidence threshold `0.2`
- Raw model downloads audited and curated:
  - accepted candidates documented in [MODEL_AUDIT.md](assets/models/MODEL_AUDIT.md)
  - generator now supports per-class model pools plus exclude lists
- Video smoke dataset v4 exists:
  - [video_clip_smoke_v4](data/generated/video_clip_smoke_v4/)
  - 9 clips / 30 seconds each / 1080 rendered frames total
  - curated rat, mouse, and cockroach model pools used
  - includes 2 fully negative clips plus negative frames inside positive clips
- Flattened video-frame detection dataset v4 exists:
  - [video_frame_smoke_v4](data/generated/video_frame_smoke_v4/)
  - 1080 flattened image-label pairs
- Split file generated for v4:
  - [split.json](data/splits/video_frame_smoke_v4/split.json)
  - train/val/test = 756/162/162
- DETR video-frame smoke training v4 completed:
  - [best](outputs/detr_video_frame_smoke_v4/best)
  - [train.log](outputs/detr_video_frame_smoke_v4/train.log)
  - best validation loss observed at epoch 2 (`1.075053`)
- DETR video-frame smoke evaluation v4 completed:
  - [eval_test.json](outputs/detr_video_frame_smoke_v4/eval_test.json)
  - at confidence threshold `0.5`: recall `0.0`, frame FPR `0.0`, negative test frames `89`
  - lower-threshold diagnostic also saved at [eval_test_thr020.json](outputs/detr_video_frame_smoke_v4/eval_test_thr020.json)
  - recall remained `0.0` at confidence threshold `0.2`
- DETR video-frame qualitative diagnostics v4 completed:
  - [summary.json](outputs/detr_video_frame_smoke_v4/diagnostics/summary.json)
  - top positive / negative overlays saved under [overlays](outputs/detr_video_frame_smoke_v4/diagnostics/test/overlays/)
  - top prediction scores on positive and negative frames both cluster around `0.025`
  - no predictions survive thresholds `0.05` and above
  - at threshold `0.01`, test recall rises to `0.4658`, but frame FPR becomes `1.0` and precision collapses to `0.0021`
  - this points to confidence-score collapse / no-object calibration failure rather than a pure "no boxes at all" failure
- DETR video-frame top-k diagnostics v4 completed:
  - [summary.json](outputs/detr_video_frame_smoke_v4/diagnostics_topk/summary.json)
  - test split at threshold `0.02`:
    - `top-1`: recall `0.0`, precision `0.0`, frame FPR `1.0`
    - `top-3`: recall `0.1233`, precision `0.0185`, frame FPR `1.0`
    - `top-20`: recall `0.2466`, precision `0.0060`, frame FPR `1.0`
  - test split at threshold `0.01`:
    - `top-1`: recall `0.0`, precision `0.0`, frame FPR `1.0`
    - `top-3`: recall `0.1233`, precision `0.0185`, frame FPR `1.0`
    - `top-20`: recall `0.2466`, precision `0.0056`, frame FPR `1.0`
  - this suggests the highest-ranked query is almost never the true pest, and limiting boxes per image does not solve the negative-frame failure
- Faster R-CNN comparison baseline on v4 completed:
  - [best](outputs/fasterrcnn_video_frame_smoke_v4_stable/best)
  - [train.log](outputs/fasterrcnn_video_frame_smoke_v4_stable/train.log)
  - [eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test.json)
  - [eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test_thr020.json)
  - [diagnostics summary](outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/summary.json)
  - best validation loss observed at epoch 3 (`0.945906`)
  - some non-finite training batches still occurred and were skipped, but training completed stably
  - at confidence threshold `0.5`: recall `1.0`, precision `1.0`, frame FPR `0.0`
  - at confidence threshold `0.2`: recall `1.0`, precision `0.9865`, frame FPR `0.0`
  - Faster R-CNN diagnostics on the same v4 test split show:
    - positive-frame top score quantiles: min `0.5637`, median `0.9830`, max `0.9986`
    - negative-frame top scores are `0.0` across the whole test split
    - `top-1` predictions remain perfect at threshold `0.2`
  - this strongly suggests the current v4 failure mode is DETR-specific rather than a global failure of the dataset or evaluation code
- Faster R-CNN positive-anchor batching run on v4 completed:
  - [best](outputs/fasterrcnn_video_frame_smoke_v4_balanced/best)
  - [train.log](outputs/fasterrcnn_video_frame_smoke_v4_balanced/train.log)
  - [eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json)
  - [eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test_thr020.json)
  - best validation loss observed at epoch 3 (`0.733513`)
  - batching strategy: `positive_anchor`
  - training and validation both completed with `skipped_train_batches=0` and `skipped_val_batches=0`
  - at confidence threshold `0.5`: recall `1.0`, precision `0.9865`, frame FPR `0.0`
  - at confidence threshold `0.2`: recall `1.0`, precision `0.9481`, frame FPR `0.0`
  - this is now the cleanest training-stability baseline, even though the earlier `stable` run remains slightly cleaner at the lower threshold
- Dataset audit for v4 completed:
  - [video_frame_smoke_v4_audit.json](outputs/video_frame_smoke_v4_audit.json)
  - train/val/test negative-frame rates are all about `55%`
  - with batch size `2`, the expected all-negative batch rate is about `30%`
  - this lines up with the skipped non-finite Faster R-CNN batches and suggests those skips are dominated by all-negative mini-batches rather than malformed labels
- Demo clips assembled as MP4:
  - [clip_000000 demo.mp4](data/generated/video_demo_v1/clips/clip_000000/demo.mp4)
  - [clip_000001 demo.mp4](data/generated/video_demo_v1/clips/clip_000001/demo.mp4)
  - [clip_000002 demo.mp4](data/generated/video_demo_v1/clips/clip_000002/demo.mp4)
- Video smoke dataset v5 exists:
  - [video_clip_smoke_v5](data/generated/video_clip_smoke_v5/)
  - 12 clips / 30 seconds each / 1440 rendered frames total
  - includes all three pest classes plus negative clips and negative frames
- Flattened video-frame detection dataset v5 exists:
  - [video_frame_smoke_v5](data/generated/video_frame_smoke_v5/)
  - 1440 flattened image-label pairs
- Split file generated for v5:
  - [split.json](data/splits/video_frame_smoke_v5/split.json)
  - train/val/test = 1008/216/216
- Dataset audit for v5 completed:
  - [video_frame_smoke_v5_audit.json](outputs/video_frame_smoke_v5_audit.json)
  - train/val/test negative-frame rates remain about `54%`
  - batch size `2` still implies about `29%` all-negative mini-batches under IID sampling
- Faster R-CNN positive-anchor batching run on v5 completed:
  - [best](outputs/fasterrcnn_video_frame_smoke_v5_balanced/best)
  - [train.log](outputs/fasterrcnn_video_frame_smoke_v5_balanced/train.log)
  - [summary.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/summary.json)
  - [eval_test.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/eval_test.json)
  - [eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/eval_test_thr020.json)
  - [diagnostics summary](outputs/fasterrcnn_video_frame_smoke_v5_balanced/diagnostics/summary.json)
  - best validation loss observed at epoch 3 (`0.831835`)
  - training and validation both completed with `skipped_train_batches=0` and `skipped_val_batches=0`
  - at confidence threshold `0.5`: recall `1.0`, precision `1.0`, frame FPR `0.0`
  - at confidence threshold `0.2`: recall `1.0`, precision `1.0`, frame FPR `0.0`
  - threshold diagnostics show positive top scores stay high while negative-frame top scores remain `0.0`
- Report/demo figures completed:
  - [report_figures_v4_comparison](outputs/report_figures_v4_comparison/)
  - [report_figures_v5_final](outputs/report_figures_v5_final/)
  - these now provide detector comparison, threshold sensitivity, and dataset composition PNGs for write-up use

### Not completed yet

- No final DETR evaluation report yet (`outputs/detr_image_smoke_v1/eval_test.json` pending).
- Current DETR video-frame evaluation is only a smoke diagnostic; held-out recall on both v3 and v4 test frames is still `0.0` at standard thresholds, while the Faster R-CNN comparison baseline succeeds strongly on the same v4 split.
- No assignment-scale 30-60 second video workflow yet beyond the current smoke-scale `v5` run.
- No assignment-scale multi-job video generation run yet.
- No single-photo kitchen layout adaptation yet.
- No evidence yet of meeting the final `TPR >= 80%` and `FPR < 5%` target on instructor-run test video data.
- No final report package (executive summary / FAQ / technical appendix) yet.

## 2) What To Do Next

1. Treat Faster R-CNN with `positive_anchor` batching on `video_frame_smoke_v5` as the main local detector baseline for the next project phase.
2. Keep the `positive_anchor` batching strategy as the training default unless a later sampler outperforms it without reintroducing non-finite skips.
3. Continue DETR diagnosis only if there is a strong reason to keep DETR as the primary model.
4. Push from the current 30-second smoke scale toward assignment-scale `30-60s` batch generation once the preferred detector is chosen.
5. Add scene adaptation from a single kitchen image or document a justified baseline approximation.
6. Prepare the final report package using the new demo clips, overlays, and report figures.
7. Evaluate the image smoke model only if that adds useful appendix material.

Reference docs:

- [REQUIREMENTS_GAP_ANALYSIS.md](REQUIREMENTS_GAP_ANALYSIS.md)
- [VIDEO_GENERATION_PLAN.md](VIDEO_GENERATION_PLAN.md)

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

Recommended shortcut for handoff instead of full regeneration:

1. share [video_frame_smoke_v5](data/generated/video_frame_smoke_v5/)
2. share [split.json](data/splits/video_frame_smoke_v5/split.json)
3. share [fasterrcnn_video_frame_smoke_v5_balanced](outputs/fasterrcnn_video_frame_smoke_v5_balanced/)
4. for report teammates, also share [video_demo_v1](data/generated/video_demo_v1/) and the figure folders under [outputs](outputs/)

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
  data/generated/image_smoke_v1 \
  60
```

Outputs:

- images: `data/generated/image_smoke_v1/images/*.png`
- labels: `data/generated/image_smoke_v1/labels/*.txt`
- metadata: `data/generated/image_smoke_v1/metadata.csv`

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
