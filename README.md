# Synthetic-Data-Generation-for-Pest-Detection


STA 561 Final Project

## Current Verified State

- This repository currently represents a **Phase 1 baseline plus a Phase 2 smoke-scale video extension**, not the full final assignment pipeline.
- Windows environment setup has been tested on this repository.
- A Conda environment named `pest-synth` is working with:
  - Python 3.10
  - PyTorch 2.6.0 + CUDA
  - `transformers`, `timm`, and `Pillow`
- Blender-based generation has been verified on Windows.
- A smoke synthetic dataset has been generated locally:
  - `data/generated/image_smoke_v1/`
  - `data/splits/image_smoke_v1/split.json`
- A video-generation MVP has been implemented locally:
  - `scripts/generate_synthetic_video_blender.py`
  - `data/generated/video_clip_smoke_v1/`
  - `scripts/flatten_video_frames_for_detection.py`
  - `data/generated/video_frame_smoke_v1/`
- Video-derived frame training has also been validated locally:
  - `data/splits/video_frame_smoke_v1/split.json`
  - `outputs/detr_video_frame_smoke_v1/`
- An upgraded video smoke dataset with negative sampling has been validated locally:
  - `data/generated/video_clip_smoke_v2/`
  - `data/generated/video_frame_smoke_v2/`
  - `data/splits/video_frame_smoke_v2/split.json`
  - `outputs/detr_video_frame_smoke_v2/`
- A longer-duration video smoke dataset has also been validated locally:
  - `data/generated/video_clip_smoke_v3/`
  - `data/generated/video_frame_smoke_v3/`
  - `data/splits/video_frame_smoke_v3/split.json`
  - `outputs/detr_video_frame_smoke_v3/`
  - `outputs/detr_video_frame_smoke_v3/eval_test.json`
- A curated-model refresh has also been validated locally:
  - `assets/models/MODEL_AUDIT.md`
  - `data/generated/video_clip_smoke_v4/`
  - `data/generated/video_frame_smoke_v4/`
  - `data/splits/video_frame_smoke_v4/split.json`
  - `outputs/detr_video_frame_smoke_v4/`
  - `outputs/detr_video_frame_smoke_v4/eval_test.json`
  - `scripts/diagnose_detr_predictions.py`
  - `outputs/detr_video_frame_smoke_v4/diagnostics/summary.json`
- A torchvision comparison baseline has also been validated locally:
  - `scripts/train_fasterrcnn.py`
  - `scripts/eval_fasterrcnn.py`
  - `scripts/diagnose_fasterrcnn_predictions.py`
  - `scripts/audit_detection_dataset.py`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/summary.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_balanced/`
  - `outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json`
  - `outputs/video_frame_smoke_v4_audit.json`
- The current recommended local detector baseline is:
  - Faster R-CNN on `video_frame_smoke_v4`
  - with `positive_anchor` batching
  - using `outputs/fasterrcnn_video_frame_smoke_v4_balanced/`

## Phase Scope

What this repository currently does:

- generates labeled synthetic pest images
- generates video-style frame sequences with per-frame labels, including 30-second smoke clips
- flattens generated video frames back into the current detection training format
- trains a DETR-based detector on those images
- trains a DETR-based detector on flattened video-derived frames
- trains a Faster R-CNN detector on flattened video-derived frames
- provides a working local baseline for data generation and training

What it does **not** yet fully do:

- generate 30-60 second labeled videos at assignment scale beyond the current smoke-scale runs
- adapt scene layout from a single user-provided kitchen image
- demonstrate final TPR/FPR target compliance on instructor test videos

For a formal requirement-by-requirement comparison, see:

- `REQUIREMENTS_GAP_ANALYSIS.md`
- `VIDEO_GENERATION_PLAN.md`

## Progress Summary

Current progress is:

- image generation pipeline working
- image-label dataset split working
- DETR smoke training working on static synthetic images
- video-generation MVP working for short clips
- duration-based clip generation working
- 30-second clip generation working
- negative clips and negative frames working
- frame flattening working for video-to-detection reuse
- DETR smoke training working on flattened video-derived frames
- DETR smoke training working on v2 video-derived frames with negative samples
- held-out evaluation working on v3 video-derived frames
- raw model audit and curated model-pool selection working
- curated-model v4 regeneration, flattening, split, and training working
- qualitative prediction overlays and threshold diagnostics working for v4
- current negative-sample smoke eval suppresses frame-level false positives, but recall remains poor on both v3 and v4
- v4 diagnostics show confidence collapse: top scores on positive and negative frames both cluster around `0.025`, so thresholds `>= 0.05` suppress every prediction
- v4 top-k diagnostics show ranking collapse too: `top-1` on test still has `0.0` recall while negative-frame FPR stays `1.0`
- Faster R-CNN comparison baseline works strongly on the same v4 split, which suggests the current held-out failure is more DETR-specific than data-pipeline-wide
- Faster R-CNN diagnostics also look healthy on v4: positive-frame top scores are high, negative-frame top scores are exactly `0.0`, and `top-1` predictions already recover perfect test recall at moderate thresholds
- v4 dataset audit shows that all splits are roughly `55%` negative frames, so with batch size `2` the expected all-negative batch rate is about `30%`, which is consistent with the skipped non-finite Faster R-CNN batches
- A positive-anchor Faster R-CNN batching strategy has now been validated locally: it eliminates skipped train/val batches entirely while preserving `1.0` held-out recall on v4

In short:

- the core rendering, labeling, flattening, and training loop now works end to end
- the current detector bottleneck is no longer general held-out recall on smoke-scale data
- the biggest remaining gap is scaling the video pipeline to longer clips, more clips, and assignment-scale runs while preserving the current Faster R-CNN baseline quality

## Current Outputs

Key currently generated artifacts:

- Static image smoke dataset:
  - `data/generated/image_smoke_v1/`
- Static image split:
  - `data/splits/image_smoke_v1/split.json`
- Static image DETR smoke training output:
  - `outputs/detr_image_smoke_v1/`
- Video smoke clip dataset:
  - `data/generated/video_clip_smoke_v1/`
- Flattened video-frame dataset:
  - `data/generated/video_frame_smoke_v1/`
- Flattened video-frame split:
  - `data/splits/video_frame_smoke_v1/split.json`
- Video-frame DETR smoke training output:
  - `outputs/detr_video_frame_smoke_v1/`
- Video smoke clip dataset v2:
  - `data/generated/video_clip_smoke_v2/`
- Flattened video-frame dataset v2:
  - `data/generated/video_frame_smoke_v2/`
- Flattened video-frame split v2:
  - `data/splits/video_frame_smoke_v2/split.json`
- Video-frame DETR smoke training output v2:
  - `outputs/detr_video_frame_smoke_v2/`
- Video smoke clip dataset v3:
  - `data/generated/video_clip_smoke_v3/`
- Flattened video-frame dataset v3:
  - `data/generated/video_frame_smoke_v3/`
- Flattened video-frame split v3:
  - `data/splits/video_frame_smoke_v3/split.json`
- Video-frame DETR smoke training output v3:
  - `outputs/detr_video_frame_smoke_v3/`
- Video-frame DETR smoke evaluation output v3:
  - `outputs/detr_video_frame_smoke_v3/eval_test.json`
- Video smoke clip dataset v4:
  - `data/generated/video_clip_smoke_v4/`
- Flattened video-frame dataset v4:
  - `data/generated/video_frame_smoke_v4/`
- Flattened video-frame split v4:
  - `data/splits/video_frame_smoke_v4/split.json`
- Video-frame DETR smoke training output v4:
  - `outputs/detr_video_frame_smoke_v4/`
- Video-frame DETR smoke evaluation output v4:
  - `outputs/detr_video_frame_smoke_v4/eval_test.json`
- Video-frame DETR diagnostics output v4:
  - `outputs/detr_video_frame_smoke_v4/diagnostics/summary.json`
  - `outputs/detr_video_frame_smoke_v4/diagnostics/test/overlays/top_positive/`
  - `outputs/detr_video_frame_smoke_v4/diagnostics/test/overlays/top_negative/`
- Video-frame DETR top-k diagnostics output v4:
  - `outputs/detr_video_frame_smoke_v4/diagnostics_topk/summary.json`
- Video-frame Faster R-CNN comparison output v4:
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test_thr020.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/summary.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/test/overlays/top_positive/`
  - `outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/test/overlays/top_negative/`
- Video-frame Faster R-CNN balanced-batching output v4:
  - `outputs/fasterrcnn_video_frame_smoke_v4_balanced/`
  - `outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json`
  - `outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test_thr020.json`
- Video-frame v4 dataset audit:
  - `outputs/video_frame_smoke_v4_audit.json`

## Repository Layout

```text
Synthetic-Data-Generation-for-Pest-Detection/
├── assets/
│   └── models/
│       ├── rat/                         # local .glb binaries + curated candidates
│       ├── mouse/
│       ├── cockroach/
│       ├── raw/                         # local raw model downloads
│       ├── MODEL_MANIFEST.md
│       ├── MODEL_AUDIT.md
│       └── sketchfab_model_catalog.csv
├── data/
│   ├── raw/                             # local background images, not tracked in git
│   ├── generated/                       # local generated datasets, not tracked in git
│   └── splits/                          # local split JSONs, not tracked in git
├── data_sources/
│   └── SYNTHETIC_DATA_SOURCES.md
├── outputs/                             # local checkpoints, evals, diagnostics, not tracked in git
├── scripts/
│   ├── generate_synthetic_blender.py
│   ├── generate_synthetic_video_blender.py
│   ├── flatten_video_frames_for_detection.py
│   ├── split_detection_dataset.py
│   ├── train_detr.py
│   ├── eval_detr.py
│   ├── diagnose_detr_predictions.py
│   ├── train_fasterrcnn.py
│   ├── eval_fasterrcnn.py
│   ├── diagnose_fasterrcnn_predictions.py
│   └── audit_detection_dataset.py
├── submissions/
│   └── README.md
├── environment.yml
├── README.md
├── PROJECT_STATUS.md
├── PROJECT_EXECUTION_PLAN.md
├── REQUIREMENTS_GAP_ANALYSIS.md
├── VIDEO_GENERATION_PLAN.md
├── WINDOWS_RUN_TROUBLESHOOTING.md
└── DOCUMENTATION_GUIDE.md
```

## Technical Flow

### Current Implemented Pipeline

```mermaid
flowchart LR
    A[Places365 kitchen backgrounds] --> B[Blender synthetic image generator]
    M[3D pest models] --> B
    B --> C[Static image dataset<br/>images + labels + metadata]
    C --> D[Dataset split]
    D --> E[DETR training]

    A --> F[Blender video MVP]
    M --> F
    F --> G[Clip folders<br/>frames + per-frame labels + clip metadata]
    G --> H[Flatten video frames<br/>to images + labels]
    H --> I[Dataset split]
    I --> J[DETR training on video-derived frames]
```

### Target Assignment Pipeline

```mermaid
flowchart LR
    A[Single kitchen photo] --> B[Scene/layout adaptation]
    B --> C[Video generation at clip scale]
    M[3D pest models] --> C
    C --> D[30-60 second clips]
    D --> E[Per-frame labels + metadata]
    E --> F[Train/val/test preparation]
    F --> G[Vision transformer training]
    G --> H[Evaluation on held-out test videos]
```

## Windows Notes

- `python ...` commands in this README work on Windows if you run them inside the `pest-synth` Conda environment.
- `bash ...` commands do **not** work in plain PowerShell unless you have Git Bash, WSL, or another Bash shell installed.
- On Windows, prefer:
  - `conda run -n pest-synth python ...`
  - calling Blender with its installed executable path, for example:
    - `"$env:ProgramFiles\Blender Foundation\Blender 5.1\blender.exe"`
- For `Places365` background extraction on Windows, use `--mode copy` instead of `--mode symlink`.

## Documentation Map

- Project status + runbook: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- Report-writing handoff: [REPORT_BRIEFING.md](REPORT_BRIEFING.md)
- Documentation guide: [DOCUMENTATION_GUIDE.md](DOCUMENTATION_GUIDE.md)
- Execution plan: [PROJECT_EXECUTION_PLAN.md](PROJECT_EXECUTION_PLAN.md)
- Data sources: [SYNTHETIC_DATA_SOURCES.md](data_sources/SYNTHETIC_DATA_SOURCES.md)
- Requirement gap analysis: [REQUIREMENTS_GAP_ANALYSIS.md](REQUIREMENTS_GAP_ANALYSIS.md)
- Video generation plan: [VIDEO_GENERATION_PLAN.md](VIDEO_GENERATION_PLAN.md)
- Windows rerun notes: [WINDOWS_RUN_TROUBLESHOOTING.md](WINDOWS_RUN_TROUBLESHOOTING.md)
- Submission notes: [submissions/README.md](submissions/README.md)
- Model manifest: [MODEL_MANIFEST.md](assets/models/MODEL_MANIFEST.md)
- Model audit: [MODEL_AUDIT.md](assets/models/MODEL_AUDIT.md)

## Start Here

Recommended reading order:

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. [REQUIREMENTS_GAP_ANALYSIS.md](REQUIREMENTS_GAP_ANALYSIS.md)
3. [VIDEO_GENERATION_PLAN.md](VIDEO_GENERATION_PLAN.md)
4. [WINDOWS_RUN_TROUBLESHOOTING.md](WINDOWS_RUN_TROUBLESHOOTING.md) if you are running locally on Windows

If you are primarily writing the report rather than running code:

1. [REPORT_BRIEFING.md](REPORT_BRIEFING.md)
2. [README.md](README.md)
3. [PROJECT_STATUS.md](PROJECT_STATUS.md)
4. [REQUIREMENTS_GAP_ANALYSIS.md](REQUIREMENTS_GAP_ANALYSIS.md)

## Team Workflow

What is tracked in git:

- source code
- environment/config files
- documentation
- lightweight manifests and catalogs that describe external assets
- report-briefing and status documents

What is intentionally **not** tracked in git:

- downloaded raw datasets
- generated image/video datasets
- generated split files
- checkpoints and training outputs
- downloaded 3D model binaries

That means teammates should expect to recreate local data artifacts after `git pull`.

## New Collaborator Quick Start

If a teammate pulls this repository onto a fresh machine, use this order:

1. Read `PROJECT_STATUS.md` for the current phase and known limitations.
2. Create the environment from `environment.yml`.
3. Install Blender.
4. Review `WINDOWS_RUN_TROUBLESHOOTING.md` if running on Windows.
5. Download the three required pest model files into:
   - `assets/models/rat/rat_primary.glb`
   - `assets/models/mouse/mouse_primary.glb`
   - `assets/models/cockroach/cockroach_primary.glb`
6. Download Places365 backgrounds and build `data/raw/kitchen_backgrounds/`.
7. Generate a smoke dataset locally before attempting larger runs.
8. Split the dataset and run smoke training.

Recommended local build order:

```text
environment -> blender -> models -> backgrounds -> generation -> split -> training
```

## Repository Contract

Use the repository like this:

- code and docs are shared through git
- datasets and checkpoints are local working artifacts
- if a path lives under `data/generated/`, `data/splits/`, or `outputs/`, assume it should be reproducible rather than committed

If you create a new generated dataset or checkpoint:

- document how to regenerate it
- do not rely on it being present after a teammate pulls the repo

## 1. Project Overview

In this project we build an automated pipeline that generates **synthetic pest detection datasets** from kitchen images and trains a **DETR-based vision transformer detector** to detect pests.

The system will:

1. Take a **single kitchen image** as input
2. Automatically generate **synthetic videos (30–60 seconds)** containing pests
3. Label each frame with:
   - pest type
   - bounding boxes
4. Train a **vision transformer detector (DETR)** to localize and classify pests
5. Evaluate the detector on **held-out test videos**

The goal is to investigate whether **synthetic data can effectively train pest detection models**.

---

# 2. Project Requirements (From Assignment)

The project must satisfy the following requirements:

### Input
- A still image of a **commercial kitchen**

### Synthetic Data Generation
The system must automatically generate videos with:

- pests (mice, rats, cockroaches)
- realistic movement
- randomized environment conditions

Video requirements:

- length: **30–60 seconds**
- frame-level annotations
- bounding boxes for pests
- labeled pest type

Generation must be:

- **fully automated**
- controlled via **Python**
- scalable to **thousands of videos on Duke compute cluster**

---

### Detection Model

Train a **vision transformer model (DETR)** to detect pests from generated videos.

The model must:

- identify pest type
- predict bounding boxes
- generalize to **unseen test videos**

Model choice for this repository:

- **DETR (Detection Transformer)** with a transformer-based detection head.
- The project objective is to train a vision transformer that identifies both pest location and pest type.

---

### Performance Target (For A+)

On instructor-provided test videos:

- **True detection rate ≥ 80%**
- **False positive rate < 5%**

---

# 3. Key Research Questions

Our project explores several research questions:

1. Can synthetic data effectively train pest detection models?
2. Which synthetic randomization strategies improve detection?
3. Does video-based generation outperform static image synthesis?
4. What factors reduce false positives in cluttered kitchen environments?

---

# 4. System Architecture

The system will consist of the following components:

---

## Quick Start: Synthetic Image Pipeline

### Environment Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate pest-synth
```

Windows users can also run project commands without activating first:

```powershell
conda run -n pest-synth python --version
```

### Step 1) Collect kitchen-only backgrounds from Places365

Download Places365 (small version) automatically:

```bash
python scripts/download_places365.py \
  --root data/raw/places365 \
  --split val \
  --small true
```

Then collect kitchen-only images:

```bash
python scripts/collect_places365_kitchen.py \
  --places-root data/raw/places365 \
  --out-dir data/raw/kitchen_backgrounds \
  --mode copy \
  --max-per-class 500
```

This creates:

- `data/raw/kitchen_backgrounds/<kitchen-class>/*.jpg`
- `data/raw/kitchen_backgrounds/manifest.csv`

### Step 2) Download 3D models from Sketchfab

Model candidates are listed in:

- `assets/models/sketchfab_model_catalog.csv`

Download one `rat`, one `mouse`, and one `cockroach` model file into `assets/models/`, for example:

- `assets/models/rat/rat_primary.glb`
- `assets/models/mouse/mouse_primary.glb`
- `assets/models/cockroach/cockroach_primary.glb`

### Step 3) Generate synthetic images with Blender

On DCC, this is a heavy stage; prefer submitting via `job.sbatch` (`STAGE=generate`) instead of running on a login node.

macOS / Linux style:

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

Windows PowerShell example:

```powershell
$blender = Join-Path $env:ProgramFiles "Blender Foundation\Blender 5.1\blender.exe"
& $blender --background --python scripts\generate_synthetic_blender.py -- --background-dir data\raw\kitchen_backgrounds --rat-model assets\models\rat\rat_primary.glb --mouse-model assets\models\mouse\mouse_primary.glb --cockroach-model assets\models\cockroach\cockroach_primary.glb --out-dir data\generated\image_smoke_v1 --num-images 60
```

Outputs:

- Images: `data/generated/image_smoke_v1/images/*.png`
- Normalized bbox labels (`class x_center y_center width height`): `data/generated/image_smoke_v1/labels/*.txt`
- Metadata: `data/generated/image_smoke_v1/metadata.csv`

## DCC Run Policy (Login Node vs Compute Node)

- Canonical script entrypoints remain under `scripts/`.
- Root-level wrappers:
  - `run.sh`: safe wrapper for lightweight/local usage.
  - `job.sbatch`: Slurm batch entrypoint for real DCC runs.
- For large outputs/checkpoints/generated data, prefer paths under `/work/$USER`.

### Lightweight (safe on login node)

Split only:

```bash
bash run.sh split \
  --data-dir /work/$USER/pest_synth/data/generated/image_smoke_v1 \
  --out-dir /work/$USER/pest_synth/data/splits/image_smoke_v1 \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### Real execution on DCC (recommended)

Train with defaults in `job.sbatch`:

```bash
sbatch job.sbatch
```

If your allocation requires explicit account/partition:

```bash
sbatch -A <ACCOUNT> -p <PARTITION> job.sbatch
```

Eval stage:

```bash
sbatch --export=ALL,STAGE=eval,WORK_ROOT=/work/$USER/pest_synth job.sbatch
```

Generate stage:

```bash
sbatch --export=ALL,STAGE=generate,WORK_ROOT=/work/$USER/pest_synth,NUM_IMAGES=500 job.sbatch
```

### Data source references

- `data_sources/SYNTHETIC_DATA_SOURCES.md`

---

## DETR Baseline (Train + Eval)

This pipeline trains and evaluates **DETR only**.
For DCC, run heavy stages on compute nodes (`sbatch`/`srun`), not login nodes.

### 1) Split synthetic dataset

```bash
python scripts/split_detection_dataset.py \
  --data-dir data/generated/image_smoke_v1 \
  --out-dir data/splits/image_smoke_v1 \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### 2) Train DETR

```bash
python scripts/train_detr.py \
  --data-dir data/generated/image_smoke_v1 \
  --split-json data/splits/image_smoke_v1/split.json \
  --output-dir outputs/detr_image_smoke_v1 \
  --epochs 5 \
  --batch-size 4 \
  --lr 1e-4
```

### 3) Evaluate DETR (TPR / FPR-style report)

```bash
python scripts/eval_detr.py \
  --data-dir data/generated/image_smoke_v1 \
  --split-json data/splits/image_smoke_v1/split.json \
  --checkpoint-dir outputs/detr_image_smoke_v1/best \
  --output-json outputs/detr_image_smoke_v1/eval_test.json \
  --confidence-threshold 0.5 \
  --iou-threshold 0.5
```
