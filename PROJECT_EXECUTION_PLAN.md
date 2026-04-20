# Synthetic Data Generation for Pest Detection - Execution Plan

## 1. Project Goal and Grading Targets

Based on `final_project_2026-1.pdf`:

- Input: one kitchen photo.
- Output: automated pipeline that generates labeled pest videos (30-60 seconds).
- Labels: per-frame pest category (`mouse`, `rat`, `cockroach`) + bounding boxes.
- Scale: feasible to generate thousands of videos on cluster.
- A+ target:
  - True detection rate >= 80%
  - False positive rate < 5%
  - End-to-end pipeline including training a ViT-based detector.

## 1.1 Current Checkpoint (April 20, 2026)

What is already working:

- static synthetic image generation with automatic bbox labels
- video-style clip generation with per-frame labels
- clip sharding controls (`start_clip`, duration-derived frame counts)
- negative clips and negative frames
- demo MP4 assembly from rendered frame folders
- flattening video frames back into the detector training format
- DETR smoke training on image data and video-derived frame data
- held-out video-frame evaluation

What the latest smoke-scale video runs proved:

- [video_clip_smoke_v4](data/generated/video_clip_smoke_v4/) generated 9 curated-model clips at 30 seconds each
- all three pest classes were covered, with curated model pools and 2 fully negative clips
- negative samples were included and still suppressed false positives on the held-out negative frames
- training quality improved substantially (`best val loss = 1.075053` on [detr_video_frame_smoke_v4](outputs/detr_video_frame_smoke_v4/))
- threshold diagnostics now show that positive and negative frames both receive top scores around `0.025`
- below `0.01`, recall partially recovers but precision collapses and every negative frame receives predictions
- top-k diagnostics now show that even `top-1` predictions on test have `0.0` recall while negative-frame FPR remains `1.0`
- a torchvision Faster R-CNN comparison baseline reaches perfect held-out recall on the same v4 split, so the current bottleneck has narrowed mostly to model choice and DETR-specific behavior rather than the old obviously broken raw assets
- Faster R-CNN diagnostics now show strong score separation on the same v4 split: positive top scores stay high while negative-frame top scores are `0.0`
- the v4 audit also shows all splits are roughly `55%` negative frames, so batch size `2` naturally creates about `30%` all-negative mini-batches, which likely explains most skipped non-finite Faster R-CNN batches
- a `positive_anchor` batching strategy now removes skipped train/val batches entirely while preserving `1.0` recall and `0.0` frame FPR at threshold `0.5`
- [video_clip_smoke_v5](data/generated/video_clip_smoke_v5/) extends the smoke run to 12 clips / 1440 frames while preserving 30-second duration and negative sampling
- [video_frame_smoke_v5](data/generated/video_frame_smoke_v5/) and [split.json](data/splits/video_frame_smoke_v5/split.json) now provide a larger held-out synthetic benchmark
- [fasterrcnn_video_frame_smoke_v5_balanced](outputs/fasterrcnn_video_frame_smoke_v5_balanced/) keeps `positive_anchor` batching stable with zero skipped train/val batches
- v5 test metrics remain perfect at both [eval_test.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/eval_test.json) and [eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/eval_test_thr020.json)
- [video_demo_v1](data/generated/video_demo_v1/) now includes 30-second MP4 demo clips for rat, mouse, and cockroach
- report-ready figures now exist at [report_figures_v4_comparison](outputs/report_figures_v4_comparison/) and [report_figures_v5_final](outputs/report_figures_v5_final/)

What remains highest priority:

- lock Faster R-CNN with `positive_anchor` batching as the working detector baseline unless DETR remains a hard requirement
- scale from the current `v5` smoke-scale 30-second clips to assignment-scale 30-60 second batches
- keep all-negative mini-batch handling explicit during detector training and cluster runs
- add scene adaptation from a single kitchen photo
- move from local smoke runs to cluster-ready generation and training jobs
- fold the current demo clips, overlays, and report figures into the final write-up package

## 2. Preconditions Checklist

## 2.1 Data and Assets

- [ ] Kitchen photo set (initial 20-50 images; later 200+).
- [ ] 3D pest assets for all 3 classes (multiple poses/variants preferred).
- [ ] License records for all external assets.

## 2.2 Environment and Compute

- [ ] Blender + Python automation (`bpy`) available.
- [ ] Training environment (PyTorch + detection framework).
- [ ] Cluster job submission script (batch generation + training).
- [ ] Storage estimate and budget prepared (images/videos/checkpoints).

## 2.3 Engineering Standards

- [ ] Label format fixed (COCO or YOLO; pick one and freeze).
- [ ] Reproducibility: config files + random seeds + run logs.
- [ ] Git workflow and code review rules agreed.

## 3. 4-Week Execution Plan

Original planning date: **March 11, 2026**.

## Week 1 (March 11 - March 17, 2026): MVP Static Synthetic Images

Goals:

- Define data schema, classes, metrics, and baseline model.
- Build first synthetic image generator (no video yet).
- Export correct labels automatically.

Deliverables:

- `kitchen_photo -> synthetic image + bbox labels` pipeline.
- At least 500 synthetic images across 3 classes.
- Baseline training run and first metrics report.

Acceptance criteria:

- Label visualization spot-check passes on >= 95% sampled images.
- End-to-end command runs without manual UI operations.

## Week 2 (March 18 - March 24, 2026): Video Generation + Frame Labels

Goals:

- Extend image generation to 30-60s video clips.
- Add realistic pest motion patterns and occlusion variation.
- Ensure per-frame labels are exported correctly.

Deliverables:

- At least 100 labeled videos generated automatically.
- Frame-level annotation integrity checker.
- First train/val split and data cards.

Acceptance criteria:

- Per-frame annotation pass rate >= 95% on sampled clips.
- Generation throughput measured and documented.

## Week 3 (March 25 - March 31, 2026): Model Iteration and Error Reduction

Goals:

- Train ViT-based detector (or ViT backbone detector).
- Run ablations on synthetic variability (lighting, scale, clutter, motion).
- Reduce false positives systematically.

Deliverables:

- Main training pipeline + evaluation script.
- Error analysis report by class and scene type.
- Candidate model checkpoint(s) for final testing.

Acceptance criteria:

- Validation metrics approaching thresholds with stable training curves.
- Top failure modes identified with mitigation actions.

## Week 4 (April 1 - April 7, 2026): Finalization and Submission Package

Goals:

- Lock reproducible pipeline.
- Prepare project write-up artifacts.
- Stress-test scale generation and full retraining.

Deliverables:

- Final end-to-end runbook.
- Executive summary (2 pages, non-technical language).
- FAQ (2-5 pages).
- Technical appendix (full reproducibility details).

Acceptance criteria:

- One-command or one-script reproducible pipeline documented.
- Submission package complete and internally reviewed.

## 4. Work Breakdown Structure (WBS)

1. Scene adaptation from kitchen photo
- Estimate scene layout primitives and navigable surfaces.
- Build reusable Blender scene template.

2. Synthetic rendering and labeling
- Randomized pest placement, appearance, and camera/light settings.
- Automatic 2D bbox export in chosen label format.

3. Video and behavior simulation
- Movement trajectories and behavior policy.
- Frame sequencing and optional track IDs.

4. Training and evaluation
- Baseline + ViT detector training.
- Metrics: TPR, FPR, per-class recall/precision.

5. Infrastructure and reproducibility
- Batch generation on cluster.
- Experiment tracking and artifact versioning.

6. Documentation and presentation
- Executive summary, FAQ, technical appendix.

## 5. Team Role Template

Use 3-4 people; each person owns one primary module and one backup module.

### Role A - Data/Graphics Lead

- Owns Blender scene pipeline and asset integration.
- Responsible for label correctness tooling.

### Role B - Simulation/Automation Lead

- Owns video generation, motion logic, and batch jobs.
- Responsible for throughput and failure recovery scripts.

### Role C - ML Lead

- Owns training, evaluation, and model selection.
- Responsible for meeting TPR/FPR targets.

### Role D - Reproducibility/Writeup Lead (optional if 4 people)

- Owns config hygiene, experiment tables, and final documents.
- Responsible for final appendix reproducibility quality.

## 6. Repository Structure

```text
Synthetic-Data-Generation-for-Pest-Detection/
  README.md
  PROJECT_EXECUTION_PLAN.md
  configs/
    data.yaml
    generation.yaml
    train.yaml
  data/
    raw/
      kitchen_photos/
      assets_3d/
    generated/
      images/
      videos/
      labels/
    splits/
  src/
    generation/
      blender_scene.py
      render_images.py
      render_videos.py
      export_labels.py
    simulation/
      pest_motion.py
      trajectory_sampling.py
    training/
      train_detector.py
      evaluate_detector.py
    utils/
      io.py
      viz.py
      metrics.py
  scripts/
    run_generate_images.sh
    run_generate_videos.sh
    run_train.sh
    run_eval.sh
    run_full_pipeline.sh
  notebooks/
    sanity_check_labels.ipynb
    error_analysis.ipynb
  reports/
    weekly/
    figures/
  outputs/
    logs/
    checkpoints/
```

## 7. Script Naming and CLI Convention

Naming:

- Python entrypoints: `verb_object.py` (e.g., `render_videos.py`).
- Shell wrappers: `run_<task>.sh` (e.g., `run_train.sh`).

CLI pattern:

```bash
python -m src.generation.render_videos --config configs/generation.yaml --seed 42 --out data/generated/videos
python -m src.training.train_detector --config configs/train.yaml --data data/splits --out outputs/checkpoints
python -m src.training.evaluate_detector --checkpoint outputs/checkpoints/best.pt --split test --out outputs/logs
```

## 8. Risk Register and Mitigations

1. Synthetic-to-real domain gap
- Mitigation: domain randomization + mixed backgrounds + photometric augmentations.

2. Annotation drift or bugs
- Mitigation: automated label overlay checks + random manual audits each run.

3. Compute bottleneck
- Mitigation: pre-render asset cache, parallel jobs, profiling and throughput benchmarks.

4. Model overfitting to synthetic artifacts
- Mitigation: diversify rendering, hard negative mining, class-balanced sampling.

## 9. Immediate Next Steps (Next 48 Hours)

1. Inspect v3 prediction overlays and failure cases to diagnose the current zero-recall behavior.
2. Generate `video_clip_smoke_v4` with more clips, more backgrounds, and stronger motion diversity.
3. Run threshold sweeps and per-class evaluation summaries on the video-frame model.
4. Add optional MP4 assembly and shard-friendly cluster configs after recall starts improving.
