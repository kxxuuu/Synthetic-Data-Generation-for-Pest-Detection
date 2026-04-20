# Report Briefing

This document is the fastest handoff for teammates who are writing the report rather than running code.

## What The Project Currently Is

The repository currently represents:

- a working synthetic pest image pipeline
- a working smoke-scale synthetic video-frame pipeline
- a working local training/evaluation pipeline
- a strong local Faster R-CNN baseline on held-out synthetic video-frame data

The repository does **not** yet represent the full final assignment submission.

The safest current positioning is:

**Phase 1 baseline + Phase 2 smoke-scale video extension**

That means:

- image generation works
- video-style clip generation works
- per-frame labels work
- flattened video-frame training works
- a local detector baseline works well on held-out synthetic smoke data

But:

- single-photo kitchen layout adaptation is not done
- assignment-scale 30-60 second batch generation is not done yet beyond smoke-scale runs
- final claims on instructor-run test videos are not supported yet

## Current Headline Results

### Synthetic video pipeline

Current best smoke-scale video dataset:

- [video_clip_smoke_v4](data/generated/video_clip_smoke_v4/)
- [video_frame_smoke_v4](data/generated/video_frame_smoke_v4/)
- [split.json](data/splits/video_frame_smoke_v4/split.json)

This dataset includes:

- 9 clips
- 30 seconds each
- 1080 rendered frames total
- all three pest classes
- negative clips and negative frames

### DETR status

DETR training works technically, but the held-out v4 result is poor.

Key points:

- [eval_test.json](outputs/detr_video_frame_smoke_v4/eval_test.json)
- [diagnostics summary](outputs/detr_video_frame_smoke_v4/diagnostics/summary.json)
- [diagnostics_topk summary](outputs/detr_video_frame_smoke_v4/diagnostics_topk/summary.json)

Safe summary:

- DETR is trainable in this repository
- but on the current v4 smoke-scale video-frame split it shows confidence and ranking collapse
- so DETR is not the current recommended detector baseline

### Faster R-CNN status

The strongest current local detector baseline is Faster R-CNN.

Recommended baseline:

- [fasterrcnn_video_frame_smoke_v4_balanced](outputs/fasterrcnn_video_frame_smoke_v4_balanced/)

Why this one is recommended:

- uses `positive_anchor` batching
- eliminates skipped train/val batches
- keeps held-out synthetic test recall at `1.0`
- keeps frame-level false positive rate at `0.0`

Key files:

- [summary.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/summary.json)
- [train.log](outputs/fasterrcnn_video_frame_smoke_v4_balanced/train.log)
- [eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json)
- [eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test_thr020.json)

Safe metric wording:

- On the local held-out synthetic `video_frame_smoke_v4` test split, the current Faster R-CNN baseline achieved:
  - recall `1.0`
  - precision `0.9865` at confidence threshold `0.5`
  - frame-level false positive rate `0.0`

Important caveat:

- these are **local synthetic smoke-scale held-out results**
- they are **not** final results on instructor-run test videos

## Safe Claims For The Report

These are safe to say:

- The project already supports automated synthetic pest data generation in Blender.
- The pipeline exports per-frame bounding-box labels.
- The repository now supports smoke-scale 30-second clip generation.
- The repository can flatten generated video frames into a standard object-detection dataset.
- A local detector baseline has been successfully trained and evaluated on held-out synthetic video-frame data.
- DETR was explored as a transformer-based detector, but the strongest current baseline is Faster R-CNN.

These should be phrased carefully:

- "video generation"
  - clarify that current validation is smoke-scale and frame-centric
- "meets assignment metrics"
  - only say this on the current local synthetic smoke split if needed
  - do **not** claim final assignment compliance
- "single kitchen image adaptation"
  - say this remains future work or partial work

These should **not** be claimed:

- full assignment completion
- validated performance on instructor-run test videos
- validated Duke-cluster-scale generation throughput
- exact layout adaptation from one user-provided kitchen image

## Recommended Report Storyline

A strong current narrative is:

1. Problem
- Pest detection in commercial kitchens needs labeled data, but real labeled data is scarce.

2. Approach
- Use Blender plus pest 3D assets and kitchen backgrounds to generate labeled synthetic training data.

3. Pipeline
- Background selection
- 3D pest insertion
- frame rendering
- per-frame bbox export
- flattening into a detector-training dataset

4. Modeling
- DETR was implemented and diagnosed.
- Faster R-CNN became the strongest working local baseline on smoke-scale synthetic video-frame data.

5. Current results
- smoke-scale synthetic video pipeline works end to end
- local held-out synthetic evaluation is strong with Faster R-CNN

6. Remaining gaps
- assignment-scale generation
- single-photo layout adaptation
- final evaluation on instructor-run test videos

## Best Docs For A Report Teammate

Recommended reading order:

1. [REPORT_BRIEFING.md](REPORT_BRIEFING.md)
2. [README.md](README.md)
3. [PROJECT_STATUS.md](PROJECT_STATUS.md)
4. [REQUIREMENTS_GAP_ANALYSIS.md](REQUIREMENTS_GAP_ANALYSIS.md)
5. [PROJECT_EXECUTION_PLAN.md](PROJECT_EXECUTION_PLAN.md)
6. [VIDEO_GENERATION_PLAN.md](VIDEO_GENERATION_PLAN.md)
7. [submissions/README.md](submissions/README.md)

Usually not necessary for a report-only teammate unless they need troubleshooting detail:

- [WINDOWS_RUN_TROUBLESHOOTING.md](WINDOWS_RUN_TROUBLESHOOTING.md)

## Useful Figures And Tables

Good figure/table sources:

- Pipeline diagrams:
  - [README.md](README.md)
- DETR diagnostic overlays:
  - [detr overlays](outputs/detr_video_frame_smoke_v4/diagnostics/test/overlays/)
- Faster R-CNN diagnostic overlays:
  - [fasterrcnn overlays](outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/test/overlays/)
- Evaluation tables:
  - [detr eval_test.json](outputs/detr_video_frame_smoke_v4/eval_test.json)
  - [fasterrcnn balanced eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json)
  - [fasterrcnn balanced eval_test_thr020.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test_thr020.json)

## Questions To Ask The Coding Team

If the report teammate needs clarifications, the most useful questions are:

- Which detector should be presented as the main baseline right now?
- Which metrics are safe to report as current local results?
- Which limitations should be made explicit in the write-up?
- Which future-work items are already planned versus still open?
