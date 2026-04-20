# Requirements Gap Analysis

This document maps the current repository state against the course project requirements for synthetic pest detection.

## Requirement Summary

The assignment asks for a pipeline that:

1. Takes a still kitchen image as input.
2. Generates labeled pest videos in the same kitchen layout.
3. Produces 30-60 second videos.
4. Labels each frame with pest class and bounding boxes.
5. Runs automatically with Python.
6. Scales to thousands of videos on the Duke compute cluster.
7. Trains a vision transformer to detect pest type and location.
8. Reaches:
   - true detection rate >= 80%
   - false positive rate < 5%
   on instructor-run test video data.

## Current Status At A Glance

Current repository status is best described as:

**Phase 1 baseline + Phase 2 smoke-scale video extension**

That means the project already demonstrates:

- automated synthetic image generation
- automated synthetic clip/frame generation
- per-frame bbox label export
- dataset splitting
- detector training and evaluation
- Windows-tested local execution

The strongest current local detector baseline is:

- Faster R-CNN with `positive_anchor` batching on `video_frame_smoke_v4`

But the repository still does **not** satisfy the full final assignment requirements.

## Requirement-By-Requirement Mapping

### 1. Input: a still kitchen image

**Status:** Partially met

What exists now:

- The generator uses kitchen-style background images.
- Backgrounds are sourced from Places365 kitchen and restaurant_kitchen categories.

Gap:

- The current system does not yet adapt to a single user-provided kitchen photo and preserve that exact layout.
- It uses sampled kitchen backgrounds rather than scene-specific reconstruction.

### 2. Generate labeled pest videos

**Status:** Partially met

What exists now:

- The repository generates synthetic clip folders with ordered frames.
- Per-frame labels are exported for each clip.
- Video-style clips have been generated at smoke scale.

Gap:

- The current workflow is still centered on frames and frame-derived datasets.
- Optional compiled video export is not yet the main validated path.
- Assignment-scale batch generation is not demonstrated yet.

### 3. Video length 30-60 seconds

**Status:** Partially met

What exists now:

- The repository has already generated 30-second smoke-scale clips.

Gap:

- 60-second clips are not yet the standard validated path.
- Large-batch 30-60 second generation has not been benchmarked or validated yet.

### 4. Each frame labeled with pest presence and bounding boxes

**Status:** Partially met

What exists now:

- Per-frame labels are exported automatically.
- Bounding boxes and class IDs are saved in YOLO-style text format.
- Frame-wise clip metadata and flattened detection datasets exist.

Gap:

- The labeling workflow is validated locally, but not yet at assignment-scale throughput.
- Final report-quality annotation QA at larger scale is still needed.

### 5. Automated generation in Python

**Status:** Met for the current local baseline

What exists now:

- Blender generation is scripted in Python.
- Dataset split, train, eval, and diagnostic scripts are Python-based.
- Both image and smoke-scale video/frame generation are automated.

Gap:

- The main remaining gap is operational scale, not basic automation.

### 6. Feasible generation at Duke cluster scale

**Status:** Partially met

What exists now:

- `job.sbatch` and `run.sh` exist as infrastructure entrypoints.
- The repo already separates generation, split, train, and eval stages.

Gap:

- Thousand-video production runs have not been demonstrated.
- Cluster throughput, storage planning, and large-batch job orchestration are still future work.

### 7. Train a vision transformer detector

**Status:** Partially met

What exists now:

- DETR training and evaluation are implemented.
- DETR has been trained on both static-image and video-derived smoke datasets.

Gap:

- The current strongest local result comes from Faster R-CNN, not the transformer-based detector.
- DETR currently suffers from score/ranking collapse on the held-out `video_frame_smoke_v4` split.
- So the repository has a transformer detector pipeline, but not a strong transformer-based final baseline yet.

### 8. Reach 80% TPR and <5% FPR on test video data

**Status:** Not yet met

What exists now:

- Local evaluation code exists.
- On the local held-out synthetic `video_frame_smoke_v4` split, the current Faster R-CNN baseline reaches:
  - recall `1.0`
  - frame-level false positive rate `0.0`

Gap:

- These are local smoke-scale synthetic held-out results, not instructor-run test video results.
- No validated claim can yet be made about final assignment compliance on the true external test set.

## What Has Been Completed

The following pieces are real, working progress and should be presented as such:

- Windows environment setup is working.
- Blender installation and local generation are working.
- Pest model integration and model-pool curation are working.
- Places365 kitchen background preparation is working.
- Synthetic image generation is working.
- Synthetic video-style clip generation is working at smoke scale.
- Per-frame label export is working.
- Flattening video frames into the current detection dataset format is working.
- Dataset splitting is working.
- DETR smoke training and diagnostics are working.
- Faster R-CNN training and evaluation are working.
- A stable local Faster R-CNN baseline now exists on held-out synthetic video-frame data.

## What Still Needs To Be Built

To satisfy the full assignment, the next major deliverables are:

1. Assignment-scale video generation
- more clips
- larger background coverage
- 30-60 second runs as the standard path
- cluster-oriented throughput validation

2. Kitchen-photo adaptation
- either real scene/layout adaptation
- or a clearly defended approximation baseline in the write-up

3. Final detector positioning
- decide how the transformer-based component will be presented
- either improve DETR materially
- or explain why the current strongest baseline is non-transformer while transformer work remains exploratory

4. Final evaluation workflow
- consistent TPR/FPR definitions
- instructor-test alignment
- reproducible reporting

## Recommended Positioning In Docs And Presentation

The safest and most honest way to describe the repository right now is:

**Current phase:** smoke-scale synthetic video pipeline with a strong local Faster R-CNN baseline and an exploratory DETR path

Suggested wording:

- The repository currently implements a working synthetic pest data pipeline from Blender generation through detector training.
- The project has progressed beyond static images to smoke-scale 30-second clip generation with per-frame labels.
- A local held-out synthetic evaluation pipeline is working.
- Faster R-CNN is the strongest current local baseline, while DETR remains implemented but not yet the strongest performer.
- Single-image kitchen adaptation, assignment-scale generation, and final external-metric validation remain future work.

## Suggested Next Milestones

### Milestone 1

Scale from smoke-scale 30-second clips to larger 30-60 second batch generation.

### Milestone 2

Add or justify the kitchen-layout adaptation story for the final write-up.

### Milestone 3

Decide whether DETR will be improved further or retained as an exploratory transformer baseline while Faster R-CNN remains the main detector.

### Milestone 4

Prepare final evaluation and report artifacts aligned with the external grading criteria.
