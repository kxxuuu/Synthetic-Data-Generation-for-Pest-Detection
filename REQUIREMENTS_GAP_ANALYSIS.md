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

## Current Status at a Glance

Current repository status is best described as:

**Phase 1: static-image synthetic data baseline with DETR training**

That means the project already demonstrates:

- automated synthetic data generation
- automated bbox label generation
- dataset splitting
- detector training
- Windows-tested local execution

But it does **not** yet satisfy the full final assignment requirements.

## Requirement-by-Requirement Mapping

### 1. Input: a still kitchen image

**Status:** Partially met

What exists now:

- The generator uses kitchen-style background images.
- Backgrounds are sourced from Places365 kitchen and restaurant_kitchen categories.

Gap:

- The current system does not yet adapt to a single user-provided kitchen photo and preserve that exact layout.
- It uses sampled kitchen backgrounds rather than scene-specific reconstruction.

### 2. Generate labeled pest videos

**Status:** Not yet met

What exists now:

- The repository generates labeled static images.

Gap:

- No 30-60 second video generator is implemented yet.
- No frame-sequence rendering pipeline exists yet.
- No exported video files are currently produced.

### 3. Video length 30-60 seconds

**Status:** Not met

What exists now:

- Image-level generation only.

Gap:

- No video duration control exists yet.

### 4. Each frame labeled with pest presence and bounding boxes

**Status:** Partially met

What exists now:

- Per-image labels are generated automatically.
- Bounding boxes and class IDs are exported.

Gap:

- There is no per-frame video annotation export yet.
- No frame indexing or video annotation manifest exists yet.

### 5. Automated generation in Python

**Status:** Met for the current static-image baseline

What exists now:

- Blender generation is scripted in Python.
- Dataset split, train, and evaluation scripts are Python-based.

Gap:

- Video generation automation still needs to be added.

### 6. Feasible generation at Duke cluster scale

**Status:** Partially met

What exists now:

- `job.sbatch` and `run.sh` exist as infrastructure entrypoints.
- The repo already distinguishes lightweight and heavy stages.

Gap:

- Large-scale video generation throughput has not been implemented or benchmarked.
- There is no demonstrated thousand-video production run yet.

### 7. Train a vision transformer detector

**Status:** Met at baseline level

What exists now:

- DETR training works locally.
- A smoke dataset was generated and used to train a model successfully.

Gap:

- The current run is a smoke test, not a meaningful final training result.
- More data and a proper validation/test protocol are still needed.

### 8. Reach 80% TPR and <5% FPR on test video data

**Status:** Not yet met

What exists now:

- Local training is functional.
- Evaluation code exists.

Gap:

- No final evaluation on test video data has been run.
- No evidence yet shows the pipeline achieves the target metrics.
- Current dataset is too small and image-based to support the final claim.

## What Has Been Completed

The following pieces are real, working progress and should be presented as such:

- Windows environment setup is working.
- Blender installation and local generation are working.
- Pest model integration is working.
- Places365 kitchen background preparation is working.
- Synthetic image generation is working.
- Label export is working.
- Dataset splitting is working.
- DETR smoke training is working.
- Short video-style clip generation is working.
- Per-frame label export is working.
- Flattening video frames into the current detection dataset format is working.
- DETR smoke training on video-derived frames is working.

## What Still Needs to Be Built

To satisfy the full assignment, the next major deliverables are:

1. A video generation pipeline
- multi-frame rendering
- pest trajectories
- per-frame bbox export
- video assembly

2. Kitchen-photo adaptation
- either real scene/layout adaptation
- or a clearly defended approximation baseline in the write-up

3. Cluster-scale generation workflow
- job parameterization
- batch generation
- throughput estimates
- storage/checkpoint planning

4. Final evaluation workflow
- consistent TPR/FPR definitions
- negative-frame handling
- test-video evaluation
- reproducible reporting

## Recommended Positioning in Docs and Presentation

The safest and most honest way to describe the repository right now is:

**Current phase:** static-image synthetic baseline for pest detection

Suggested wording:

- The repository currently implements a Phase 1 baseline that generates labeled synthetic pest images and trains a DETR detector.
- This establishes the core automation, rendering, labeling, and training pipeline.
- Video generation, scene adaptation from a single kitchen image, and final metric-target validation remain future work for full assignment compliance.

## Suggested Next Milestones

### Milestone 1

Convert static generation into frame-sequence generation with persistent object identity and per-frame labels.

### Milestone 2

Render 30-60 second videos and export frame annotations plus optional compiled video files.

### Milestone 3

Train on larger synthetic video-derived frame datasets and evaluate on held-out test videos.

### Milestone 4

Add or justify kitchen-layout adaptation from a single input image.
