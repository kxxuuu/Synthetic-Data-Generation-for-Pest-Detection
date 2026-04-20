# Video Generation Plan

This document outlines the next development stage needed to move from the current static-image baseline to the full assignment target of synthetic pest videos with frame-level labels.

## Goal

Extend the current repository from:

- synthetic labeled images

to:

- synthetic labeled videos
- 30-60 seconds each
- frame-level pest labels and bounding boxes
- generation pipeline suitable for Duke cluster scaling

## Current Starting Point

The repository already has:

- Blender automation
- pest 3D models
- kitchen backgrounds
- class labels and bbox export
- dataset split utilities
- DETR training and evaluation scripts
- a working video MVP:
  - [generate_synthetic_video_blender.py](scripts/generate_synthetic_video_blender.py)
  - per-clip frames
  - per-frame labels
  - clip metadata
  - `manifest.csv`
- a flattening utility for reusing video frames in the current training pipeline:
  - [flatten_video_frames_for_detection.py](scripts/flatten_video_frames_for_detection.py)

That means the main missing work is no longer the first temporal prototype. The main missing work is now extending that prototype into a full assignment-scale video pipeline.

## Current Implemented MVP

The repository can now:

- generate clip sequences ranging from short smoke clips to 30-second smoke-scale runs
- generate clip IDs from an arbitrary start offset for shard-friendly runs
- generate frame counts from `duration_seconds * fps`
- export frame images
- export YOLO-format labels for each frame
- save clip metadata
- assemble rendered frame folders into MP4 demo clips
- flatten video frames into `images/` + `labels/` for DETR reuse
- generate negative clips with no pest present
- generate negative frames inside otherwise positive clips
- train DETR on flattened video-frame smoke datasets, including negative samples

What still remains:

- 60-second clips and assignment-scale 30-60 second batches
- larger numbers of clips
- more realistic motion
- optional multi-pest scenes
- large-scale video packaging
- scale-out execution

## Current Verified Checkpoint

As of April 20, 2026, the repository has already validated:

- [MODEL_AUDIT.md](assets/models/MODEL_AUDIT.md)
  - raw downloads were filtered into curated per-class pools
  - the generator now supports model directories plus exclude lists
- [video_clip_smoke_v4](data/generated/video_clip_smoke_v4/)
  - 9 clips
  - 30 seconds each at 4 FPS
  - 1080 rendered frames total
  - all three pest classes represented
  - 2 fully negative clips plus negative frames inside positive clips
  - curated rat, mouse, and cockroach model pools used
- [video_frame_smoke_v4](data/generated/video_frame_smoke_v4/)
  - 1080 flattened frame-label pairs for detector training reuse
- [detr_video_frame_smoke_v4](outputs/detr_video_frame_smoke_v4/)
  - DETR smoke training completed
  - best validation loss: `1.075053`
- [eval_test.json](outputs/detr_video_frame_smoke_v4/eval_test.json)
  - held-out test recall at confidence `0.5`: `0.0`
  - frame FPR on negative test frames: `0.0`
- [eval_test_thr020.json](outputs/detr_video_frame_smoke_v4/eval_test_thr020.json)
  - lower-threshold diagnostic still produced `0.0` recall
- [diagnostics summary](outputs/detr_video_frame_smoke_v4/diagnostics/summary.json)
  - positive and negative frame top scores both cluster around `0.025`
  - thresholds `>= 0.05` suppress every prediction
  - threshold `0.01` recovers roughly `46-50%` recall, but precision collapses and negative-frame FPR becomes `1.0`
- [diagnostics_topk summary](outputs/detr_video_frame_smoke_v4/diagnostics_topk/summary.json)
  - `top-1` on test still yields `0.0` recall
  - `top-3` recovers only about `12%` recall
  - even capped predictions still produce `frame_fpr = 1.0` on negative test frames
- [fasterrcnn stable eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_stable/eval_test.json)
  - a simpler torchvision detector reaches `recall = 1.0`, `precision = 1.0`, `frame_fpr = 0.0` on the same v4 test split
  - this suggests the current video data pipeline is at least sufficient for one strong baseline detector
- [fasterrcnn stable diagnostics summary](outputs/fasterrcnn_video_frame_smoke_v4_stable/diagnostics/summary.json)
  - positive-frame top scores are well separated from negative-frame top scores
  - negative test frames produce top scores of `0.0` throughout the split
  - `top-1` predictions remain perfect at threshold `0.2`
- [video_frame_smoke_v4_audit.json](outputs/video_frame_smoke_v4_audit.json)
  - train/val/test are all roughly `55%` negative frames
  - with batch size `2`, about `30%` of mini-batches are expected to be all-negative
  - this likely explains most skipped non-finite Faster R-CNN batches without pointing to a label-format bug
- [fasterrcnn balanced eval_test.json](outputs/fasterrcnn_video_frame_smoke_v4_balanced/eval_test.json)
  - `positive_anchor` batching removes skipped train/val batches entirely
  - held-out recall remains `1.0` at threshold `0.5`
  - frame-level FPR remains `0.0`
- [video_demo_v1](data/generated/video_demo_v1/)
  - 3 demo clips
  - 30 seconds each at 4 FPS
  - MP4s assembled for direct qualitative review
- [video_clip_smoke_v5](data/generated/video_clip_smoke_v5/)
  - 12 clips
  - 30 seconds each at 4 FPS
  - 1440 rendered frames total
  - all three pest classes represented
  - negative clips and negative frames still included
- [video_frame_smoke_v5](data/generated/video_frame_smoke_v5/)
  - 1440 flattened frame-label pairs for detector training reuse
- [video_frame_smoke_v5_audit.json](outputs/video_frame_smoke_v5_audit.json)
  - train/val/test remain about `54%` negative frames
  - batch size `2` still implies about `29%` all-negative mini-batches under IID sampling
- [fasterrcnn balanced v5 eval_test.json](outputs/fasterrcnn_video_frame_smoke_v5_balanced/eval_test.json)
  - held-out test recall `1.0`
  - precision `1.0`
  - frame FPR `0.0`
- [fasterrcnn balanced v5 diagnostics summary](outputs/fasterrcnn_video_frame_smoke_v5_balanced/diagnostics/summary.json)
  - positive-frame top scores stay high
  - negative-frame top scores remain `0.0`
  - thresholds `0.5` and `0.2` are both clean
- report-ready figures now exist:
  - [report_figures_v4_comparison](outputs/report_figures_v4_comparison/)
  - [report_figures_v5_final](outputs/report_figures_v5_final/)

This means the pipeline gap has shifted again. The blocker is no longer obviously broken raw assets or obviously broken generation. The current local baseline is now Faster R-CNN with `positive_anchor` batching on `video_frame_smoke_v5`, and the next blocker is "how fast can the project scale the video pipeline around the detector family that already works, while keeping negative-batch handling explicit and stable?"

## Recommended Development Strategy

Do this in stages instead of trying to jump directly to full videos.

## Phase 2A: Frame Sequence Generation

### Objective

Generate a sequence of frames from a single scene with one or more moving pests.

### What to add

Create a new Blender script, for example:

- `scripts/generate_synthetic_video_blender.py`

Core additions:

- fixed background per clip
- fixed scene per clip
- pest trajectory across time
- per-frame rendering
- per-frame bbox export
- clip-level metadata

### Output structure

Recommended layout:

```text
data/generated/video_v1/
  clips/
    clip_000001/
      frames/
        frame_000000.png
        frame_000001.png
        ...
      labels/
        frame_000000.txt
        frame_000001.txt
        ...
      clip_metadata.json
  manifest.csv
```

## Phase 2B: Motion Modeling

### Objective

Make pest movement look like video instead of independent random placements.

### Minimum viable motion

For each clip:

- sample initial pest position
- sample velocity vector
- apply bounded random walk over time
- keep pests constrained to the floor plane
- randomize heading gradually, not independently every frame

### Recommended motion features

- speed ranges by class
  - cockroach: fast, jittery
  - mouse: medium, darting
  - rat: slower, heavier turns
- occasional pauses
- boundary reflection or path redirection
- optional simple occlusion by foreground props later

## Phase 2C: Frame-Level Labeling

### Objective

Export labels for every frame in a format the training code can consume.

### Recommended format

Keep the current YOLO-style normalized bbox text format for short-term compatibility:

`class x_center y_center width height`

This minimizes retraining code changes.

### Recommended metadata per clip

Store:

- background path
- model path
- class name
- frame count
- FPS
- clip duration
- seed
- trajectory parameters

## Phase 2D: Video Assembly

### Objective

Produce actual video files in addition to frame folders.

### Recommendation

Treat frames as the authoritative data source and compile videos as a convenience artifact.

Possible outputs:

- `clip.mp4`
- `frames/`
- `labels/`

This is safer because:

- training often consumes frames anyway
- debugging labels is easier frame-by-frame
- rendering failures are easier to recover

## Phase 2E: Cluster-Scale Execution

### Objective

Make generation parallel and scalable.

### Recommended approach

Parameterize generation by clip range:

- `--start-clip`
- `--num-clips`
- `--frames-per-clip`
- `--seed`

Then schedule many jobs, each generating a disjoint clip shard.

### Example shard design

- job 1 generates clips `0-99`
- job 2 generates clips `100-199`
- etc.

### Why this works

- avoids write collisions
- easy to retry failed shards
- easy to combine metadata afterward

## Phase 2F: Training Transition

### Objective

Use video-generated frames for model training and held-out evaluation.

### Near-term training option

Flatten labeled video frames into the same image/label structure already used by `train_detr.py`.

Example:

```text
data/generated/video_frames_v1/
  images/
  labels/
```

This lets you reuse the current DETR training code immediately.

### Longer-term option

If needed later:

- add clip-aware evaluation
- add temporal consistency metrics
- test frame subsampling strategies

## Kitchen Photo Adaptation Options

The requirement says the video should match the same layout as an input kitchen image. There are two realistic paths:

### Option A: True adaptation

Implement some scene adaptation from a single photo:

- estimate floor plane
- estimate camera pose approximately
- anchor pest movement to plausible floor regions

This is harder, but closest to the assignment wording.

### Option B: Defended baseline approximation

Use a selected kitchen background image as the scene and argue that:

- the key scientific question is synthetic pest detection feasibility
- the baseline first solves rendering, labeling, and training
- full layout adaptation is future work

This is weaker than true adaptation, but much easier to finish.

## Suggested File Additions

Recommended next files:

- `scripts/generate_synthetic_video_blender.py`
- `scripts/render_video_frames.sh`
- `scripts/assemble_video.py`
- `scripts/flatten_video_frames_for_detection.py`
- `scripts/check_video_annotations.py`

## Suggested Immediate Implementation Order

1. Duplicate the image-generation script into a video-generation script.
2. Add per-frame loop with trajectory updates.
3. Export frames and per-frame labels.
4. Save clip metadata.
5. Flatten generated frames into the current training format.
6. Reuse `split_detection_dataset.py`.
7. Retrain DETR on video-derived frames.
8. Add optional MP4 assembly.

## Success Criteria for the Next Stage

The next stage should be considered successful when the repository can:

1. generate multiple 30-60 second clips automatically
2. export per-frame labels for the full clips
3. repeat this across multiple clips without manual Blender interaction
4. flatten the generated frames into a trainable detection dataset
5. document how to scale the process across multiple jobs
6. achieve non-trivial held-out recall before major packaging work

## Updated Immediate Next Steps

1. Diagnose the v3 zero-recall result with qualitative prediction overlays and class-wise checks.
2. Generate a larger `video_clip_smoke_v4` dataset with more clips, more backgrounds, and more motion diversity.
3. Add threshold sweeps and per-class summaries to the evaluation loop.
4. Add optional MP4 assembly only after the detection metrics start improving.

## Practical Recommendation

The highest-value next step is:

**improve held-out recall on video-derived frames while continuing to scale generation**

That gives you:

- evidence that the synthetic video pipeline is actually helping the detector
- clearer guidance on whether motion, backgrounds, labels, or class balance need work
- a stronger foundation before spending time on packaging and polish

Actual MP4 packaging can be added after the frame pipeline is stable.
