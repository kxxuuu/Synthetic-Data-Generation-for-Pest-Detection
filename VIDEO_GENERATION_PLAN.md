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

That means the main missing work is not the entire pipeline, but specifically the **temporal/video generation layer**.

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

1. generate one 30-60 second clip automatically
2. export per-frame labels for the full clip
3. repeat this across multiple clips without manual Blender interaction
4. flatten the generated frames into a trainable detection dataset
5. document how to scale the process across multiple jobs

## Practical Recommendation

The highest-value next step is:

**implement frame-sequence generation first, not polished final videos**

That gives you:

- the temporal component the assignment requires
- per-frame labels
- faster progress toward training and evaluation

Actual MP4 packaging can be added after the frame pipeline is stable.
