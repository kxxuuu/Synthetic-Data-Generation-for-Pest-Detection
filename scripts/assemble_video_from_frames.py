#!/usr/bin/env python3
"""
Assemble an mp4 video from a directory of PNG frames.

Example:
  python scripts/assemble_video_from_frames.py \
    --frames-dir data/generated/video_demo_v1/clips/clip_000000/frames \
    --output-video data/generated/video_demo_v1/clips/clip_000000/demo.mp4 \
    --fps 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--output-video", type=Path, required=True)
    p.add_argument("--fps", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output_video), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output_video}")

    written = 0
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(f"Frame size mismatch: {frame_path}")
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"frames={written}")
    print(f"wrote: {args.output_video}")


if __name__ == "__main__":
    main()
