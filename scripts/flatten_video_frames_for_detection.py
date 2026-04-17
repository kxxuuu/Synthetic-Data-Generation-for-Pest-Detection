#!/usr/bin/env python3
"""
Flatten generated video clip frames into a standard detection dataset layout.

Input layout:
  <video-dir>/clips/<clip_id>/frames/frame_000000.png
  <video-dir>/clips/<clip_id>/labels/frame_000000.txt

Output layout:
  <out-dir>/images/<clip_id>_frame_000000.png
  <out-dir>/labels/<clip_id>_frame_000000.txt
  <out-dir>/manifest.csv

Example:
  python scripts/flatten_video_frames_for_detection.py \
    --video-dir data/generated/video_clip_smoke_v1 \
    --out-dir data/generated/video_frame_smoke_v1 \
    --mode copy
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    out_dir = args.out_dir.resolve()
    clips_dir = video_dir / "clips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing clips directory: {clips_dir}")

    image_out = out_dir / "images"
    label_out = out_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []
    total = 0

    for clip_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        clip_id = clip_dir.name
        frames_dir = clip_dir / "frames"
        labels_dir = clip_dir / "labels"
        if not frames_dir.exists() or not labels_dir.exists():
            continue

        for frame_path in sorted(frames_dir.glob("*.png")):
            label_path = labels_dir / f"{frame_path.stem}.txt"
            if not label_path.exists():
                continue

            stem = f"{clip_id}_{frame_path.stem}"
            out_image = image_out / f"{stem}.png"
            out_label = label_out / f"{stem}.txt"
            link_or_copy(frame_path, out_image, args.mode)
            link_or_copy(label_path, out_label, args.mode)
            rows.append(
                [
                    clip_id,
                    frame_path.name,
                    stem,
                    str(frame_path.resolve()),
                    str(label_path.resolve()),
                    str(out_image),
                    str(out_label),
                ]
            )
            total += 1

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "frame_name",
                "flat_stem",
                "source_frame_path",
                "source_label_path",
                "flat_image_path",
                "flat_label_path",
            ]
        )
        writer.writerows(rows)

    print(f"flattened_frames={total}")
    print(f"wrote: {manifest_path}")
    print(f"images_dir: {image_out}")
    print(f"labels_dir: {label_out}")


if __name__ == "__main__":
    main()
