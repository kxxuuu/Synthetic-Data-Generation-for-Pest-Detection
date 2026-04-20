#!/usr/bin/env python3
"""
Audit a detection dataset split for class balance, empty-frame rate, and bbox size.

Example:
  python scripts/audit_detection_dataset.py \
    --data-dir data/generated/video_frame_smoke_v4 \
    --split-json data/splits/video_frame_smoke_v4/split.json \
    --batch-size 2 \
    --output-json outputs/video_frame_smoke_v4_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ID2LABEL = {0: "rat", 1: "mouse", 2: "cockroach"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    vals = sorted(values)

    def pick(q: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
        return float(vals[idx])

    return {
        "min": float(vals[0]),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "max": float(vals[-1]),
    }


def parse_label_file(label_path: Path) -> list[dict]:
    txt = label_path.read_text().strip()
    if not txt:
        return []
    rows = []
    for line in txt.splitlines():
        cls, xc, yc, bw, bh = line.split()
        rows.append(
            {
                "label": int(cls),
                "xc": float(xc),
                "yc": float(yc),
                "bw": float(bw),
                "bh": float(bh),
                "area": float(bw) * float(bh),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    split_payload = json.loads(args.split_json.read_text())
    label_dir = args.data_dir / "labels"

    report = {
        "data_dir": str(args.data_dir.resolve()),
        "split_json": str(args.split_json.resolve()),
        "batch_size": args.batch_size,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        if split_name not in split_payload:
            continue
        stems = split_payload[split_name]
        class_counts = {name: 0 for name in ID2LABEL.values()}
        empty_images = 0
        box_areas: list[float] = []
        box_widths: list[float] = []
        box_heights: list[float] = []
        negative_flags: list[int] = []

        for stem in stems:
            rows = parse_label_file(label_dir / f"{stem}.txt")
            if not rows:
                empty_images += 1
                negative_flags.append(1)
                continue

            negative_flags.append(0)
            for row in rows:
                class_counts[ID2LABEL[row["label"]]] += 1
                box_areas.append(row["area"])
                box_widths.append(row["bw"])
                box_heights.append(row["bh"])

        num_images = len(stems)
        positive_images = num_images - empty_images
        negative_rate = empty_images / num_images if num_images else 0.0

        consecutive_all_negative_batches = 0
        total_batches = 0
        for start in range(0, len(negative_flags), args.batch_size):
            batch_flags = negative_flags[start : start + args.batch_size]
            if not batch_flags:
                continue
            total_batches += 1
            if all(flag == 1 for flag in batch_flags):
                consecutive_all_negative_batches += 1

        split_report = {
            "num_images": num_images,
            "num_positive_images": positive_images,
            "num_negative_images": empty_images,
            "negative_image_rate": negative_rate,
            "estimated_all_negative_batch_rate_iid": negative_rate**args.batch_size if num_images else 0.0,
            "consecutive_all_negative_batch_rate_no_shuffle": (
                consecutive_all_negative_batches / total_batches if total_batches > 0 else 0.0
            ),
            "class_box_counts": class_counts,
            "total_boxes": sum(class_counts.values()),
            "box_area_quantiles": quantiles(box_areas),
            "box_width_quantiles": quantiles(box_widths),
            "box_height_quantiles": quantiles(box_heights),
        }
        report["splits"][split_name] = split_report

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
