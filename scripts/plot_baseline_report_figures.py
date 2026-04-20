#!/usr/bin/env python3
"""
Generate simple report/demo figures from eval, diagnostics, and audit JSONs.

This version intentionally uses Pillow instead of matplotlib because the
Windows environment used for this project can crash on headless matplotlib
imports. The output PNGs are still suitable for README/report embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 900
HEIGHT = 560
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 70
MARGIN_BOTTOM = 80
BG = "white"
AXIS = "#222222"
GRID = "#d9d9d9"
TEXT = "#111111"
COLORS = ["#4e79a7", "#f28e2b", "#59a14f"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detr-eval", type=Path, default=None)
    p.add_argument("--baseline-eval", type=Path, required=True)
    p.add_argument("--baseline-diagnostics", type=Path, required=True)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def create_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((MARGIN_LEFT, 20), title, fill=TEXT, font=font)
    return image, draw, font


def plot_bounds() -> tuple[int, int, int, int]:
    return (
        MARGIN_LEFT,
        MARGIN_TOP,
        WIDTH - MARGIN_RIGHT,
        HEIGHT - MARGIN_BOTTOM,
    )


def draw_axes(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    y_max: float,
    y_label: str,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = plot_bounds()
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
    for i in range(6):
        value = y_max * (i / 5)
        y = bottom - int((bottom - top) * (i / 5))
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((20, y - 6), f"{value:.2f}" if y_max <= 1.5 else f"{int(value)}", fill=TEXT, font=font)
    draw.text((20, 20), y_label, fill=TEXT, font=font)
    return left, top, right, bottom


def legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, items: list[tuple[str, str]]) -> None:
    x = WIDTH - 210
    y = 22
    for label, color in items:
        draw.rectangle((x, y, x + 14, y + 14), fill=color, outline=color)
        draw.text((x + 20, y), label, fill=TEXT, font=font)
        y += 20


def value_to_y(value: float, top: int, bottom: int, y_max: float) -> int:
    usable = bottom - top
    return bottom - int((value / y_max) * usable)


def save_detector_comparison(detr_eval: dict | None, baseline_eval: dict, output_path: Path) -> None:
    labels = []
    recalls = []
    precisions = []
    frame_fprs = []

    if detr_eval is not None:
        labels.append("DETR v4")
        recalls.append(float(detr_eval["tpr_recall"]))
        precisions.append(float(detr_eval["precision"]))
        frame_fprs.append(float(detr_eval["frame_fpr"] or 0.0))

    labels.append("Faster R-CNN")
    recalls.append(float(baseline_eval["tpr_recall"]))
    precisions.append(float(baseline_eval["precision"]))
    frame_fprs.append(float(baseline_eval["frame_fpr"] or 0.0))

    image, draw, font = create_canvas("Detector Comparison On Held-Out Synthetic Test Data")
    left, top, right, bottom = draw_axes(draw, font, 1.05, "Metric Value")
    legend(draw, font, [("Recall", COLORS[0]), ("Precision", COLORS[1]), ("Frame FPR", COLORS[2])])

    num_groups = len(labels)
    group_width = (right - left) / max(num_groups, 1)
    bar_width = max(int(group_width / 6), 18)
    offsets = [-bar_width, 0, bar_width]

    for idx, label in enumerate(labels):
        center_x = int(left + group_width * (idx + 0.5))
        values = [recalls[idx], precisions[idx], frame_fprs[idx]]
        for metric_idx, value in enumerate(values):
            x0 = center_x + offsets[metric_idx] - bar_width // 2
            x1 = x0 + bar_width
            y = value_to_y(value, top, bottom, 1.05)
            draw.rectangle((x0, y, x1, bottom), fill=COLORS[metric_idx], outline=COLORS[metric_idx])
        draw.text((center_x - 24, bottom + 12), label, fill=TEXT, font=font)

    image.save(output_path)


def save_threshold_curve(diagnostics: dict, output_path: Path) -> None:
    split = diagnostics["splits"]["test"]
    threshold_metrics = split["threshold_metrics"]
    thresholds = sorted((float(k) for k in threshold_metrics.keys()))
    recalls = [float(threshold_metrics[str(t)]["recall"]) for t in thresholds]
    precisions = [float(threshold_metrics[str(t)]["precision"]) for t in thresholds]
    frame_fprs = [float(threshold_metrics[str(t)]["frame_fpr"] or 0.0) for t in thresholds]

    image, draw, font = create_canvas("Faster R-CNN Threshold Sensitivity")
    left, top, right, bottom = draw_axes(draw, font, 1.05, "Metric Value")
    legend(draw, font, [("Recall", COLORS[0]), ("Precision", COLORS[1]), ("Frame FPR", COLORS[2])])

    x_span = max(thresholds[-1] - thresholds[0], 1e-6)

    def to_x(t: float) -> int:
        return left + int(((t - thresholds[0]) / x_span) * (right - left))

    def draw_series(values: list[float], color: str) -> None:
        points = []
        for t, value in zip(thresholds, values):
            points.append((to_x(t), value_to_y(value, top, bottom, 1.05)))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)

    draw_series(recalls, COLORS[0])
    draw_series(precisions, COLORS[1])
    draw_series(frame_fprs, COLORS[2])

    for t in thresholds:
        x = to_x(t)
        draw.text((x - 10, bottom + 12), f"{t:.2f}", fill=TEXT, font=font)
    draw.text((WIDTH // 2 - 60, HEIGHT - 30), "Confidence Threshold", fill=TEXT, font=font)

    image.save(output_path)


def panel_title(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, x: int, y: int, title: str) -> None:
    draw.text((x, y), title, fill=TEXT, font=font)


def save_dataset_composition(audit_payload: dict, output_path: Path) -> None:
    image = Image.new("RGB", (1000, 480), BG)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 20), "Dataset Composition Summary", fill=TEXT, font=font)

    splits = ["train", "val", "test"]
    neg_rates = [float(audit_payload["splits"][s]["negative_image_rate"]) for s in splits]
    total_boxes = [int(audit_payload["splits"][s]["total_boxes"]) for s in splits]

    def draw_bar_panel(x0: int, y0: int, width: int, height: int, values: list[float], labels: list[str], y_max: float, title: str) -> None:
        left = x0 + 40
        top = y0 + 30
        right = x0 + width - 20
        bottom = y0 + height - 40
        panel_title(draw, font, x0 + 40, y0, title)
        draw.line((left, top, left, bottom), fill=AXIS, width=2)
        draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
        for i in range(6):
            value = y_max * (i / 5)
            y = bottom - int((bottom - top) * (i / 5))
            draw.line((left, y, right, y), fill=GRID, width=1)
            text = f"{value:.2f}" if y_max <= 1.5 else f"{int(value)}"
            draw.text((x0, y - 6), text, fill=TEXT, font=font)
        bar_area = right - left
        group_width = bar_area / len(values)
        bar_width = max(int(group_width / 2), 22)
        for idx, value in enumerate(values):
            cx = int(left + group_width * (idx + 0.5))
            x1 = cx - bar_width // 2
            x2 = x1 + bar_width
            y = bottom - int((value / y_max) * (bottom - top))
            draw.rectangle((x1, y, x2, bottom), fill=COLORS[idx], outline=COLORS[idx])
            draw.text((cx - 12, bottom + 10), labels[idx], fill=TEXT, font=font)

    draw_bar_panel(20, 60, 460, 360, neg_rates, splits, 1.0, "Negative Frame Rate By Split")
    draw_bar_panel(510, 60, 460, 360, total_boxes, splits, max(total_boxes) * 1.1, "Annotated Boxes By Split")
    image.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detr_eval = read_json(args.detr_eval) if args.detr_eval else None
    baseline_eval = read_json(args.baseline_eval)
    diagnostics = read_json(args.baseline_diagnostics)
    audit_payload = read_json(args.audit_json)

    save_detector_comparison(detr_eval, baseline_eval, args.output_dir / "detector_comparison.png")
    save_threshold_curve(diagnostics, args.output_dir / "threshold_sensitivity.png")
    save_dataset_composition(audit_payload, args.output_dir / "dataset_composition.png")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
