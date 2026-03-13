#!/usr/bin/env python3
"""
Evaluate DETR checkpoint on test split and report TPR / FPR-style metrics.

Example:
  python scripts/eval_detr.py \
    --data-dir data/generated/synth_v1_smoke4 \
    --split-json data/splits/synth_v1_smoke4/split.json \
    --checkpoint-dir outputs/detr_smoke4/best \
    --output-json outputs/detr_smoke4/eval_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, DetrForObjectDetection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_gt(label_path: Path, w: int, h: int) -> list[dict]:
    txt = label_path.read_text().strip()
    if not txt:
        return []
    out = []
    for line in txt.splitlines():
        cls, xc, yc, bw, bh = line.split()
        cls = int(cls)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        box_w = bw * w
        box_h = bh * h
        x1 = (xc * w) - box_w / 2.0
        y1 = (yc * h) - box_h / 2.0
        x2 = x1 + box_w
        y2 = y1 + box_h
        out.append({"label": cls, "box": [x1, y1, x2, y2]})
    return out


def iou_xyxy(a: list[float], b: list[float]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[int, int, int]:
    used = set()
    tp = 0
    fp = 0

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    for p in preds_sorted:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if j in used or p["label"] != g["label"]:
                continue
            v = iou_xyxy(p["box"], g["box"])
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            tp += 1
        else:
            fp += 1

    fn = len(gts) - len(used)
    return tp, fp, fn


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt_dir = args.checkpoint_dir.resolve()
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. "
            "Run training first (scripts/train_detr.py) and use the generated 'best' directory."
        )
    if not (ckpt_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Missing config.json in checkpoint directory: {ckpt_dir}. "
            "Expected a HuggingFace-saved DETR checkpoint."
        )

    split = json.loads(args.split_json.read_text())
    stems = split["test"]
    image_dir = args.data_dir / "images"
    label_dir = args.data_dir / "labels"

    processor = AutoImageProcessor.from_pretrained(str(ckpt_dir), local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(
        str(ckpt_dir), local_files_only=True
    ).to(args.device)
    model.eval()

    tp = fp = fn = 0
    negative_frames = 0
    fp_on_negative = 0

    for stem in tqdm(stems, desc="eval", leave=False):
        img_path = image_dir / f"{stem}.png"
        lbl_path = label_dir / f"{stem}.txt"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt = load_gt(lbl_path, w, h)

        enc = processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(args.device)
        pixel_mask = enc["pixel_mask"].to(args.device) if "pixel_mask" in enc else None
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        processed = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]], device=args.device),
            threshold=args.confidence_threshold,
        )[0]

        preds = []
        for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            preds.append(
                {
                    "score": float(score.item()),
                    "label": int(label.item()),
                    "box": [x1, y1, x2, y2],
                }
            )

        _tp, _fp, _fn = match_detections(preds, gt, args.iou_threshold)
        tp += _tp
        fp += _fp
        fn += _fn

        if len(gt) == 0:
            negative_frames += 1
            if len(preds) > 0:
                fp_on_negative += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    frame_fpr = (
        fp_on_negative / negative_frames if negative_frames > 0 else None
    )

    report = {
        "num_test_images": len(stems),
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tpr_recall": tpr,
        "precision": precision,
        "negative_frames": negative_frames,
        "fp_on_negative_frames": fp_on_negative,
        "frame_fpr": frame_fpr,
        "note": (
            "frame_fpr requires negative frames. If your dataset has pests in every frame, "
            "frame_fpr will be null."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
