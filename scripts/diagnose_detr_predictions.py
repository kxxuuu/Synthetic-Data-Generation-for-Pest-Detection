#!/usr/bin/env python3
"""
Generate threshold diagnostics and prediction overlays for a DETR checkpoint.

Example:
  python scripts/diagnose_detr_predictions.py \
    --data-dir data/generated/video_frame_smoke_v4 \
    --split-json data/splits/video_frame_smoke_v4/split.json \
    --checkpoint-dir outputs/detr_video_frame_smoke_v4/best \
    --out-dir outputs/detr_video_frame_smoke_v4/diagnostics \
    --splits train,val,test \
    --thresholds 0.5,0.2,0.1,0.05,0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoImageProcessor, DetrForObjectDetection


ID2LABEL = {0: "rat", 1: "mouse", 2: "cockroach"}
GT_COLOR = (0, 255, 0)
PRED_COLOR = (255, 64, 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", default="test")
    p.add_argument("--thresholds", default="0.5,0.2,0.1,0.05,0.01")
    p.add_argument(
        "--max-preds-per-image-values",
        default="all,1,3,5,10",
        help="Comma-separated max predictions to keep after thresholding. Use 'all' for no cap.",
    )
    p.add_argument("--overlay-threshold", type=float, default=0.05)
    p.add_argument("--overlay-max-preds", type=int, default=5)
    p.add_argument("--max-overlays-per-split", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_thresholds(raw: str) -> list[float]:
    out = sorted({float(x.strip()) for x in raw.split(",") if x.strip()}, reverse=True)
    if 0.0 not in out:
        out.append(0.0)
    return out


def parse_max_preds_values(raw: str) -> list[int | None]:
    values: list[int | None] = []
    seen: set[str] = set()
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"all", "none", "null"}:
            key = "all"
            value = None
        else:
            value = int(token)
            key = str(value)
        if key not in seen:
            values.append(value)
            seen.add(key)
    if "all" not in seen:
        values.insert(0, None)
    return values


def max_preds_label(max_preds: int | None) -> str:
    return "all" if max_preds is None else str(max_preds)


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


def filtered_preds(preds: list[dict], threshold: float) -> list[dict]:
    return [p for p in preds if p["score"] >= threshold]


def apply_pred_policy(preds: list[dict], threshold: float, max_preds: int | None) -> list[dict]:
    out = filtered_preds(preds, threshold)
    if max_preds is not None:
        out = out[:max_preds]
    return out


def draw_overlay(
    image: Image.Image,
    gts: list[dict],
    preds: list[dict],
    stem: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for gt in gts:
        x1, y1, x2, y2 = gt["box"]
        label = ID2LABEL.get(gt["label"], str(gt["label"]))
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)
        draw.text((x1 + 2, max(0, y1 - 16)), f"GT:{label}", fill=GT_COLOR)
    for pred in preds:
        x1, y1, x2, y2 = pred["box"]
        label = ID2LABEL.get(pred["label"], str(pred["label"]))
        score = pred["score"]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=2)
        draw.text((x1 + 2, min(image.height - 16, y2 + 2)), f"P:{label} {score:.3f}", fill=PRED_COLOR)
    draw.text((8, 8), stem, fill=(255, 255, 0))
    canvas.save(out_path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    split_names = parse_csv_list(args.splits)
    thresholds = parse_thresholds(args.thresholds)
    max_preds_values = parse_max_preds_values(args.max_preds_per_image_values)
    iou_threshold = 0.5

    split_payload = json.loads(args.split_json.read_text())
    image_dir = args.data_dir / "images"
    label_dir = args.data_dir / "labels"
    processor = AutoImageProcessor.from_pretrained(str(args.checkpoint_dir), local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(str(args.checkpoint_dir), local_files_only=True).to(args.device)
    model.eval()

    overall = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "splits": {},
        "thresholds": thresholds,
        "max_preds_per_image_values": [max_preds_label(v) for v in max_preds_values],
        "overlay_threshold": args.overlay_threshold,
        "overlay_max_preds": args.overlay_max_preds,
        "iou_threshold": iou_threshold,
    }

    for split_name in split_names:
        stems = split_payload[split_name]
        rows: list[dict] = []
        top_scores_positive: list[float] = []
        top_scores_negative: list[float] = []
        aggregate = {
            str(thr): {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "total_predictions": 0,
                "images_with_prediction": 0,
                "positive_images_with_prediction": 0,
                "negative_images_with_prediction": 0,
            }
            for thr in thresholds
        }
        topk_aggregate = {
            str(thr): {
                max_preds_label(max_preds): {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "total_predictions": 0,
                    "images_with_prediction": 0,
                    "positive_images_with_prediction": 0,
                    "negative_images_with_prediction": 0,
                }
                for max_preds in max_preds_values
            }
            for thr in thresholds
        }

        for stem in tqdm(stems, desc=f"diagnose:{split_name}", leave=False):
            img_path = image_dir / f"{stem}.png"
            lbl_path = label_dir / f"{stem}.txt"
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            gts = load_gt(lbl_path, w, h)

            enc = processor(images=image, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(args.device)
            pixel_mask = enc["pixel_mask"].to(args.device) if "pixel_mask" in enc else None
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            processed = processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[h, w]], device=args.device),
                threshold=0.0,
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
            preds.sort(key=lambda x: x["score"], reverse=True)
            top_score = float(preds[0]["score"]) if preds else 0.0
            if gts:
                top_scores_positive.append(top_score)
            else:
                top_scores_negative.append(top_score)

            row = {
                "stem": stem,
                "gt_count": len(gts),
                "top_score": top_score,
                "top_label": preds[0]["label"] if preds else None,
                "top_label_name": ID2LABEL.get(preds[0]["label"]) if preds else None,
            }

            for thr in thresholds:
                preds_thr = filtered_preds(preds, thr)
                tp, fp, fn = match_detections(preds_thr, gts, iou_threshold)
                key = str(thr)
                aggregate[key]["tp"] += tp
                aggregate[key]["fp"] += fp
                aggregate[key]["fn"] += fn
                aggregate[key]["total_predictions"] += len(preds_thr)
                if preds_thr:
                    aggregate[key]["images_with_prediction"] += 1
                    if gts:
                        aggregate[key]["positive_images_with_prediction"] += 1
                    else:
                        aggregate[key]["negative_images_with_prediction"] += 1
                row[f"num_preds_ge_{thr}"] = len(preds_thr)
                row[f"tp_ge_{thr}"] = tp
                row[f"fp_ge_{thr}"] = fp
                row[f"fn_ge_{thr}"] = fn

                for max_preds in max_preds_values:
                    preds_policy = apply_pred_policy(preds, thr, max_preds)
                    topk_key = max_preds_label(max_preds)
                    tp_k, fp_k, fn_k = match_detections(preds_policy, gts, iou_threshold)
                    topk_aggregate[key][topk_key]["tp"] += tp_k
                    topk_aggregate[key][topk_key]["fp"] += fp_k
                    topk_aggregate[key][topk_key]["fn"] += fn_k
                    topk_aggregate[key][topk_key]["total_predictions"] += len(preds_policy)
                    if preds_policy:
                        topk_aggregate[key][topk_key]["images_with_prediction"] += 1
                        if gts:
                            topk_aggregate[key][topk_key]["positive_images_with_prediction"] += 1
                        else:
                            topk_aggregate[key][topk_key]["negative_images_with_prediction"] += 1

            rows.append(row)

        split_dir = args.out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        examples_path = split_dir / "examples.json"
        examples_path.write_text(json.dumps(rows, indent=2))

        positive_rows = [r for r in rows if r["gt_count"] > 0]
        negative_rows = [r for r in rows if r["gt_count"] == 0]
        top_positive = sorted(positive_rows, key=lambda x: x["top_score"], reverse=True)[: args.max_overlays_per_split]
        top_negative = sorted(negative_rows, key=lambda x: x["top_score"], reverse=True)[: args.max_overlays_per_split]

        for group_name, group_rows in [("top_positive", top_positive), ("top_negative", top_negative)]:
            for row in group_rows:
                stem = row["stem"]
                img_path = image_dir / f"{stem}.png"
                lbl_path = label_dir / f"{stem}.txt"
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                gts = load_gt(lbl_path, w, h)

                enc = processor(images=image, return_tensors="pt")
                pixel_values = enc["pixel_values"].to(args.device)
                pixel_mask = enc["pixel_mask"].to(args.device) if "pixel_mask" in enc else None
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                processed = processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([[h, w]], device=args.device),
                    threshold=args.overlay_threshold,
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
                preds.sort(key=lambda x: x["score"], reverse=True)
                preds = apply_pred_policy(preds, args.overlay_threshold, args.overlay_max_preds)
                out_path = split_dir / "overlays" / group_name / f"{stem}.png"
                draw_overlay(image, gts, preds, stem, out_path)

        split_summary = {
            "num_images": len(rows),
            "num_positive_images": len(positive_rows),
            "num_negative_images": len(negative_rows),
            "top_score_positive_quantiles": quantiles(top_scores_positive),
            "top_score_negative_quantiles": quantiles(top_scores_negative),
            "threshold_metrics": {},
            "threshold_topk_metrics": {},
            "examples_json": str(examples_path.resolve()),
        }

        for thr in thresholds:
            key = str(thr)
            tp = aggregate[key]["tp"]
            fp = aggregate[key]["fp"]
            fn = aggregate[key]["fn"]
            neg_frames = len(negative_rows)
            neg_frames_with_pred = aggregate[key]["negative_images_with_prediction"]
            split_summary["threshold_metrics"][key] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "total_predictions": aggregate[key]["total_predictions"],
                "avg_predictions_per_image": aggregate[key]["total_predictions"] / len(rows) if rows else 0.0,
                "images_with_prediction": aggregate[key]["images_with_prediction"],
                "positive_images_with_prediction": aggregate[key]["positive_images_with_prediction"],
                "negative_images_with_prediction": neg_frames_with_pred,
                "frame_fpr": neg_frames_with_pred / neg_frames if neg_frames > 0 else None,
            }
            split_summary["threshold_topk_metrics"][key] = {}
            for topk_key, metrics in topk_aggregate[key].items():
                tp_k = metrics["tp"]
                fp_k = metrics["fp"]
                fn_k = metrics["fn"]
                neg_with_pred_k = metrics["negative_images_with_prediction"]
                split_summary["threshold_topk_metrics"][key][topk_key] = {
                    "tp": tp_k,
                    "fp": fp_k,
                    "fn": fn_k,
                    "recall": tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0,
                    "precision": tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0,
                    "total_predictions": metrics["total_predictions"],
                    "avg_predictions_per_image": metrics["total_predictions"] / len(rows) if rows else 0.0,
                    "images_with_prediction": metrics["images_with_prediction"],
                    "positive_images_with_prediction": metrics["positive_images_with_prediction"],
                    "negative_images_with_prediction": neg_with_pred_k,
                    "frame_fpr": neg_with_pred_k / neg_frames if neg_frames > 0 else None,
                }

        overall["splits"][split_name] = split_summary

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
