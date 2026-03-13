#!/usr/bin/env python3
"""
Train DETR on synthetic normalized-bbox labels.

Example:
  python scripts/train_detr.py \
    --data-dir data/generated/synth_v1_smoke4 \
    --split-json data/splits/synth_v1_smoke4/split.json \
    --output-dir outputs/detr_smoke4 \
    --epochs 20 --batch-size 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, DetrForObjectDetection


ID2LABEL = {0: "rat", 1: "mouse", 2: "cockroach"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="facebook/detr-resnet-50")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_normalized_bbox_labels(label_path: Path, w: int, h: int) -> list[dict[str, Any]]:
    txt = label_path.read_text().strip()
    if not txt:
        return []
    anns = []
    for line in txt.splitlines():
        cls, xc, yc, bw, bh = line.split()
        cls = int(cls)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        box_w = bw * w
        box_h = bh * h
        x_min = (xc * w) - box_w / 2.0
        y_min = (yc * h) - box_h / 2.0
        anns.append(
            {
                "bbox": [x_min, y_min, box_w, box_h],  # COCO xywh absolute
                "category_id": cls,
                "area": box_w * box_h,
                "iscrowd": 0,
            }
        )
    return anns


@dataclass
class Sample:
    pixel_values: torch.Tensor
    labels: dict[str, Any]


class DetectionDetrDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        stems: list[str],
        processor: AutoImageProcessor,
    ) -> None:
        self.data_dir = data_dir
        self.stems = stems
        self.processor = processor

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Sample:
        stem = self.stems[idx]
        img_path = self.data_dir / "images" / f"{stem}.png"
        lbl_path = self.data_dir / "labels" / f"{stem}.txt"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        anns = read_normalized_bbox_labels(lbl_path, w, h)
        target = {"image_id": idx, "annotations": anns}
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        return Sample(
            pixel_values=encoded["pixel_values"].squeeze(0),
            labels=encoded["labels"][0],
        )


def build_loader(
    dataset: Dataset,
    processor: AutoImageProcessor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    def collate_fn(batch: list[Sample]) -> dict[str, Any]:
        pixel_values = [x.pixel_values for x in batch]
        enc = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],
            "pixel_mask": enc["pixel_mask"],
            "labels": [x.labels for x in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def move_labels_to_device(labels: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    out = []
    for t in labels:
        out.append({k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()})
    return out


@torch.no_grad()
def evaluate_loss(model: DetrForObjectDetection, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = move_labels_to_device(batch["labels"], device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        losses.append(outputs.loss.item())
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_json.read_text())
    train_stems = split["train"]
    val_stems = split["val"]

    print(f"train={len(train_stems)} val={len(val_stems)}")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = DetrForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    device = args.device
    model.to(device)

    train_ds = DetectionDetrDataset(args.data_dir, train_stems, processor)
    val_ds = DetectionDetrDataset(args.data_dir, val_stems, processor)
    train_loader = build_loader(train_ds, processor, args.batch_size, args.num_workers, True)
    val_loader = build_loader(val_ds, processor, args.batch_size, args.num_workers, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    log_lines = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = move_labels_to_device(batch["labels"], device)

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(sum(losses) / max(len(losses), 1))
        val_loss = evaluate_loss(model, val_loader, device)
        line = f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        print(line)
        log_lines.append(line)

        last_dir = args.output_dir / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)
        (args.output_dir / "train.log").write_text("\n".join(log_lines) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            best_dir = args.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"saved best checkpoint (val_loss={best_val:.6f})")

    summary = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "checkpoint": args.checkpoint,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
