#!/usr/bin/env python3
"""
Train a torchvision Faster R-CNN baseline on normalized-bbox labels.

Example:
  python scripts/train_fasterrcnn.py \
    --data-dir data/generated/video_frame_smoke_v4 \
    --split-json data/splits/video_frame_smoke_v4/split.json \
    --output-dir outputs/fasterrcnn_video_frame_smoke_v4 \
    --epochs 3 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


ID2LABEL = {0: "rat", 1: "mouse", 2: "cockroach"}
NUM_CLASSES = len(ID2LABEL) + 1  # include background


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--arch",
        choices=["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn_v2"],
        default="fasterrcnn_mobilenet_v3_large_320_fpn",
    )
    p.add_argument(
        "--weights",
        choices=["default", "none"],
        default="default",
        help="Torchvision detection weights to initialize from.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument(
        "--batching-strategy",
        choices=["standard", "positive_anchor"],
        default="positive_anchor",
        help="How to form mini-batches. positive_anchor keeps at least one positive image in each batch.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_xyxy_boxes(label_path: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    txt = label_path.read_text().strip()
    if not txt:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0,), dtype=torch.float32),
        )

    boxes = []
    labels = []
    areas = []
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
        boxes.append([x1, y1, x2, y2])
        labels.append(cls + 1)  # reserve 0 for background
        areas.append(box_w * box_h)

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(areas, dtype=torch.float32),
    )


class DetectionTorchvisionDataset(Dataset):
    def __init__(self, data_dir: Path, stems: list[str]) -> None:
        self.data_dir = data_dir
        self.stems = stems
        self.positive_indices = []
        self.negative_indices = []
        for idx, stem in enumerate(stems):
            lbl_path = self.data_dir / "labels" / f"{stem}.txt"
            has_boxes = bool(lbl_path.read_text().strip())
            if has_boxes:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        stem = self.stems[idx]
        img_path = self.data_dir / "images" / f"{stem}.png"
        lbl_path = self.data_dir / "labels" / f"{stem}.txt"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes, labels, areas = read_xyxy_boxes(lbl_path, w, h)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return to_tensor(image), target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


class PositiveAnchoredBatchSampler(BatchSampler):
    def __init__(
        self,
        positive_indices: list[int],
        negative_indices: list[int],
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.positive_indices = list(positive_indices)
        self.negative_indices = list(negative_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        total = len(self.positive_indices) + len(self.negative_indices)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        positives = self.positive_indices.copy()
        negatives = self.negative_indices.copy()
        rng.shuffle(positives)
        rng.shuffle(negatives)

        if not positives:
            all_indices = negatives.copy()
            rng.shuffle(all_indices)
            for start in range(0, len(all_indices), self.batch_size):
                batch = all_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
            return

        num_batches = len(self)
        anchors = []
        while len(anchors) < num_batches:
            block = positives.copy()
            rng.shuffle(block)
            anchors.extend(block)
        anchors = anchors[:num_batches]

        filler_pool = negatives.copy()
        extra_positives = positives.copy()
        rng.shuffle(extra_positives)
        filler_pool.extend(extra_positives)
        filler_cursor = 0
        batches: list[list[int]] = []

        for anchor in anchors:
            batch = [anchor]
            while len(batch) < self.batch_size:
                if filler_cursor >= len(filler_pool):
                    refill = negatives.copy() + positives.copy()
                    rng.shuffle(refill)
                    filler_pool.extend(refill)
                candidate = filler_pool[filler_cursor]
                filler_cursor += 1
                if candidate in batch:
                    continue
                batch.append(candidate)
            batches.append(batch)

        rng.shuffle(batches)
        for batch in batches:
            yield batch


def build_loader(
    dataset: DetectionTorchvisionDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    batching_strategy: str,
    seed: int,
) -> DataLoader:
    if batching_strategy == "positive_anchor" and batch_size > 1:
        batch_sampler = PositiveAnchoredBatchSampler(
            positive_indices=dataset.positive_indices,
            negative_indices=dataset.negative_indices,
            batch_size=batch_size,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _resolve_weights(arch: str, weights: str):
    if weights == "none":
        return None
    if arch == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    return FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT


def build_model(arch: str, weights: str):
    resolved_weights = _resolve_weights(arch, weights)
    if arch == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=resolved_weights)
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=resolved_weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def move_targets_to_device(targets: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    out = []
    for target in targets:
        out.append({k: v.to(device) if hasattr(v, "to") else v for k, v in target.items()})
    return out


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, int]:
    was_training = model.training
    model.train()
    losses = []
    skipped = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        if not torch.isfinite(loss):
            skipped += 1
            continue
        losses.append(float(loss.item()))
    model.train(was_training)
    if not losses:
        return float("inf"), skipped
    return float(sum(losses) / max(len(losses), 1)), skipped


def save_checkpoint(out_dir: Path, model: torch.nn.Module, arch: str, weights: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "weights": weights,
            "num_classes": NUM_CLASSES,
            "id2label": ID2LABEL,
            "state_dict": model.state_dict(),
        },
        out_dir / "model.pt",
    )
    metadata = {
        "arch": arch,
        "weights": weights,
        "num_classes": NUM_CLASSES,
        "id2label": ID2LABEL,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    split = json.loads(args.split_json.read_text())
    train_stems = split["train"]
    val_stems = split["val"]

    train_ds = DetectionTorchvisionDataset(args.data_dir, train_stems)
    val_ds = DetectionTorchvisionDataset(args.data_dir, val_stems)
    print(
        " ".join(
            [
                f"train={len(train_stems)}",
                f"val={len(val_stems)}",
                f"train_pos={len(train_ds.positive_indices)}",
                f"train_neg={len(train_ds.negative_indices)}",
                f"val_pos={len(val_ds.positive_indices)}",
                f"val_neg={len(val_ds.negative_indices)}",
                f"batching_strategy={args.batching_strategy}",
            ]
        )
    )
    train_loader = build_loader(
        train_ds,
        args.batch_size,
        args.num_workers,
        True,
        args.batching_strategy,
        args.seed,
    )
    val_loader = build_loader(
        val_ds,
        args.batch_size,
        args.num_workers,
        False,
        args.batching_strategy,
        args.seed + 10_000,
    )

    model = build_model(args.arch, args.weights)
    device = args.device
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    log_lines = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        skipped_train = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)

            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            if not torch.isfinite(loss):
                skipped_train += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nan_skip")
                continue
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(sum(losses) / max(len(losses), 1))
        val_loss, skipped_val = evaluate_loss(model, val_loader, device)
        line = (
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"skipped_train_batches={skipped_train} skipped_val_batches={skipped_val}"
        )
        print(line)
        log_lines.append(line)

        last_dir = args.output_dir / "last"
        save_checkpoint(last_dir, model, args.arch, args.weights)
        (args.output_dir / "train.log").write_text("\n".join(log_lines) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            best_dir = args.output_dir / "best"
            save_checkpoint(best_dir, model, args.arch, args.weights)
            print(f"saved best checkpoint (val_loss={best_val:.6f})")

    summary = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_positive_images": len(train_ds.positive_indices),
        "train_negative_images": len(train_ds.negative_indices),
        "val_positive_images": len(val_ds.positive_indices),
        "val_negative_images": len(val_ds.negative_indices),
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "arch": args.arch,
        "weights": args.weights,
        "grad_clip_norm": args.grad_clip_norm,
        "batching_strategy": args.batching_strategy,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()
