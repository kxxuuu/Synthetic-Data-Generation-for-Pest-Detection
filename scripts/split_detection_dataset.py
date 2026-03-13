#!/usr/bin/env python3
"""
Split normalized-bbox image/label pairs into train/val/test.

Example:
  python scripts/split_detection_dataset.py \
    --data-dir data/generated/synth_v1_smoke4 \
    --out-dir data/splits/synth_v1_smoke4 \
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def first_label_id(label_file: Path) -> int:
    txt = label_file.read_text().strip()
    if not txt:
        return -1
    return int(txt.splitlines()[0].split()[0])


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    image_dir = args.data_dir / "images"
    label_dir = args.data_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Expected <data-dir>/images and <data-dir>/labels")

    stems = sorted([p.stem for p in image_dir.glob("*.png")])
    pairs = []
    for stem in stems:
        img = image_dir / f"{stem}.png"
        lbl = label_dir / f"{stem}.txt"
        if img.exists() and lbl.exists():
            pairs.append((img, lbl, first_label_id(lbl)))

    if not pairs:
        raise RuntimeError("No image/label pairs found.")

    random.seed(args.seed)
    grouped: dict[int, list[tuple[Path, Path, int]]] = defaultdict(list)
    for item in pairs:
        grouped[item[2]].append(item)

    train, val, test = [], [], []
    for cls_id, items in grouped.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val : n_train + n_val + n_test])
        print(f"class={cls_id}: train={n_train}, val={n_val}, test={n_test}")

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "train": [x[0].stem for x in train],
        "val": [x[0].stem for x in val],
        "test": [x[0].stem for x in test],
    }
    out_json = args.out_dir / "split.json"
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"pairs={len(pairs)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"wrote: {out_json}")


if __name__ == "__main__":
    main()
