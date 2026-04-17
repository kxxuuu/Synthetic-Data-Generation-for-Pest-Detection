#!/usr/bin/env python3
"""
Collect kitchen images from a local Places365 extraction.

Example:
  python scripts/collect_places365_kitchen.py \
    --places-root /data/places365 \
    --out-dir data/raw/kitchen_backgrounds \
    --mode symlink \
    --max-per-class 5000
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--places-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    p.add_argument("--max-per-class", type=int, default=0)
    p.add_argument(
        "--kitchen-keywords",
        nargs="+",
        default=["kitchen"],
        help="Category keywords matched against path parts.",
    )
    return p.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def infer_class(path: Path, keywords: list[str]) -> str | None:
    parts = [p.lower() for p in path.parts]
    for part in parts:
        for kw in keywords:
            if kw.lower() in part:
                return part
    return None


def load_category_map(root: Path) -> dict[int, str]:
    categories_path = root / "categories_places365.txt"
    if not categories_path.exists():
        return {}

    mapping: dict[int, str] = {}
    for line in categories_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        category, idx = raw.rsplit(" ", 1)
        mapping[int(idx)] = category.strip("/")
    return mapping


def collect_flat_val_split(
    root: Path,
    out: Path,
    mode: str,
    keywords: list[str],
    max_per_class: int,
) -> tuple[dict[str, int], list[tuple[str, str, str]], int]:
    labels_path = root / "places365_val.txt"
    image_root = root / "val_256"
    category_map = load_category_map(root)
    if not labels_path.exists() or not image_root.exists() or not category_map:
        return {}, [], 0

    class_counts: dict[str, int] = {}
    rows: list[tuple[str, str, str]] = []
    scanned = 0

    for line in labels_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rel_name, idx_str = raw.split()
        scanned += 1
        cls_path = category_map.get(int(idx_str))
        if cls_path is None:
            continue
        cls = infer_class(Path(cls_path), keywords)
        if cls is None:
            continue

        count = class_counts.get(cls, 0)
        if max_per_class > 0 and count >= max_per_class:
            continue

        src = image_root / rel_name
        if not src.exists() or not is_image(src):
            continue

        rel_out = f"{cls}/{count:06d}{src.suffix.lower()}"
        dst = out / rel_out
        link_or_copy(src, dst, mode)

        class_counts[cls] = count + 1
        rows.append((str(src.resolve()), str(dst), cls))

    return class_counts, rows, scanned


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    args = parse_args()
    root = args.places_root.resolve()
    out = args.out_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"--places-root does not exist or is not a directory: {root}"
        )
    out.mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, int] = {}
    manifest_path = out / "manifest.csv"
    rows: list[tuple[str, str, str]] = []

    class_counts, rows, scanned = collect_flat_val_split(
        root=root,
        out=out,
        mode=args.mode,
        keywords=args.kitchen_keywords,
        max_per_class=args.max_per_class,
    )

    if not rows:
        scanned = 0
        for path in root.rglob("*"):
            if not path.is_file() or not is_image(path):
                continue
            scanned += 1

            cls = infer_class(path, args.kitchen_keywords)
            if cls is None:
                continue

            count = class_counts.get(cls, 0)
            if args.max_per_class > 0 and count >= args.max_per_class:
                continue

            rel_name = f"{cls}/{count:06d}{path.suffix.lower()}"
            dst = out / rel_name
            link_or_copy(path, dst, args.mode)

            class_counts[cls] = count + 1
            rows.append((str(path.resolve()), str(dst), cls))

    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_path", "output_path", "class_name"])
        w.writerows(rows)

    total = sum(class_counts.values())
    print(f"Collected {total} images into {out}")
    print(f"Scanned image files: {scanned}")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
