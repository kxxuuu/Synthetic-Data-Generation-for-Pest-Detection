#!/usr/bin/env python3
"""
Download Places365 through torchvision.

Example:
  python scripts/download_places365.py \
    --root data/raw/places365 \
    --split train-standard \
    --small true
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import Places365


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=["train-standard", "train-challenge", "val", "test"],
        default="train-standard",
    )
    p.add_argument("--small", type=str2bool, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading Places365 split={args.split}, small={args.small} into {args.root}"
    )
    ds = Places365(
        root=str(args.root),
        split=args.split,
        small=args.small,
        download=True,
    )
    print(f"Done. Dataset size (examples): {len(ds)}")
    print("Tip: image files are extracted under the chosen root directory.")


if __name__ == "__main__":
    main()
