"""Validate an HMDF-kNN saved embedding profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_profile(profile: Path) -> dict[str, object]:
    base = profile / "embeddings" / "multicapa_norm"
    if not base.exists():
        raise FileNotFoundError(f"Missing embedding directory: {base}")

    report: dict[str, object] = {"profile": str(profile.resolve()), "splits": {}}
    expected_dimensions: list[int] | None = None
    for split in ("train", "val", "test"):
        split_dir = base / split
        labels = np.load(split_dir / "labels.npy")
        view_files = sorted(
            split_dir.glob("z_dim_*.npy"),
            key=lambda path: int(path.stem.removeprefix("z_dim_")),
        )
        if not view_files:
            raise FileNotFoundError(f"No z_dim_*.npy files under {split_dir}")
        dimensions = [int(path.stem.removeprefix("z_dim_")) for path in view_files]
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError(f"{split} dimensions differ from train")

        for path in view_files:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or len(array) != len(labels):
                raise ValueError(f"Invalid alignment in {path}")
            if not np.isfinite(array).all():
                raise ValueError(f"NaN or Inf values in {path}")
        report["splits"][split] = {
            "samples": int(len(labels)),
            "classes": int(len(np.unique(labels))),
            "dimensions": dimensions,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = load_profile(args.profile)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
