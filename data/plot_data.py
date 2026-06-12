"""Plot class counts from labels.npy files in an embedding profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("class_counts.png"))
    args = parser.parse_args()

    base = args.profile / "embeddings" / "multicapa_norm"
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for axis, split in zip(axes, ("train", "val", "test")):
        labels = np.load(base / split / "labels.npy")
        classes, counts = np.unique(labels, return_counts=True)
        axis.bar(classes.astype(str), counts, color="#377EB8")
        axis.set_title(split.capitalize())
        axis.set_xlabel("Class")
        axis.set_ylabel("Samples")
        axis.tick_params(axis="x", labelrotation=90)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
