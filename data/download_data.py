"""Print verified download guidance without redistributing medical data."""

from __future__ import annotations

import argparse
from pathlib import Path


KNOWN_KAGGLE_DATASETS = {
    "brain_tumor_mri_4c": "masoudnickparvar/brain-tumor-mri-dataset",
    "brain_tumor_mri_17c": "fernando2rad/brain-tumor-mri-images-17-classes",
    "brain_tumor_mri_44c": "fernando2rad/brain-tumor-mri-images-44c",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(KNOWN_KAGGLE_DATASETS))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download through kagglehub after accepting the dataset terms.",
    )
    args = parser.parse_args()

    slug = KNOWN_KAGGLE_DATASETS[args.dataset]
    print(f"Dataset: https://www.kaggle.com/datasets/{slug}")
    if not args.download:
        print("Review and accept the dataset license, then rerun with --download.")
        return

    import kagglehub

    source = Path(kagglehub.dataset_download(slug))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Kaggle cache: {source}")
    print(f"Requested project data root: {args.output.resolve()}")
    print("The source remains in the Kaggle cache; preparation is dataset-specific.")


if __name__ == "__main__":
    main()
