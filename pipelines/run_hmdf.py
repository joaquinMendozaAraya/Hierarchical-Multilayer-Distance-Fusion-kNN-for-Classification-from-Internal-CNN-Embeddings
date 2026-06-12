"""Run HMDF-kNN on one validated saved embedding profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hmdf_knn import HMDFKNN


def load_split(profile: Path, split: str) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    split_dir = profile / "embeddings" / "multicapa_norm" / split
    metadata_path = profile / "split_info.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata_path.exists()
        else {}
    )
    dimensions = [int(value) for value in metadata.get("embedding_dims", [])]
    files = [split_dir / f"z_dim_{dimension}.npy" for dimension in dimensions]
    files = [path for path in files if path.exists()]
    if not files:
        files = sorted(
            split_dir.glob("z_dim_*.npy"),
            key=lambda path: int(path.stem.removeprefix("z_dim_")),
        )
    if not files:
        raise FileNotFoundError(f"No embedding views in {split_dir}")
    views = [np.load(path).astype(np.float32) for path in files]
    labels = np.load(split_dir / "labels.npy").astype(int)
    names = [str(value) for value in metadata.get("hook_modules", [])]
    if len(names) != len(files):
        names = [path.stem for path in files]
    return views, labels, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/hmdf_run"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_views, y_train, names = load_split(args.profile, "train")
    val_views, y_val, _ = load_split(args.profile, "val")
    test_views, y_test, _ = load_split(args.profile, "test")

    classifier = HMDFKNN(seed=args.seed)
    classifier.fit(train_views, y_train, val_views, y_val, view_names=names)
    test_metrics = classifier.evaluate(test_views, y_test)

    args.output.mkdir(parents=True, exist_ok=True)
    classifier.save_selection(args.output / "selected_configuration.json")
    predictions = classifier.predict(test_views)
    np.savez_compressed(
        args.output / "test_predictions.npz",
        y_test=y_test,
        pred_test=predictions,
        classes=classifier.classes_,
    )
    summary = {
        "profile": str(args.profile.resolve()),
        "validation_selection": classifier.selection_.validation_metrics,
        "test_metrics": test_metrics,
        "test_used_for_selection": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
