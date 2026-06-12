"""Example inference from saved train views and a frozen HMDF configuration.

HMDF-kNN is a retrieval classifier. Deployment therefore requires the selected
training embeddings, their labels, and the frozen configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hmdf_knn import HMDFKNN
from pipelines.run_hmdf import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--query-split", choices=("val", "test"), default="test"
    )
    args = parser.parse_args()

    train_views, y_train, names = load_split(args.profile, "train")
    val_views, y_val, _ = load_split(args.profile, "val")
    query_views, y_query, _ = load_split(args.profile, args.query_split)

    classifier = HMDFKNN(seed=42).fit(
        train_views, y_train, val_views, y_val, view_names=names
    )
    predictions = classifier.predict(query_views)
    print("Frozen selection:", classifier.selection_)
    print("Metrics:", classifier.evaluate(query_views, y_query))
    print("First predictions:", predictions[:20].tolist())


if __name__ == "__main__":
    main()
