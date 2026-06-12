"""Readable implementation of Hierarchical Multilayer Distance Fusion kNN.

The implementation follows the protocol used in the paper:

1. L2-normalize every layer view independently.
2. Rank layers using validation macro-F1, balanced accuracy, and accuracy.
3. Evaluate ranked prefixes with uniform, score-based, and reproducible
   Dirichlet distance weights.
4. Select the complete configuration on validation only.
5. Freeze that configuration before predicting test or unseen samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


EPS = 1e-12


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize each sample to unit L2 norm."""
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {array.shape}")
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), EPS, None)


def normalize_views(views: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [l2_normalize(view) for view in views]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics used for validation selection and test reporting."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def distance_vote(
    distances: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weighted kNN voting from a precomputed query-train matrix."""
    k = min(int(k), distances.shape[1])
    if k < 1:
        raise ValueError("k must be at least 1")
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    order = np.argsort(selected_distances, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)
    neighbor_labels = y_train[indices]
    neighbor_weights = 1.0 / np.clip(selected_distances, 1e-8, None)

    probabilities = np.zeros((len(distances), len(classes)), dtype=np.float32)
    for class_index, label in enumerate(classes):
        probabilities[:, class_index] = np.sum(
            neighbor_weights * (neighbor_labels == label), axis=1
        )
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), EPS, None)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


@dataclass(frozen=True)
class HMDFSelection:
    layer_indices: list[int]
    layer_names: list[str]
    dimensions: list[int]
    weights: list[float]
    weighting: str
    k: int
    validation_metrics: dict[str, float]
    candidate_count: int
    seed: int


class HMDFKNN:
    """Validation-selected multilayer distance-fusion classifier."""

    def __init__(
        self,
        *,
        k_grid: tuple[int, ...] = (1, 3, 5, 7, 11),
        max_views: int = 4,
        weight_powers: tuple[float, ...] = (0.5, 1.0, 2.0),
        dirichlet_trials: int = 8,
        seed: int = 42,
        distance_chunk_size: int = 512,
    ) -> None:
        self.k_grid = tuple(int(k) for k in k_grid)
        self.max_views = int(max_views)
        self.weight_powers = tuple(float(x) for x in weight_powers)
        self.dirichlet_trials = int(dirichlet_trials)
        self.seed = int(seed)
        self.distance_chunk_size = int(distance_chunk_size)

        self.selection_: HMDFSelection | None = None
        self.layer_ranking_: list[dict[str, Any]] = []
        self.validation_candidates_: list[dict[str, Any]] = []
        self.classes_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.train_views_: list[np.ndarray] | None = None

    @staticmethod
    def _validate_inputs(
        train_views: list[np.ndarray],
        val_views: list[np.ndarray],
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        if not train_views or len(train_views) != len(val_views):
            raise ValueError("Train and validation must contain the same layer views")
        if any(len(view) != len(y_train) for view in train_views):
            raise ValueError("A train embedding view is not aligned with y_train")
        if any(len(view) != len(y_val) for view in val_views):
            raise ValueError("A validation embedding view is not aligned with y_val")

    @staticmethod
    def _candidate_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(row["val_f1_macro"]),
            float(row["val_balanced_accuracy"]),
            float(row["val_accuracy"]),
            -int(row["final_dim"]),
        )

    def fit(
        self,
        train_views: list[np.ndarray],
        y_train: np.ndarray,
        val_views: list[np.ndarray],
        y_val: np.ndarray,
        *,
        view_names: list[str] | None = None,
    ) -> "HMDFKNN":
        """Select all HMDF-kNN hyperparameters using validation data only."""
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        self._validate_inputs(train_views, val_views, y_train, y_val)

        train_views = normalize_views(train_views)
        val_views = normalize_views(val_views)
        view_names = view_names or [f"layer_{i}" for i in range(len(train_views))]
        if len(view_names) != len(train_views):
            raise ValueError("view_names must match the number of embedding views")

        dimensions = [int(view.shape[1]) for view in train_views]
        classes = np.unique(np.concatenate([y_train, y_val]))

        layer_rows: list[dict[str, Any]] = []
        layer_scores: list[float] = []
        for index, (name, dimension, x_train, x_val) in enumerate(
            zip(view_names, dimensions, train_views, val_views)
        ):
            candidates = []
            for k in self.k_grid:
                if k >= len(y_train):
                    continue
                model = KNeighborsClassifier(
                    n_neighbors=k, weights="distance", n_jobs=-1
                ).fit(x_train, y_train)
                prediction = model.predict(x_val)
                metrics = classification_metrics(y_val, prediction)
                candidates.append(
                    {
                        "layer_index": index,
                        "layer_name": name,
                        "dim": dimension,
                        "k": k,
                        **{f"val_{key}": value for key, value in metrics.items()},
                    }
                )
            if not candidates:
                raise RuntimeError(f"No valid k candidate for layer {name}")
            best = max(
                candidates,
                key=lambda row: (
                    row["val_f1_macro"],
                    row["val_balanced_accuracy"],
                    row["val_accuracy"],
                    -row["k"],
                ),
            )
            layer_rows.append(best)
            layer_scores.append(float(best["val_f1_macro"]))

        ranking = sorted(
            range(len(layer_rows)),
            key=lambda index: (
                layer_rows[index]["val_f1_macro"],
                layer_rows[index]["val_balanced_accuracy"],
                layer_rows[index]["val_accuracy"],
            ),
            reverse=True,
        )

        rng = np.random.default_rng(self.seed)
        validation_candidates: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None

        for count in range(1, min(self.max_views, len(ranking)) + 1):
            chosen = ranking[:count]
            scores = np.clip(
                np.asarray([layer_scores[index] for index in chosen]), 1e-6, None
            )
            weight_options: list[tuple[str, np.ndarray]] = [
                ("uniform", np.ones(count, dtype=np.float32) / count)
            ]
            for power in self.weight_powers:
                raw = scores**power
                weight_options.append((f"val_power_{power:g}", raw / raw.sum()))
            alpha = np.clip(scores / max(float(scores.mean()), 1e-6), 0.25, 8.0)
            for trial in range(self.dirichlet_trials):
                weight_options.append(
                    (
                        f"dirichlet_{trial:02d}",
                        rng.dirichlet(alpha).astype(np.float32),
                    )
                )

            # This reproduces the audited implementation: identical vectors
            # are deduplicated by rounded weight values.
            deduplicated: dict[tuple[float, ...], tuple[str, np.ndarray]] = {}
            for label, weights in weight_options:
                deduplicated[tuple(np.round(weights, 8).tolist())] = (label, weights)

            selected_train = [train_views[index] for index in chosen]
            selected_val = [val_views[index] for index in chosen]
            for label, weights in deduplicated.values():
                fused_distance = sum(
                    float(weight)
                    * pairwise_distances(x_val, x_train, metric="euclidean", n_jobs=-1)
                    for weight, x_train, x_val in zip(
                        weights, selected_train, selected_val
                    )
                ).astype(np.float32)

                for k in self.k_grid:
                    if k >= len(y_train):
                        continue
                    prediction, _ = distance_vote(
                        fused_distance, y_train, classes, k
                    )
                    metrics = classification_metrics(y_val, prediction)
                    row = {
                        "layer_indices": list(chosen),
                        "layer_names": [view_names[index] for index in chosen],
                        "dimensions": [dimensions[index] for index in chosen],
                        "weights": weights.astype(float).tolist(),
                        "weighting": label,
                        "k": int(k),
                        "final_dim": int(sum(dimensions[index] for index in chosen)),
                        **{f"val_{key}": value for key, value in metrics.items()},
                    }
                    validation_candidates.append(row)
                    if best_row is None or self._candidate_key(row) > self._candidate_key(
                        best_row
                    ):
                        best_row = row

        if best_row is None:
            raise RuntimeError("No HMDF-kNN validation candidate was produced")

        self.classes_ = classes
        self.y_train_ = y_train
        self.train_views_ = train_views
        self.layer_ranking_ = layer_rows
        self.validation_candidates_ = validation_candidates
        self.selection_ = HMDFSelection(
            layer_indices=list(best_row["layer_indices"]),
            layer_names=list(best_row["layer_names"]),
            dimensions=list(best_row["dimensions"]),
            weights=list(best_row["weights"]),
            weighting=str(best_row["weighting"]),
            k=int(best_row["k"]),
            validation_metrics={
                key.removeprefix("val_"): float(value)
                for key, value in best_row.items()
                if key.startswith("val_")
            },
            candidate_count=len(validation_candidates),
            seed=self.seed,
        )
        return self

    def _check_fitted(self) -> None:
        if (
            self.selection_ is None
            or self.classes_ is None
            or self.y_train_ is None
            or self.train_views_ is None
        ):
            raise RuntimeError("Call fit() before prediction")

    def predict_proba(self, query_views: list[np.ndarray]) -> np.ndarray:
        """Predict with the frozen validation-selected configuration."""
        self._check_fitted()
        assert self.selection_ is not None
        assert self.classes_ is not None
        assert self.y_train_ is not None
        assert self.train_views_ is not None

        query_views = normalize_views(query_views)
        if len(query_views) != len(self.train_views_):
            raise ValueError("query_views must contain every original candidate view")

        indices = self.selection_.layer_indices
        weights = np.asarray(self.selection_.weights, dtype=np.float32)
        probabilities = []
        for start in range(0, len(query_views[0]), self.distance_chunk_size):
            stop = min(start + self.distance_chunk_size, len(query_views[0]))
            fused_distance = sum(
                float(weight)
                * pairwise_distances(
                    query_views[index][start:stop],
                    self.train_views_[index],
                    metric="euclidean",
                    n_jobs=-1,
                )
                for weight, index in zip(weights, indices)
            ).astype(np.float32)
            _, chunk_probabilities = distance_vote(
                fused_distance,
                self.y_train_,
                self.classes_,
                self.selection_.k,
            )
            probabilities.append(chunk_probabilities)
        return np.concatenate(probabilities, axis=0)

    def predict(self, query_views: list[np.ndarray]) -> np.ndarray:
        probabilities = self.predict_proba(query_views)
        assert self.classes_ is not None
        return self.classes_[np.argmax(probabilities, axis=1)]

    def evaluate(
        self, query_views: list[np.ndarray], y_true: np.ndarray
    ) -> dict[str, float]:
        return classification_metrics(np.asarray(y_true), self.predict(query_views))

    def save_selection(self, path: str | Path) -> None:
        self._check_fitted()
        assert self.selection_ is not None
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self.selection_), indent=2), encoding="utf-8"
        )
