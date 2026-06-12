"""Post-hoc prototype classifiers used by experiment 79.

The implementations in this module are deliberately independent from WinMax.
They operate either on the final classifier-input embedding (KMEx) or on
spatial feature maps (B4/B234).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class PrototypeModel:
    prototypes: np.ndarray
    prototype_classes: np.ndarray
    classes: np.ndarray
    prototypes_per_class: int
    fit_diagnostics: dict[str, Any]


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return (exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)).astype(
        np.float32
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    classes: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc_ovr_macro": float("nan"),
    }
    if probabilities is not None and len(classes) > 1:
        try:
            metrics["auroc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            pass
    return metrics


def fit_classwise_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    prototypes_per_class: int,
    seed: int,
    mode: str = "auto",
    minibatch_threshold: int = 750,
    minibatch_size: int = 256,
    max_iter: int = 200,
) -> PrototypeModel:
    """Fit one independent k-means model per class.

    This is the central KMEx operation and is also used by B4/B234. The
    resulting classifier is nearest prototype under Euclidean distance.
    """

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    classes = np.asarray(classes)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError(f"Expected aligned 2D features and labels, got {x.shape}, {y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Prototype fitting features contain NaN or Inf")

    centers: list[np.ndarray] = []
    center_classes: list[np.ndarray] = []
    class_rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(classes):
        class_features = x[y == label]
        if len(class_features) < prototypes_per_class:
            raise ValueError(
                f"Class {label} has {len(class_features)} observations, fewer than "
                f"{prototypes_per_class} requested prototypes"
            )
        selected_mode = mode
        if mode == "auto":
            selected_mode = (
                "minibatch" if len(class_features) > int(minibatch_threshold) else "full"
            )
        random_state = int(seed + 1009 * (class_index + 1) + 37 * prototypes_per_class)
        if selected_mode == "minibatch":
            estimator = MiniBatchKMeans(
                n_clusters=int(prototypes_per_class),
                random_state=random_state,
                n_init=10,
                max_iter=int(max_iter),
                batch_size=min(int(minibatch_size), len(class_features)),
                reassignment_ratio=0.0,
            )
        elif selected_mode == "full":
            estimator = KMeans(
                n_clusters=int(prototypes_per_class),
                random_state=random_state,
                n_init=10,
                max_iter=int(max_iter),
                algorithm="lloyd",
            )
        else:
            raise ValueError(f"Unknown k-means mode: {mode}")
        estimator.fit(class_features)
        centers.append(estimator.cluster_centers_.astype(np.float32))
        center_classes.append(
            np.full(int(prototypes_per_class), label, dtype=classes.dtype)
        )
        class_rows.append(
            {
                "class": int(label) if np.issubdtype(classes.dtype, np.integer) else str(label),
                "n_fit": int(len(class_features)),
                "mode": selected_mode,
                "inertia": float(estimator.inertia_),
                "n_iter": int(estimator.n_iter_),
            }
        )
    return PrototypeModel(
        prototypes=np.concatenate(centers, axis=0),
        prototype_classes=np.concatenate(center_classes, axis=0),
        classes=classes,
        prototypes_per_class=int(prototypes_per_class),
        fit_diagnostics={
            "class_fits": class_rows,
            "kmeans_mode_requested": mode,
            "minibatch_threshold": int(minibatch_threshold),
            "minibatch_size": int(minibatch_size),
            "max_iter": int(max_iter),
            "n_features": int(x.shape[1]),
            "n_fit_total": int(len(x)),
        },
    )


def _squared_distances(
    query: np.ndarray,
    prototypes: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    prototype_norm = np.sum(prototypes * prototypes, axis=1)[None, :]
    chunks = []
    for start in range(0, len(query), int(chunk_size)):
        part = query[start : start + int(chunk_size)]
        distance = (
            np.sum(part * part, axis=1, keepdims=True)
            + prototype_norm
            - 2.0 * (part @ prototypes.T)
        )
        chunks.append(np.maximum(distance, 0.0).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def predict_nearest_prototype(
    model: PrototypeModel,
    features: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """KMEx prediction and distance-derived class scores.

    Predictions exactly follow the nearest prototype. Probabilities are only
    distance-derived ranking scores used for AUROC and do not alter predictions.
    """

    distances = _squared_distances(features, model.prototypes, chunk_size)
    nearest = np.argmin(distances, axis=1)
    predictions = model.prototype_classes[nearest]
    class_distance = np.empty((len(features), len(model.classes)), dtype=np.float32)
    for index, label in enumerate(model.classes):
        class_distance[:, index] = np.min(
            distances[:, model.prototype_classes == label], axis=1
        )
    probabilities = softmax_rows(-np.sqrt(np.maximum(class_distance, 0.0)))
    return predictions, probabilities


class StratifiedPatchReservoir:
    """Bounded deterministic patch sample maintained independently by class."""

    def __init__(self, classes: np.ndarray, capacity_per_class: int, seed: int):
        self.classes = np.asarray(classes)
        self.capacity = int(capacity_per_class)
        self.rng = np.random.default_rng(int(seed))
        self._features: dict[Any, np.ndarray] = {}
        self._priorities: dict[Any, np.ndarray] = {}
        self.seen: dict[Any, int] = {label.item() if hasattr(label, "item") else label: 0 for label in self.classes}

    def update(self, label: Any, patches: np.ndarray) -> None:
        patches = np.asarray(patches, dtype=np.float32)
        if patches.ndim != 2 or len(patches) == 0:
            return
        key = label.item() if hasattr(label, "item") else label
        incoming_count = len(patches)
        priorities = self.rng.random(len(patches), dtype=np.float64)
        old_x = self._features.get(key)
        old_p = self._priorities.get(key)
        if old_x is not None:
            patches = np.concatenate([old_x, patches], axis=0)
            priorities = np.concatenate([old_p, priorities], axis=0)
        if len(patches) > self.capacity:
            keep = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            patches = patches[keep]
            priorities = priorities[keep]
        self._features[key] = patches
        self._priorities[key] = priorities
        self.seen[key] = int(self.seen.get(key, 0) + incoming_count)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        features = []
        labels = []
        retained = {}
        for label in self.classes:
            key = label.item() if hasattr(label, "item") else label
            class_features = self._features.get(key)
            if class_features is None or len(class_features) == 0:
                raise ValueError(f"No spatial patches retained for class {key}")
            features.append(class_features)
            labels.append(np.full(len(class_features), label, dtype=self.classes.dtype))
            retained[str(key)] = int(len(class_features))
        return (
            np.concatenate(features, axis=0).astype(np.float32),
            np.concatenate(labels, axis=0),
            {
                "capacity_per_class": self.capacity,
                "retained_per_class": retained,
                "seen_per_class": {str(k): int(v) for k, v in self.seen.items()},
            },
        )


def sample_patch_rows(
    descriptors: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if len(descriptors) <= int(max_rows):
        return descriptors
    indices = rng.choice(len(descriptors), size=int(max_rows), replace=False)
    return descriptors[indices]


def spatial_vote_from_distances(
    nearest_prototype_indices: np.ndarray,
    prototype_classes: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-patch nearest prototypes into the paper's class vote."""

    assignments = prototype_classes[np.asarray(nearest_prototype_indices)]
    probabilities = np.zeros((len(assignments), len(classes)), dtype=np.float32)
    for class_index, label in enumerate(classes):
        probabilities[:, class_index] = np.mean(assignments == label, axis=1)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return predictions, probabilities
