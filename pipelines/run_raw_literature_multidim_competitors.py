#!/usr/bin/env python
"""Run raw literature multi-view competitors against a frozen WinMax reference.

Protocol:
1. The same frozen layer embeddings and split are used by every method.
2. External competitors cannot call WinMax/VCHMF selection or fusion code.
3. Every candidate is fitted on train and selected on validation.
4. Test is evaluated once, after the candidate is frozen.
5. Every selected row reports its delta against WinMax in the same context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multidataset_benchmark import (  # noqa: E402
    EmbeddingProfile,
    discover_embedding_profiles,
    safe_name,
    write_profile_registry,
)
from src.raw_multiview_competitors import (  # noqa: E402
    METHOD_FUNCTIONS,
    MethodOutput,
    MultiViewData,
    classification_metrics,
    estimator_proba,
    l2_views,
    set_candidate_reporter,
)


DEFAULT_EMBEDDING_ROOT = (
    ROOT / "experiments" / "51_final_vchmf_all_scenarios" / "embeddings"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "78_raw_literature_multidim_competitors"

CONTROL_METHODS = [
    "raw_concat_linear",
    "concat_pca_linear",
    "uniform_layer_softvote",
    "uniform_kernel_svm",
]
CORE_LITERATURE_METHODS = [
    "fradi_mlcff",
    "maxvar_gcca",
    "gmlda",
    "mvda",
]
HEAVY_LITERATURE_METHODS = [
    "head2toe",
    "easymkl",
    "concat_nca_knn",
]
ALL_EXTERNAL_METHODS = CONTROL_METHODS + CORE_LITERATURE_METHODS + HEAVY_LITERATURE_METHODS


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in parse_csv_strings(text)]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in parse_csv_strings(text)]


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def profile_metadata(profile: EmbeddingProfile) -> dict[str, Any]:
    candidates = [
        profile.path / "split_info.json",
        profile.path / "splits" / "split_info.json",
    ]
    for path in candidates:
        info = read_json(path)
        if isinstance(info, dict):
            return info
    return {}


def ordered_views(profile: EmbeddingProfile) -> tuple[list[int], list[str]]:
    info = profile_metadata(profile)
    dimensions = [int(x) for x in info.get("embedding_dims", [])]
    dimensions = [x for x in dimensions if x in profile.available_dims]
    if not dimensions:
        dimensions = list(profile.available_dims)
    names = [str(x) for x in info.get("hook_modules", [])]
    if len(names) != len(dimensions):
        names = [f"layer_dim_{dim}" for dim in dimensions]
    return dimensions, names


def load_multiview_data(profile: EmbeddingProfile) -> MultiViewData:
    dimensions, names = ordered_views(profile)
    split_views: dict[str, list[np.ndarray]] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        base = profile.path / "embeddings" / "multicapa_norm" / split
        labels = np.load(base / "labels.npy").astype(int)
        views = []
        for dim in dimensions:
            array = np.load(base / f"z_dim_{dim}.npy").astype(np.float32)
            if len(array) != len(labels):
                raise ValueError(
                    f"{split} dimension {dim}: {len(array)} rows but {len(labels)} labels"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{split} dimension {dim} contains NaN/Inf")
            views.append(array)
        split_views[split] = views
        split_labels[split] = labels
    classes = np.unique(
        np.concatenate(
            [split_labels["train"], split_labels["val"], split_labels["test"]]
        )
    )
    missing_train = sorted(set(classes.tolist()) - set(np.unique(split_labels["train"]).tolist()))
    if missing_train:
        raise ValueError(f"Classes absent from train: {missing_train}")
    return MultiViewData(
        train_views=split_views["train"],
        val_views=split_views["val"],
        test_views=split_views["test"],
        y_train=split_labels["train"],
        y_val=split_labels["val"],
        y_test=split_labels["test"],
        view_names=names,
        view_dims=dimensions,
        classes=classes,
    )


def audit_profile(profile: EmbeddingProfile, data: MultiViewData) -> dict[str, Any]:
    split_class_counts = {
        "train": {str(k): int(v) for k, v in zip(*np.unique(data.y_train, return_counts=True))},
        "val": {str(k): int(v) for k, v in zip(*np.unique(data.y_val, return_counts=True))},
        "test": {str(k): int(v) for k, v in zip(*np.unique(data.y_test, return_counts=True))},
    }
    identity_status = "unavailable_no_saved_sample_ids"
    duplicate_count = None
    paths_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        candidates = [
            profile.path / "embeddings" / "multicapa_norm" / split / "paths.npy",
            profile.path / "embeddings" / "multicapa_norm" / split / "paths.csv",
        ]
        for candidate in candidates:
            if candidate.suffix == ".npy" and candidate.exists():
                paths_by_split[split] = set(map(str, np.load(candidate, allow_pickle=True).tolist()))
                break
            if candidate.suffix == ".csv" and candidate.exists():
                frame = pd.read_csv(candidate)
                paths_by_split[split] = set(frame.iloc[:, 0].astype(str).tolist())
                break
    if len(paths_by_split) == 3:
        duplicate_count = sum(
            len(paths_by_split[a] & paths_by_split[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        identity_status = "passed" if duplicate_count == 0 else "failed"
        if duplicate_count:
            raise ValueError(f"Detected {duplicate_count} duplicated sample IDs across splits")
    return {
        "dataset_id": profile.dataset_id,
        "run_id": profile.run_id,
        "profile_name": profile.profile_name,
        "profile_path": str(profile.path.resolve()),
        "view_names": data.view_names,
        "view_dims": data.view_dims,
        "n_views": len(data.view_dims),
        "n_train": len(data.y_train),
        "n_val": len(data.y_val),
        "n_test": len(data.y_test),
        "classes": data.classes.tolist(),
        "split_class_counts": split_class_counts,
        "sample_identity_audit": identity_status,
        "cross_split_duplicate_count": duplicate_count,
        "finite_values": True,
        "label_alignment": True,
    }


def _distance_vote(
    distances: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(k), distances.shape[1])
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    order = np.argsort(selected_distances, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)
    neighbor_labels = y_train[indices]
    weights = 1.0 / np.clip(selected_distances, 1e-8, None)
    probabilities = np.zeros((len(distances), len(classes)), dtype=np.float32)
    for class_idx, label in enumerate(classes):
        probabilities[:, class_idx] = np.sum(
            weights * (neighbor_labels == label), axis=1
        )
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def _fused_distance_predict(
    train_views: list[np.ndarray],
    query_views: list[np.ndarray],
    weights: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_pred = []
    all_proba = []
    for start in range(0, len(query_views[0]), chunk_size):
        stop = min(start + chunk_size, len(query_views[0]))
        fused = None
        for weight, train, query in zip(weights, train_views, query_views):
            distance = pairwise_distances(
                query[start:stop],
                train,
                metric="euclidean",
                n_jobs=-1,
            ).astype(np.float32)
            fused = float(weight) * distance if fused is None else fused + float(weight) * distance
        pred, proba = _distance_vote(fused, y_train, classes, k)
        all_pred.append(pred)
        all_proba.append(proba)
    return np.concatenate(all_pred), np.concatenate(all_proba)


def _winmax_candidate(
    config: dict[str, Any],
    data: MultiViewData,
    val_pred: np.ndarray,
    val_proba: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    row = {
        "config_json": json.dumps(config, sort_keys=True),
        "fit_seconds": float(elapsed),
        "final_dim": int(sum(config["dims"])),
        "n_train_fit": len(data.y_train),
    }
    row.update(
        {
            f"val_{key}": value
            for key, value in classification_metrics(
                data.y_val, val_pred, val_proba, data.classes
            ).items()
        }
    )
    return row


def make_candidate_reporter(
    method_id: str,
    method_dir: Path,
    enabled: bool,
):
    counter = 0
    log_path = method_dir / "candidate_progress.jsonl"
    method_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    def report(row: dict[str, Any]) -> None:
        nonlocal counter
        counter += 1
        record = {
            "candidate": counter,
            "method_id": method_id,
            "config_json": row.get("config_json", "{}"),
            "val_f1_macro": row.get("val_f1_macro"),
            "val_balanced_accuracy": row.get("val_balanced_accuracy"),
            "val_accuracy": row.get("val_accuracy"),
            "fit_seconds": row.get("fit_seconds"),
        }
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if enabled:
            config = str(record["config_json"])
            if len(config) > 180:
                config = config[:177] + "..."
            print(
                f"    candidate {counter:03d} | "
                f"val F1={float(record['val_f1_macro']):.5f} | "
                f"bal={float(record['val_balanced_accuracy']):.5f} | "
                f"acc={float(record['val_accuracy']):.5f} | "
                f"{config}",
                flush=True,
            )

    return report


def _candidate_is_better(row: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current = (
        row["val_f1_macro"],
        row["val_balanced_accuracy"],
        row["val_accuracy"],
        -row["final_dim"],
    )
    previous = (
        best["val_f1_macro"],
        best["val_balanced_accuracy"],
        best["val_accuracy"],
        -best["final_dim"],
    )
    return current > previous


def run_winmax_reference(
    data: MultiViewData,
    args: argparse.Namespace,
    candidate_reporter=None,
) -> MethodOutput:
    train_views, val_views, test_views = map(
        l2_views,
        [data.train_views, data.val_views, data.test_views],
    )
    layer_rows = []
    layer_scores = []
    for idx, (name, dim, xtr, xva) in enumerate(
        zip(data.view_names, data.view_dims, train_views, val_views)
    ):
        best = None
        for k in args.k_grid:
            if k >= len(data.y_train):
                continue
            start = time.perf_counter()
            model = KNeighborsClassifier(
                n_neighbors=k, weights="distance", n_jobs=-1
            ).fit(xtr, data.y_train)
            pred = model.predict(xva)
            proba = estimator_proba(model, xva, data.classes)
            metrics = classification_metrics(data.y_val, pred, proba, data.classes)
            row = {
                "layer_index": idx,
                "layer_name": name,
                "dim": dim,
                "k": k,
                "fit_seconds": time.perf_counter() - start,
                **{f"val_{key}": value for key, value in metrics.items()},
            }
            if best is None or (
                row["val_f1_macro"],
                row["val_balanced_accuracy"],
                row["val_accuracy"],
                -k,
            ) > (
                best["val_f1_macro"],
                best["val_balanced_accuracy"],
                best["val_accuracy"],
                -best["k"],
            ):
                best = row
        if best is None:
            raise RuntimeError(f"No WinMax layer candidate for {name}")
        layer_rows.append(best)
        layer_scores.append(best["val_f1_macro"])
    ranking = sorted(
        range(len(layer_rows)),
        key=lambda idx: (
            layer_rows[idx]["val_f1_macro"],
            layer_rows[idx]["val_balanced_accuracy"],
            layer_rows[idx]["val_accuracy"],
        ),
        reverse=True,
    )
    max_views = min(args.winmax_max_views, len(ranking))
    rng = np.random.default_rng(args.seed)
    candidates = []
    best_row = None
    best_pack = None
    for count in range(1, max_views + 1):
        chosen = ranking[:count]
        scores = np.clip(np.asarray([layer_scores[i] for i in chosen]), 1e-6, None)
        weight_options: list[tuple[str, np.ndarray]] = [
            ("uniform", np.ones(count, dtype=np.float32) / count)
        ]
        for power in args.winmax_weight_powers:
            raw = scores ** float(power)
            weight_options.append((f"val_power_{power:g}", raw / raw.sum()))
        alpha = np.clip(scores / max(float(scores.mean()), 1e-6), 0.25, 8.0)
        for trial in range(args.winmax_dirichlet_trials):
            weight_options.append(
                (f"dirichlet_{trial:02d}", rng.dirichlet(alpha).astype(np.float32))
            )
        deduplicated = {}
        for label, weights in weight_options:
            deduplicated[tuple(np.round(weights, 8).tolist())] = (label, weights)
        for label, weights in deduplicated.values():
            start = time.perf_counter()
            selected_train = [train_views[i] for i in chosen]
            selected_val = [val_views[i] for i in chosen]
            fused = None
            for weight, train, val in zip(weights, selected_train, selected_val):
                distance = pairwise_distances(
                    val, train, metric="euclidean", n_jobs=-1
                ).astype(np.float32)
                fused = float(weight) * distance if fused is None else fused + float(weight) * distance
            distance_seconds = time.perf_counter() - start
            for k in args.k_grid:
                if k >= len(data.y_train):
                    continue
                pred, proba = _distance_vote(
                    fused, data.y_train, data.classes, int(k)
                )
                config = {
                    "layer_indices": chosen,
                    "layer_names": [data.view_names[i] for i in chosen],
                    "dims": [data.view_dims[i] for i in chosen],
                    "weights": weights.astype(float).tolist(),
                    "weighting": label,
                    "k": int(k),
                }
                row = _winmax_candidate(
                    config, data, pred, proba, distance_seconds
                )
                candidates.append(row)
                if candidate_reporter is not None:
                    candidate_reporter(row)
                if _candidate_is_better(row, best_row):
                    best_row, best_pack = row, (config, pred, proba)
    if best_pack is None or best_row is None:
        raise RuntimeError("No WinMax candidate was produced")
    config, val_pred, val_proba = best_pack
    indices = config["layer_indices"]
    test_pred, test_proba = _fused_distance_predict(
        [train_views[i] for i in indices],
        [test_views[i] for i in indices],
        np.asarray(config["weights"], dtype=np.float32),
        data.y_train,
        data.classes,
        int(config["k"]),
        args.distance_chunk_size,
    )
    return MethodOutput(
        method_id="winmax_reference",
        display_name="Proposed method (WinMax)",
        fidelity="frozen_proposed_reference",
        source="Current thesis implementation; validation-selected weighted layer-distance fusion with kNN",
        candidates=candidates,
        selected_config=config,
        val_pred=val_pred,
        test_pred=test_pred,
        val_proba=val_proba,
        test_proba=test_proba,
        diagnostics={
            "layer_validation_ranking": layer_rows,
            "test_evaluations_after_selection": 1,
            "uses_external_data": False,
        },
    )


def method_kwargs(method_id: str, args: argparse.Namespace) -> dict[str, Any]:
    common = {"seed": args.seed}
    mapping = {
        "raw_concat_linear": {**common, "c_grid": args.c_grid},
        "concat_pca_linear": {
            **common,
            "pca_dims": args.pca_dims,
            "c_grid": args.c_grid,
        },
        "uniform_layer_softvote": {**common, "c_grid": args.c_grid},
        "uniform_kernel_svm": {**common, "c_grid": args.c_grid},
        "fradi_mlcff": {
            **common,
            "pca_dims": args.pca_dims,
            "c_grid": args.c_grid,
        },
        "head2toe": {
            **common,
            "lambdas": args.head2toe_lambdas,
            "keep_fractions": args.head2toe_keep_fractions,
            "steps": args.head2toe_steps,
            "batch_size": args.head2toe_batch_size,
            "learning_rate": args.head2toe_learning_rate,
            "device": args.device,
        },
        "easymkl": {
            **common,
            "lambda_grid": args.easymkl_lambdas,
            "c_grid": args.c_grid,
            "max_train": args.easymkl_max_train,
        },
        "maxvar_gcca": {
            **common,
            "q_grid": args.view_pca_dims,
            "latent_dims": args.latent_dims,
            "ridge_grid": args.projection_regularization,
            "k_grid": args.k_grid,
        },
        "gmlda": {
            **common,
            "q_grid": args.view_pca_dims,
            "latent_dims": args.latent_dims,
            "alpha_grid": args.gmlda_alpha,
            "regularization_grid": args.projection_regularization,
            "k_grid": args.k_grid,
        },
        "mvda": {
            **common,
            "q_grid": args.view_pca_dims,
            "latent_dims": args.latent_dims,
            "regularization_grid": args.projection_regularization,
            "k_grid": args.k_grid,
        },
        "concat_nca_knn": {
            **common,
            "pca_dims": args.pca_dims,
            "nca_dims": args.nca_dims,
            "k_grid": args.k_grid,
            "max_fit_samples": args.nca_max_fit_samples,
            "max_iter": args.nca_max_iter,
        },
    }
    return mapping[method_id]


def profile_output_dir(
    output: Path, profile: EmbeddingProfile, arch: str
) -> Path:
    path = (
        output
        / "profiles"
        / safe_name(profile.dataset_id)
        / safe_name(arch)
        / safe_name(profile.profile_name)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_result_path(profile_dir: Path, method_id: str) -> Path:
    return profile_dir / "methods" / method_id / "selected_result.json"


def save_method_output(
    output: MethodOutput,
    data: MultiViewData,
    method_dir: Path,
    context: dict[str, Any],
    winmax_result: dict[str, Any] | None,
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame = pd.DataFrame(output.candidates)
    candidate_frame.insert(0, "method_id", output.method_id)
    candidate_frame.to_csv(method_dir / "validation_candidates.csv", index=False)
    val_metrics = classification_metrics(
        data.y_val, output.val_pred, output.val_proba, data.classes
    )
    test_metrics = classification_metrics(
        data.y_test, output.test_pred, output.test_proba, data.classes
    )
    row: dict[str, Any] = {
        **context,
        "method_id": output.method_id,
        "display_name": output.display_name,
        "fidelity": output.fidelity,
        "source": output.source,
        "selected_config": output.selected_config,
        "n_validation_candidates": len(output.candidates),
        "test_evaluations_after_selection": 1,
        "uses_test_for_selection": False,
        "uses_external_data": False,
        "uses_alternative_backbone": False,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    if winmax_result is not None:
        row.update(
            {
                "winmax_test_f1_macro": winmax_result["test_f1_macro"],
                "winmax_test_accuracy": winmax_result["test_accuracy"],
                "winmax_test_balanced_accuracy": winmax_result[
                    "test_balanced_accuracy"
                ],
                "delta_test_f1_macro_vs_winmax": test_metrics["f1_macro"]
                - winmax_result["test_f1_macro"],
                "delta_test_accuracy_vs_winmax": test_metrics["accuracy"]
                - winmax_result["test_accuracy"],
                "delta_test_balanced_accuracy_vs_winmax": test_metrics[
                    "balanced_accuracy"
                ]
                - winmax_result["test_balanced_accuracy"],
            }
        )
    else:
        row.update(
            {
                "winmax_test_f1_macro": test_metrics["f1_macro"],
                "winmax_test_accuracy": test_metrics["accuracy"],
                "winmax_test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "delta_test_f1_macro_vs_winmax": 0.0,
                "delta_test_accuracy_vs_winmax": 0.0,
                "delta_test_balanced_accuracy_vs_winmax": 0.0,
            }
        )
    np.savez_compressed(
        method_dir / "predictions.npz",
        y_val=data.y_val,
        pred_val=output.val_pred,
        proba_val=np.asarray(output.val_proba)
        if output.val_proba is not None
        else np.empty((0, 0)),
        y_test=data.y_test,
        pred_test=output.test_pred,
        proba_test=np.asarray(output.test_proba)
        if output.test_proba is not None
        else np.empty((0, 0)),
        classes=data.classes,
    )
    matrix = confusion_matrix(data.y_test, output.test_pred, labels=data.classes)
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in data.classes],
        columns=[f"pred_{label}" for label in data.classes],
    ).to_csv(method_dir / "test_confusion.csv")
    report = classification_report(
        data.y_test,
        output.test_pred,
        labels=data.classes,
        output_dict=True,
        zero_division=0,
    )
    # Short filenames avoid the legacy Windows MAX_PATH limit for the longest
    # dataset/profile/method combinations.
    pd.DataFrame(report).T.to_csv(method_dir / "test_report.csv")
    atomic_json(output.diagnostics, method_dir / "diagnostics.json")
    atomic_json(row, method_dir / "selected_result.json")
    atomic_json(
        {
            "status": "completed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method_id": output.method_id,
        },
        method_dir / "done.json",
    )
    return row


def load_selected(path: Path) -> dict[str, Any] | None:
    value = read_json(path)
    return value if isinstance(value, dict) else None


def load_completed_result(method_dir: Path) -> dict[str, Any] | None:
    selected = load_selected(method_dir / "selected_result.json")
    done = read_json(method_dir / "done.json")
    if (
        selected is None
        or not isinstance(done, dict)
        or done.get("status") != "completed"
        or not (method_dir / "predictions.npz").exists()
    ):
        return None
    return selected


def resolve_methods(specification: str) -> list[str]:
    requested = parse_csv_strings(specification)
    if not requested or requested == ["all"]:
        return list(ALL_EXTERNAL_METHODS)
    expanded: list[str] = []
    for item in requested:
        if item == "controls":
            expanded.extend(CONTROL_METHODS)
        elif item == "core":
            expanded.extend(CORE_LITERATURE_METHODS)
        elif item == "heavy":
            expanded.extend(HEAVY_LITERATURE_METHODS)
        elif item in METHOD_FUNCTIONS:
            expanded.append(item)
        else:
            raise ValueError(
                f"Unknown method '{item}'. Valid: controls, core, heavy, all, "
                + ", ".join(METHOD_FUNCTIONS)
            )
    return list(dict.fromkeys(expanded))


def write_error(
    output: Path,
    profile: EmbeddingProfile,
    arch: str,
    method_id: str,
    exc: BaseException,
) -> None:
    error_path = output / "error_log.csv"
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_id": profile.dataset_id,
        "arch": arch,
        "profile_name": profile.profile_name,
        "method_id": method_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    old = pd.read_csv(error_path) if error_path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
        error_path, index=False
    )


def collect_results(output: Path) -> pd.DataFrame:
    rows = []
    for path in output.glob("profiles/*/*/*/methods/*/selected_result.json"):
        value = read_json(path)
        if isinstance(value, dict):
            value["selected_result_path"] = str(path.resolve())
            value["selected_config_json"] = json.dumps(
                value.pop("selected_config", {}), sort_keys=True
            )
            rows.append(value)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["dataset_id", "arch", "method_id"]
    ).reset_index(drop=True)
    frame.to_csv(output / "all_selected_results.csv", index=False)
    delta = frame[
        [
            "dataset_id",
            "arch",
            "profile_name",
            "method_id",
            "display_name",
            "test_f1_macro",
            "test_accuracy",
            "test_balanced_accuracy",
            "delta_test_f1_macro_vs_winmax",
            "delta_test_accuracy_vs_winmax",
            "delta_test_balanced_accuracy_vs_winmax",
        ]
    ].copy()
    delta.to_csv(output / "delta_vs_winmax_by_context.csv", index=False)
    external = frame[frame["method_id"] != "winmax_reference"].copy()
    external["is_ham_control"] = external["dataset_id"].str.contains(
        "ham", case=False, na=False
    )
    external["f1_outcome_vs_winmax"] = np.select(
        [
            external["delta_test_f1_macro_vs_winmax"] > 0,
            external["delta_test_f1_macro_vs_winmax"] < 0,
        ],
        ["win", "loss"],
        default="tie",
    )
    summary_rows = []
    for scope, scoped in (
        ("brain_mri_main", external[~external["is_ham_control"]]),
        ("ham_external_control", external[external["is_ham_control"]]),
        ("all_contexts", external),
    ):
        for method_id, group in scoped.groupby("method_id"):
            summary_rows.append(
                {
                    "scope": scope,
                    "method_id": method_id,
                    "display_name": group["display_name"].iloc[0],
                    "n_contexts": len(group),
                    "wins_vs_winmax": int(
                        (group["delta_test_f1_macro_vs_winmax"] > 0).sum()
                    ),
                    "ties_vs_winmax": int(
                        (group["delta_test_f1_macro_vs_winmax"] == 0).sum()
                    ),
                    "losses_vs_winmax": int(
                        (group["delta_test_f1_macro_vs_winmax"] < 0).sum()
                    ),
                    "mean_test_f1_macro": float(group["test_f1_macro"].mean()),
                    "mean_delta_f1_vs_winmax": float(
                        group["delta_test_f1_macro_vs_winmax"].mean()
                    ),
                    "median_delta_f1_vs_winmax": float(
                        group["delta_test_f1_macro_vs_winmax"].median()
                    ),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output / "method_summary_vs_winmax.csv", index=False
    )
    discovered_path = output / "discovered_contexts.csv"
    if discovered_path.exists():
        contexts = pd.read_csv(discovered_path)[
            ["dataset_id", "arch", "profile_name"]
        ].drop_duplicates()
    else:
        contexts = frame[["dataset_id", "arch", "profile_name"]].drop_duplicates()
    coverage_rows = []
    expected_methods = ["winmax_reference"] + ALL_EXTERNAL_METHODS
    for method in expected_methods:
        observed = frame[frame["method_id"] == method][
            ["dataset_id", "arch", "profile_name"]
        ].drop_duplicates()
        coverage_rows.append(
            {
                "method_id": method,
                "completed_contexts": len(observed),
                "discovered_contexts": len(contexts),
                "coverage_fraction": len(observed) / max(len(contexts), 1),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(output / "coverage_audit.csv", index=False)
    return frame


def collect_input_audits(output: Path) -> pd.DataFrame:
    rows = []
    for path in output.glob("profiles/*/*/*/input_audit.json"):
        value = read_json(path)
        if isinstance(value, dict):
            value["input_audit_path"] = str(path.resolve())
            for key in ("view_names", "view_dims", "classes", "split_class_counts"):
                if key in value and not isinstance(value[key], str):
                    value[key] = json.dumps(value[key], ensure_ascii=False, sort_keys=True)
            rows.append(value)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["dataset_id", "arch", "profile_name"]
        ).reset_index(drop=True)
        frame.to_csv(output / "input_audit.csv", index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw literature multi-view competitors versus WinMax"
    )
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run", default="colab_seed42")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--arches", default="")
    parser.add_argument(
        "--methods",
        default="all",
        help="all, controls, core, heavy, or comma-separated method IDs",
    )
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--print-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one concise validation line as each setting is evaluated.",
    )
    parser.add_argument(
        "--force-methods",
        default="",
        help="Comma-separated completed external methods to recompute without recomputing WinMax.",
    )
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument(
        "--fail-fast",
        action="store_false",
        dest="continue_on_error",
        help="Stop immediately on the first failed context/method.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--c-grid", type=parse_csv_floats, default=[0.1, 1.0, 10.0])
    parser.add_argument("--k-grid", type=parse_csv_ints, default=[1, 3, 5, 7, 11])
    parser.add_argument("--pca-dims", type=parse_csv_ints, default=[64, 128, 256])
    parser.add_argument("--view-pca-dims", type=parse_csv_ints, default=[32, 64])
    parser.add_argument("--latent-dims", type=parse_csv_ints, default=[16, 32])
    parser.add_argument(
        "--projection-regularization",
        type=parse_csv_floats,
        default=[0.001, 0.01],
    )
    parser.add_argument("--gmlda-alpha", type=parse_csv_floats, default=[1.0, 10.0])
    parser.add_argument(
        "--head2toe-lambdas", type=parse_csv_floats, default=[1e-5, 1e-4]
    )
    parser.add_argument(
        "--head2toe-keep-fractions",
        type=parse_csv_floats,
        default=[0.01, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--head2toe-steps", type=int, default=500)
    parser.add_argument("--head2toe-batch-size", type=int, default=256)
    parser.add_argument("--head2toe-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--easymkl-lambdas", type=parse_csv_floats, default=[0.1, 0.5, 0.9]
    )
    parser.add_argument("--easymkl-max-train", type=int, default=1000)
    parser.add_argument("--nca-dims", type=parse_csv_ints, default=[16, 32, 64])
    parser.add_argument("--nca-max-fit-samples", type=int, default=2500)
    parser.add_argument("--nca-max-iter", type=int, default=50)
    parser.add_argument("--winmax-max-views", type=int, default=4)
    parser.add_argument(
        "--winmax-weight-powers",
        type=parse_csv_floats,
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--winmax-dirichlet-trials", type=int, default=8)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output = args.output.resolve()
    args.embedding_root = args.embedding_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = resolve_methods(args.methods)
    force_methods = set(parse_csv_strings(args.force_methods))
    unknown_force = force_methods - set(METHOD_FUNCTIONS)
    if unknown_force:
        raise ValueError(f"Unknown --force-methods values: {sorted(unknown_force)}")
    run_config = {
        key: value
        for key, value in vars(args).items()
    }
    run_config["methods_resolved"] = methods
    run_config["force_methods_resolved"] = sorted(force_methods)
    run_config["protocol"] = {
        "fit": "train only",
        "selection": "validation only",
        "test": "once after freezing selected candidate",
        "external_data": False,
        "alternative_backbones": False,
        "winmax_is_not_available_to_competitor_fit": True,
    }
    history_dir = args.output / "run_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / (
        time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}.json"
    )
    atomic_json(run_config, history_path)
    atomic_json(run_config, args.output / "last_run_config.json")
    canonical_config = args.output / "run_config.json"
    if not canonical_config.exists():
        atomic_json(run_config, canonical_config)

    profiles = discover_embedding_profiles([args.embedding_root], expected_dims=[])
    profiles = [p for p in profiles if p.usable and p.run_id == args.source_run]
    dataset_filter = set(parse_csv_strings(args.datasets))
    arch_filter = set(parse_csv_strings(args.arches))
    selected_profiles = []
    for profile in profiles:
        metadata = profile_metadata(profile)
        arch = str(metadata.get("arch") or profile.profile_name.split("_imagenet")[0])
        if dataset_filter and profile.dataset_id not in dataset_filter:
            continue
        if arch_filter and arch not in arch_filter:
            continue
        selected_profiles.append((profile, arch))
    selected_profiles.sort(key=lambda item: (item[0].dataset_id, item[1]))
    if args.max_profiles > 0:
        selected_profiles = selected_profiles[: args.max_profiles]
    registry_path = args.output / "input_profile_registry.csv"
    existing_registry_size = (
        len(pd.read_csv(registry_path)) if registry_path.exists() else 0
    )
    if len(selected_profiles) >= existing_registry_size:
        write_profile_registry([p for p, _ in selected_profiles], args.output)
    discovered_now = pd.DataFrame(
        [
            {
                "dataset_id": profile.dataset_id,
                "run_id": profile.run_id,
                "arch": arch,
                "profile_name": profile.profile_name,
                "profile_path": str(profile.path.resolve()),
            }
            for profile, arch in selected_profiles
        ]
    )
    discovered_path = args.output / "discovered_contexts.csv"
    if discovered_path.exists():
        discovered_before = pd.read_csv(discovered_path)
        discovered_now = pd.concat(
            [discovered_before, discovered_now], ignore_index=True
        ).drop_duplicates(["dataset_id", "arch", "profile_name"], keep="last")
    discovered_now.sort_values(
        ["dataset_id", "arch", "profile_name"]
    ).to_csv(discovered_path, index=False)
    print(
        f"Discovered {len(selected_profiles)} usable contexts from {args.source_run}. "
        f"External methods: {', '.join(methods)}",
        flush=True,
    )

    audit_rows = []
    for context_index, (profile, arch) in enumerate(selected_profiles, start=1):
        print(
            f"\n[{context_index}/{len(selected_profiles)}] "
            f"{profile.dataset_id} | {arch}",
            flush=True,
        )
        try:
            data = load_multiview_data(profile)
            audit = audit_profile(profile, data)
            audit["arch"] = arch
            audit_rows.append(audit)
            profile_dir = profile_output_dir(args.output, profile, arch)
            atomic_json(audit, profile_dir / "input_audit.json")
            collect_input_audits(args.output)
        except Exception as exc:
            write_error(args.output, profile, arch, "input_audit", exc)
            print(f"  INPUT ERROR: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue

        context = {
            "dataset_id": profile.dataset_id,
            "run_id": profile.run_id,
            "arch": arch,
            "profile_name": profile.profile_name,
            "profile_path": str(profile.path.resolve()),
            "n_views": len(data.view_dims),
            "view_names": "|".join(data.view_names),
            "view_dims": "|".join(map(str, data.view_dims)),
            "n_train": len(data.y_train),
            "n_val": len(data.y_val),
            "n_test": len(data.y_test),
            "n_classes": len(data.classes),
            "seed": args.seed,
        }

        winmax_path = selected_result_path(profile_dir, "winmax_reference")
        winmax_result = (
            None
            if args.force
            else load_completed_result(winmax_path.parent)
        )
        if winmax_result is None:
            try:
                print("  Running Proposed method (WinMax) reference...", flush=True)
                winmax_reporter = make_candidate_reporter(
                    "winmax_reference",
                    winmax_path.parent,
                    args.print_candidates,
                )
                output = run_winmax_reference(
                    data, args, candidate_reporter=winmax_reporter
                )
                winmax_result = save_method_output(
                    output,
                    data,
                    winmax_path.parent,
                    context,
                    None,
                )
            except Exception as exc:
                write_error(args.output, profile, arch, "winmax_reference", exc)
                print(f"  WINMAX ERROR: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
                continue
        print(
            f"  WinMax test macro-F1={winmax_result['test_f1_macro']:.6f} | "
            f"accuracy={winmax_result['test_accuracy']:.6f}",
            flush=True,
        )

        for method_id in methods:
            result_path = selected_result_path(profile_dir, method_id)
            existing = (
                None
                if args.force or method_id in force_methods
                else load_completed_result(result_path.parent)
            )
            if existing is not None:
                print(
                    f"  {existing['display_name']}: cached test macro-F1="
                    f"{existing['test_f1_macro']:.6f} | "
                    f"delta vs WinMax={existing['delta_test_f1_macro_vs_winmax']:+.6f}",
                    flush=True,
                )
                continue
            try:
                print(f"  Running {method_id}...", flush=True)
                function = METHOD_FUNCTIONS[method_id]
                reporter = make_candidate_reporter(
                    method_id,
                    result_path.parent,
                    args.print_candidates,
                )
                set_candidate_reporter(reporter)
                try:
                    output = function(data, **method_kwargs(method_id, args))
                finally:
                    set_candidate_reporter(None)
                result = save_method_output(
                    output,
                    data,
                    result_path.parent,
                    context,
                    winmax_result,
                )
                print(
                    f"  {result['display_name']}: test macro-F1="
                    f"{result['test_f1_macro']:.6f} | "
                    f"WinMax={result['winmax_test_f1_macro']:.6f} | "
                    f"delta={result['delta_test_f1_macro_vs_winmax']:+.6f}",
                    flush=True,
                )
            except Exception as exc:
                write_error(args.output, profile, arch, method_id, exc)
                print(f"  {method_id} ERROR: {type(exc).__name__}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
        collect_results(args.output)

    collect_input_audits(args.output)
    final = collect_results(args.output)
    print(
        f"\nCompleted. Selected rows: {len(final)}. "
        f"Results: {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
