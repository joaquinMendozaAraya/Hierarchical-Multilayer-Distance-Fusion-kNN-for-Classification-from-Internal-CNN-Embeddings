#!/usr/bin/env python
"""Build harmonized result matrices for the paper.

This script consolidates the already-computed experiments into one set of
tables. It does not train models and it does not re-select hyperparameters.

Inputs:
- 61: fair matrix with aggregate validation/test metrics.
- 78: raw literature multidimensional competitors and WinMax, with NPZ
  prediction/probability artifacts.
- 79: post-hoc prototype competitors, with NPZ prediction/probability
  artifacts.
- Colab softmax baselines, with CSV predictions and NPY probabilities.

Outputs are saved under experiments/80_paper_result_matrices by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXP61 = ROOT / "experiments" / "61_final_paper_fair_matrix"
EXP78 = ROOT / "experiments" / "78_raw_literature_multidim_competitors"
EXP79 = ROOT / "experiments" / "79_posthoc_prototype_competitors"
SOFTMAX_ROOT = ROOT / "results" / "baselines_from_colab" / "results"
DEFAULT_OUTPUT = ROOT / "experiments" / "80_paper_result_matrices"


METRIC_NAMES = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "mcc",
    "auroc_ovr_macro",
    "auprc_ovr_macro",
    "auprc_ovr_weighted",
    "auprc_ovr_micro",
    "log_loss",
    "brier_multiclass",
    "mean_confidence",
    "ece_15",
]


DATASET_LABELS = {
    "brain_tumor_mri_4c": "Brain tumor MRI 4C",
    "brain_tumor_mri_14c": "Brain tumor MRI 14/15C",
    "brain_tumor_mri_17c": "Brain tumor MRI 17C",
    "brain_tumor_mri_44c": "Brain tumor MRI 44C",
    "sciencedb_brain_tumor_3c": "ScienceDB brain tumor 3C",
    "ham10000_skin_7c": "HAM10000 skin 7C control",
}


ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b3": "EfficientNet-B3",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


METHOD_LABELS = {
    "softmax_full_finetuned": "Softmax head",
    "final_embedding_classical": "Final embedding classifier",
    "final_embedding_gmm": "Final embedding GMM",
    "final_feature_knn": "Final embedding kNN/PCA",
    "concat_nca_knn": "NCA + kNN",
    "concat_pca_linear": "Concat + PCA + linear",
    "raw_concat_linear": "Raw concat + linear",
    "uniform_layer_softvote": "Uniform layer soft vote",
    "uniform_kernel_svm": "Uniform kernel SVM",
    "fradi_mlcff": "MLCFF-style fusion",
    "gmlda": "GMLDA",
    "head2toe": "Head2Toe-style fusion",
    "maxvar_gcca": "MAXVAR-GCCA",
    "mvda": "MvDA",
    "winmax_reference": "Proposed method",
    "kmex_final_embedding": "KMEx final embedding",
    "posthoc_self_explanation_b4": "B4 post-hoc prototype",
    "posthoc_self_explanation_b234": "B234 post-hoc prototype",
}


LITERATURE_METHOD_SOURCES = {
    "concat_nca_knn": "Neighborhood Components Analysis / metric learning baseline",
    "fradi_mlcff": "Fradi et al. MLCFF-style multilayer feature fusion",
    "head2toe": "Head2Toe-style deep feature reuse",
    "easymkl": "EasyMKL multiple-kernel learning",
    "maxvar_gcca": "MAXVAR-GCCA multi-view projection",
    "gmlda": "Generalized multi-view LDA / GMLDA",
    "mvda": "Multi-view discriminant analysis",
    "kmex_final_embedding": "Gautam et al. KMEx final-embedding prototypes",
    "posthoc_self_explanation_b4": "Boubekki and Clemmensen B4 prototype rule",
    "posthoc_self_explanation_b234": "Boubekki and Clemmensen B234 prototype rule",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp61", type=Path, default=EXP61)
    parser.add_argument("--exp78", type=Path, default=EXP78)
    parser.add_argument("--exp79", type=Path, default=EXP79)
    parser.add_argument("--softmax-root", type=Path, default=SOFTMAX_ROOT)
    parser.add_argument(
        "--include-aggregate-61-candidates",
        action="store_true",
        help="Also build a harmonized candidate table from experiment 61.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_arch(profile_name: str, arch: Any = None) -> str:
    if isinstance(arch, str) and arch and arch != "nan":
        return arch
    text = str(profile_name)
    known = sorted(ARCH_LABELS, key=len, reverse=True)
    for candidate in known:
        if text.startswith(candidate + "_") or text == candidate:
            return candidate
    if "_imagenet" in text:
        return text.split("_imagenet", 1)[0]
    return ""


def infer_profile_name(row: pd.Series) -> str:
    existing = str(row.get("profile_name", "") or "")
    if existing and existing != "nan":
        return existing
    arch = str(row.get("arch", "") or "")
    image_size = safe_int(row.get("image_size"))
    if arch and image_size:
        return f"{arch}_imagenet_full_img{image_size}_colab_best_gap"
    return arch


def context_id(dataset_id: str, arch: str, profile_name: str = "") -> str:
    profile = profile_name or arch
    return f"{dataset_id}__{arch}__{profile}"


def dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def arch_label(arch: str) -> str:
    return ARCH_LABELS.get(arch, arch)


def method_label(method_id: str, default: str = "") -> str:
    if method_id in METHOD_LABELS:
        return METHOD_LABELS[method_id]
    if default:
        return default
    return str(method_id).replace("_", " ")


def result_group_from_61(method_block: str) -> str:
    mapping = {
        "softmax_output": "softmax",
        "final_embedding_classifiers": "last_layer",
        "multidimensional_competitors": "multilayer",
        "proposed_vchmf_family": "proposed",
        "diagnostic_layer_probes_not_main_baseline": "diagnostic",
    }
    return mapping.get(str(method_block), "unknown")


def paper_comparison_group(result_group: str) -> str:
    if result_group == "proposed":
        return "multilayer"
    return result_group


def result_group_from_78(method_id: str) -> str:
    if method_id == "winmax_reference":
        return "proposed"
    return "multilayer"


def result_group_from_79(method_id: str) -> str:
    if method_id == "kmex_final_embedding":
        return "last_layer"
    if method_id == "posthoc_self_explanation_b4":
        return "last_layer"
    return "multilayer"


def method_origin_from_group(result_group: str, source: str, method_id: str) -> str:
    if method_id == "winmax_reference" or result_group == "proposed":
        return "proposed_method"
    if source == "softmax_colab":
        return "softmax_baseline"
    if source.startswith("exp78"):
        if method_id in {"raw_concat_linear", "concat_pca_linear", "uniform_layer_softvote", "uniform_kernel_svm"}:
            return "multilayer_control"
        return "literature_multilayer_competitor"
    if source.startswith("exp79"):
        return "posthoc_literature_prototype_competitor"
    if result_group == "last_layer":
        return "classic_final_embedding_classifier"
    if result_group == "multilayer":
        return "legacy_or_internal_multilayer_variant"
    if result_group == "diagnostic":
        return "diagnostic_layer_probe"
    return "unknown"


def is_primary_for_paper(source: str, method_id: str, result_group: str) -> bool:
    if source == "softmax_colab":
        return True
    if source == "exp78_selected":
        return True
    if source == "exp79_selected":
        return True
    if source == "exp61_selected" and result_group == "last_layer":
        return True
    return False


def primary_reason(source: str, method_id: str, result_group: str) -> str:
    if is_primary_for_paper(source, method_id, result_group):
        return "included_in_primary_matrix"
    if source == "exp61_selected" and result_group == "softmax":
        return "duplicate_softmax_without_probability_artifacts"
    if source == "exp61_selected" and result_group == "proposed":
        return "legacy_proposed_family_superseded_by_winmax_reference"
    if source == "exp61_selected" and result_group == "multilayer":
        return "legacy_internal_or_early_multilayer_variant"
    if source == "exp61_selected" and result_group == "diagnostic":
        return "diagnostic_not_main_baseline"
    if source == "exp61_candidates":
        return "validation_candidate_appendix_only"
    return "not_marked_primary"


def rankdata_average(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = np.asarray(y_binary).astype(bool)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y_binary = y_binary[mask]
    scores = scores[mask]
    n_pos = int(y_binary.sum())
    n_neg = int((~y_binary).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(scores)
    pos_rank_sum = float(ranks[y_binary].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_average_precision(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = np.asarray(y_binary).astype(bool)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y_binary = y_binary[mask]
    scores = scores[mask]
    total_pos = int(y_binary.sum())
    if total_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_binary[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision_at_rank = tp / ranks
    return float((precision_at_rank * y_sorted).sum() / total_pos)


def multiclass_auc_auprc(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    if proba.ndim != 2 or len(classes) != proba.shape[1]:
        return {
            "auroc_ovr_macro": float("nan"),
            "auprc_ovr_macro": float("nan"),
            "auprc_ovr_weighted": float("nan"),
            "auprc_ovr_micro": float("nan"),
        }

    aucs: list[float] = []
    aps: list[float] = []
    supports: list[int] = []
    y_micro_parts: list[np.ndarray] = []
    score_micro_parts: list[np.ndarray] = []
    for idx, cls in enumerate(classes):
        y_bin = y_true == cls
        supports.append(int(y_bin.sum()))
        aucs.append(binary_auc(y_bin, proba[:, idx]))
        aps.append(binary_average_precision(y_bin, proba[:, idx]))
        y_micro_parts.append(y_bin.astype(int))
        score_micro_parts.append(proba[:, idx])

    auc_array = np.asarray(aucs, dtype=float)
    ap_array = np.asarray(aps, dtype=float)
    support_array = np.asarray(supports, dtype=float)

    valid_ap = np.isfinite(ap_array)
    if valid_ap.any() and support_array[valid_ap].sum() > 0:
        weighted_ap = float(np.average(ap_array[valid_ap], weights=support_array[valid_ap]))
    else:
        weighted_ap = float("nan")

    y_micro = np.concatenate(y_micro_parts) if y_micro_parts else np.array([])
    score_micro = np.concatenate(score_micro_parts) if score_micro_parts else np.array([])
    micro_ap = binary_average_precision(y_micro, score_micro) if len(y_micro) else float("nan")

    return {
        "auroc_ovr_macro": float(np.nanmean(auc_array)) if np.isfinite(auc_array).any() else float("nan"),
        "auprc_ovr_macro": float(np.nanmean(ap_array)) if np.isfinite(ap_array).any() else float("nan"),
        "auprc_ovr_weighted": weighted_ap,
        "auprc_ovr_micro": micro_ap,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes.tolist())}
    matrix = np.zeros((len(classes), len(classes)), dtype=float)
    for true_value, pred_value in zip(y_true, y_pred):
        if true_value in class_to_idx and pred_value in class_to_idx:
            matrix[class_to_idx[true_value], class_to_idx[pred_value]] += 1

    total = float(matrix.sum())
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    # Match sklearn's zero_division=0 behavior for macro metrics. A class that
    # is present in y_true but never predicted must contribute F1=0; excluding
    # it with nanmean inflates macro-F1 on fine-grained datasets.
    active = (support + predicted) > 0
    recall = np.divide(
        tp,
        support,
        out=np.zeros_like(tp, dtype=float),
        where=support > 0,
    )
    precision = np.divide(
        tp,
        predicted,
        out=np.zeros_like(tp, dtype=float),
        where=predicted > 0,
    )
    f1_denominator = 2.0 * tp + (predicted - tp) + (support - tp)
    f1 = np.divide(
        2.0 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=float),
        where=f1_denominator > 0,
    )

    accuracy = float(tp.sum() / total) if total else float("nan")
    support_sum = support.sum()
    f1_weighted = (
        float(np.nansum(f1 * support) / support_sum) if support_sum > 0 else float("nan")
    )

    c = float(np.trace(matrix))
    s = float(matrix.sum())
    p = matrix.sum(axis=0)
    t = matrix.sum(axis=1)
    denom = math.sqrt(max((s * s - float(np.dot(p, p))) * (s * s - float(np.dot(t, t))), 0.0))
    mcc = float((c * s - float(np.dot(p, t))) / denom) if denom > 0 else float("nan")

    result = {
        "accuracy": accuracy,
        "f1_macro": float(np.mean(f1[active])) if active.any() else float("nan"),
        "f1_weighted": f1_weighted,
        "balanced_accuracy": (
            float(np.mean(recall[support > 0]))
            if np.any(support > 0)
            else float("nan")
        ),
        "precision_macro": (
            float(np.mean(precision[active])) if active.any() else float("nan")
        ),
        "recall_macro": (
            float(np.mean(recall[active])) if active.any() else float("nan")
        ),
        "mcc": mcc,
        "auroc_ovr_macro": float("nan"),
        "auprc_ovr_macro": float("nan"),
        "auprc_ovr_weighted": float("nan"),
        "auprc_ovr_micro": float("nan"),
        "log_loss": float("nan"),
        "brier_multiclass": float("nan"),
        "mean_confidence": float("nan"),
        "ece_15": float("nan"),
    }

    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2 and proba.shape[0] == len(y_true):
            if classes is None or len(classes) != proba.shape[1]:
                classes = np.arange(proba.shape[1])
            result.update(multiclass_auc_auprc(y_true, proba, classes))
            eps = 1e-12
            proba_clipped = np.clip(proba, eps, 1.0)
            proba_clipped = proba_clipped / proba_clipped.sum(axis=1, keepdims=True)
            indices = np.array([class_to_idx.get(value, -1) for value in y_true])
            valid = indices >= 0
            if valid.any():
                result["log_loss"] = float(-np.log(proba_clipped[np.where(valid)[0], indices[valid]]).mean())
                one_hot = np.zeros_like(proba_clipped)
                one_hot[np.where(valid)[0], indices[valid]] = 1.0
                result["brier_multiclass"] = float(np.mean(np.sum((proba_clipped - one_hot) ** 2, axis=1)))
            confidence = proba_clipped.max(axis=1)
            correctness = (y_true == y_pred).astype(float)
            result["mean_confidence"] = float(confidence.mean()) if len(confidence) else float("nan")
            result["ece_15"] = expected_calibration_error(confidence, correctness, n_bins=15)
    return result


def expected_calibration_error(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        else:
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if not mask.any():
            continue
        ece += float(mask.sum() / total) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return ece


def load_npz_prediction_metrics(npz_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction_artifact_path": str(npz_path),
        "probability_metrics_available": False,
    }
    if not npz_path.exists():
        result["prediction_artifact_path"] = ""
        return result
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - artifact-dependent
        result["prediction_read_error"] = repr(exc)
        return result
    classes = data["classes"] if "classes" in data.files else None
    for split in ["val", "test"]:
        y_key = f"y_{split}"
        pred_key = f"pred_{split}"
        proba_key = f"proba_{split}"
        if y_key not in data.files or pred_key not in data.files:
            continue
        y_true = data[y_key]
        y_pred = data[pred_key]
        proba = data[proba_key] if proba_key in data.files else None
        metrics = classification_metrics(y_true, y_pred, proba=proba, classes=classes)
        for name, value in metrics.items():
            result[f"{split}_{name}"] = value
        result[f"n_{split}"] = int(len(y_true))
        if is_valid_probability_array(proba, len(y_true)):
            result["probability_metrics_available"] = True
    if classes is not None:
        result["n_classes_from_predictions"] = int(len(classes))
    return result


def load_softmax_prediction_metrics(prediction_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction_artifact_path": str(prediction_dir),
        "probability_metrics_available": False,
    }
    if not prediction_dir.exists():
        result["prediction_artifact_path"] = ""
        return result
    for split in ["val", "test"]:
        pred_csv = prediction_dir / f"{split}_predictions.csv"
        proba_npy = prediction_dir / f"{split}_probs.npy"
        if not pred_csv.exists():
            continue
        pred_df = pd.read_csv(pred_csv)
        if "y_true" not in pred_df or "y_pred" not in pred_df:
            continue
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        proba = np.load(proba_npy) if proba_npy.exists() else None
        classes = np.arange(proba.shape[1]) if proba is not None and proba.ndim == 2 else np.unique(np.concatenate([y_true, y_pred]))
        metrics = classification_metrics(y_true, y_pred, proba=proba, classes=classes)
        for name, value in metrics.items():
            result[f"{split}_{name}"] = value
        result[f"n_{split}"] = int(len(y_true))
        if is_valid_probability_array(proba, len(y_true)):
            result["probability_metrics_available"] = True
            result["n_classes_from_predictions"] = int(proba.shape[1])
    return result


def is_valid_probability_array(proba: Any, n_samples: int) -> bool:
    if proba is None:
        return False
    try:
        arr = np.asarray(proba)
    except Exception:
        return False
    return bool(arr.ndim == 2 and arr.shape[0] == n_samples and arr.shape[1] >= 2)


def selected_result_to_npz_path(selected_result_path: Any, method_id: str, exp_root: Path, row: pd.Series) -> Path:
    raw = str(selected_result_path or "")
    if raw and raw != "nan":
        path = Path(raw)
        if not path.is_absolute():
            path = exp_root / path
        if path.name == "predictions.npz":
            return path
        candidate = path.parent / "predictions.npz"
        if candidate.exists():
            return candidate
    dataset_id = str(row.get("dataset_id", ""))
    arch = str(row.get("arch", ""))
    profile_name = str(row.get("profile_name", ""))
    return exp_root / "profiles" / dataset_id / arch / profile_name / "methods" / method_id / "predictions.npz"


def base_row(
    *,
    source_experiment: str,
    result_level: str,
    dataset_id: str,
    arch: str,
    profile_name: str,
    method_id: str,
    method_display: str,
    result_group: str,
    row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    comparison_group = paper_comparison_group(result_group)
    origin = method_origin_from_group(result_group, source_experiment, method_id)
    out = {
        "source_experiment": source_experiment,
        "result_level": result_level,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label(dataset_id),
        "is_brain_tumor_dataset": dataset_id != "ham10000_skin_7c",
        "is_external_domain_control": dataset_id == "ham10000_skin_7c",
        "arch": arch,
        "arch_label": arch_label(arch),
        "profile_name": profile_name,
        "context_id": context_id(dataset_id, arch, profile_name),
        "context_short": f"{dataset_id} | {arch}",
        "method_id": method_id,
        "method_display": method_display,
        "method_matrix_label": f"{comparison_group} | {method_display} | {source_experiment}",
        "result_group": result_group,
        "paper_comparison_group": comparison_group,
        "method_origin": origin,
        "literature_source_hint": LITERATURE_METHOD_SOURCES.get(method_id, ""),
        "is_primary_for_paper": is_primary_for_paper(source_experiment, method_id, result_group),
        "primary_reason": primary_reason(source_experiment, method_id, result_group),
        "selection_protocol": "fit_train_select_validation_evaluate_test_once",
        "uses_test_for_selection": safe_bool(get_value(row, "uses_test_for_selection", False)),
        "uses_external_data": safe_bool(get_value(row, "uses_external_data", False)),
        "uses_alternative_backbone": safe_bool(get_value(row, "uses_alternative_backbone", False)),
        "probability_metrics_available": False,
        "prediction_artifact_path": "",
    }
    passthrough = [
        "run_id",
        "method_block",
        "method_family",
        "representation",
        "dims",
        "input_dim",
        "pca_dim",
        "classifier",
        "params",
        "pca_dim_effective",
        "pca_explained_variance",
        "standardize",
        "n_views",
        "view_names",
        "view_dims",
        "n_train",
        "n_val",
        "n_test",
        "n_classes",
        "seed",
        "fidelity",
        "source",
        "n_validation_candidates",
        "test_evaluations_after_selection",
        "selected_result_path",
        "selected_config_json",
        "adaptations_json",
        "checkpoint_path",
        "image_size",
        "smoke_only",
    ]
    for key in passthrough:
        out[key] = get_value(row, key, "")
    return out


def get_value(row: pd.Series | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row[key] if key in row.index else default
    return row.get(key, default)


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "si", "sí"}


def copy_metric_columns(out: dict[str, Any], row: pd.Series | dict[str, Any]) -> None:
    aliases = {
        "val_f1_macro": ["val_f1_macro", "val_macro_f1"],
        "test_f1_macro": ["test_f1_macro", "test_macro_f1"],
    }
    for split in ["val", "test"]:
        for name in METRIC_NAMES:
            column = f"{split}_{name}"
            if column in aliases:
                value = first_existing(row, aliases[column])
            else:
                value = get_value(row, column, float("nan"))
            out[column] = safe_float(value)
    # Common aggregate aliases from Colab.
    if not np.isfinite(out.get("val_f1_macro", float("nan"))):
        out["val_f1_macro"] = safe_float(get_value(row, "val_macro_f1", float("nan")))
    if not np.isfinite(out.get("test_f1_macro", float("nan"))):
        out["test_f1_macro"] = safe_float(get_value(row, "test_macro_f1", float("nan")))


def first_existing(row: pd.Series | dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = get_value(row, name, None)
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except TypeError:
                pass
            return value
    return float("nan")


def merge_prediction_metrics(out: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if key in {"prediction_artifact_path", "probability_metrics_available", "prediction_read_error", "n_classes_from_predictions"}:
            out[key] = value
            continue
        if key.startswith(("val_", "test_", "n_")):
            out[key] = value
    if metrics.get("n_classes_from_predictions"):
        out["n_classes"] = metrics["n_classes_from_predictions"]


def normalize_exp61_selected(exp61: Path) -> pd.DataFrame:
    path = exp61 / "fast_selected_by_profile_and_block.csv"
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        profile_name = str(row.get("profile_name", ""))
        arch = infer_arch(profile_name)
        method_block = str(row.get("method_block", ""))
        result_group = result_group_from_61(method_block)
        method_family = str(row.get("method_family", ""))
        classifier = str(row.get("classifier", ""))
        representation = str(row.get("representation", ""))
        if result_group == "softmax":
            method_id = "softmax_full_finetuned"
        else:
            method_id = method_family if method_family else method_block
        display = method_label(method_id)
        if result_group in {"last_layer", "multilayer", "proposed", "diagnostic"}:
            display = f"{display} ({classifier}, {representation})"
        out = base_row(
            source_experiment="exp61_selected",
            result_level="selected",
            dataset_id=dataset_id,
            arch=arch,
            profile_name=profile_name,
            method_id=method_id,
            method_display=display,
            result_group=result_group,
            row=row,
        )
        copy_metric_columns(out, row)
        out["source_table_path"] = str(path)
        rows.append(out)
    return pd.DataFrame(rows)


def normalize_exp61_candidates(exp61: Path) -> pd.DataFrame:
    path = exp61 / "fast_all_candidate_results.csv"
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        profile_name = str(row.get("profile_name", ""))
        arch = infer_arch(profile_name)
        method_block = str(row.get("method_block", ""))
        result_group = result_group_from_61(method_block)
        method_family = str(row.get("method_family", ""))
        classifier = str(row.get("classifier", ""))
        representation = str(row.get("representation", ""))
        if result_group == "softmax":
            method_id = "softmax_full_finetuned"
        else:
            method_id = f"{method_family}__{classifier}__{representation}"
        display = method_label(method_family, method_family)
        display = f"{display} ({classifier}, {representation})"
        out = base_row(
            source_experiment="exp61_candidates",
            result_level="validation_candidate",
            dataset_id=dataset_id,
            arch=arch,
            profile_name=profile_name,
            method_id=method_id,
            method_display=display,
            result_group=result_group,
            row=row,
        )
        copy_metric_columns(out, row)
        out["source_table_path"] = str(path)
        rows.append(out)
    return pd.DataFrame(rows)


def normalize_exp78_selected(exp78: Path) -> pd.DataFrame:
    path = exp78 / "all_selected_results.csv"
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        arch = str(row.get("arch", ""))
        profile_name = str(row.get("profile_name", ""))
        method_id = str(row.get("method_id", ""))
        result_group = result_group_from_78(method_id)
        display = str(row.get("display_name", "") or method_label(method_id))
        if method_id == "winmax_reference":
            display = "Proposed method"
        out = base_row(
            source_experiment="exp78_selected",
            result_level="selected",
            dataset_id=dataset_id,
            arch=arch,
            profile_name=profile_name,
            method_id=method_id,
            method_display=display,
            result_group=result_group,
            row=row,
        )
        copy_metric_columns(out, row)
        npz_path = selected_result_to_npz_path(row.get("selected_result_path", ""), method_id, exp78, row)
        merge_prediction_metrics(out, load_npz_prediction_metrics(npz_path))
        out["source_table_path"] = str(path)
        rows.append(out)
    return pd.DataFrame(rows)


def normalize_exp79_selected(exp79: Path) -> pd.DataFrame:
    path = exp79 / "all_selected_results.csv"
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        arch = str(row.get("arch", ""))
        profile_name = str(row.get("profile_name", ""))
        method_id = str(row.get("method_id", ""))
        result_group = result_group_from_79(method_id)
        display = str(row.get("display_name", "") or method_label(method_id))
        out = base_row(
            source_experiment="exp79_selected",
            result_level="selected",
            dataset_id=dataset_id,
            arch=arch,
            profile_name=profile_name,
            method_id=method_id,
            method_display=display,
            result_group=result_group,
            row=row,
        )
        copy_metric_columns(out, row)
        npz_path = selected_result_to_npz_path(row.get("selected_result_path", ""), method_id, exp79, row)
        merge_prediction_metrics(out, load_npz_prediction_metrics(npz_path))
        out["source_table_path"] = str(path)
        rows.append(out)
    return pd.DataFrame(rows)


def normalize_softmax(softmax_root: Path) -> pd.DataFrame:
    path = softmax_root / "ALL_FINETUNE_RESULTS.csv"
    df = read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        arch = str(row.get("arch", ""))
        profile_name = infer_profile_name(row)
        job_id = str(row.get("job_id", ""))
        out = base_row(
            source_experiment="softmax_colab",
            result_level="selected",
            dataset_id=dataset_id,
            arch=arch,
            profile_name=profile_name,
            method_id="softmax_full_finetuned",
            method_display="Softmax head",
            result_group="softmax",
            row=row,
        )
        out["run_id"] = job_id
        out["job_id"] = job_id
        out["freeze_mode"] = row.get("freeze_mode", "")
        out["weights"] = row.get("weights", "")
        out["best_epoch"] = row.get("best_epoch", "")
        copy_metric_columns(out, row)
        pred_dir = softmax_root / dataset_id / arch / job_id / "predictions"
        merge_prediction_metrics(out, load_softmax_prediction_metrics(pred_dir))
        out["source_table_path"] = str(path)
        rows.append(out)
    return pd.DataFrame(rows)


def metric_long(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "source_experiment",
        "result_level",
        "dataset_id",
        "dataset_label",
        "is_brain_tumor_dataset",
        "is_external_domain_control",
        "arch",
        "arch_label",
        "profile_name",
        "context_id",
        "context_short",
        "method_id",
        "method_display",
        "method_matrix_label",
        "result_group",
        "paper_comparison_group",
        "method_origin",
        "is_primary_for_paper",
        "primary_reason",
        "probability_metrics_available",
        "prediction_artifact_path",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        base = {column: row.get(column, "") for column in id_columns}
        for split in ["val", "test"]:
            for metric in METRIC_NAMES:
                value = row.get(f"{split}_{metric}", np.nan)
                if pd.isna(value):
                    continue
                out = dict(base)
                out["split"] = split
                out["metric"] = metric
                out["value"] = float(value)
                rows.append(out)
    return pd.DataFrame(rows)


def write_group_tables(df: pd.DataFrame, output: Path) -> None:
    selected_dir = output / "selected_by_group"
    primary_dir = output / "primary_selected_by_group"
    selected_dir.mkdir(parents=True, exist_ok=True)
    primary_dir.mkdir(parents=True, exist_ok=True)
    for group, subset in df.groupby("result_group", dropna=False):
        safe_group = str(group).replace("/", "_")
        subset.to_csv(selected_dir / f"{safe_group}_selected.csv", index=False)
        primary_subset = subset[subset["is_primary_for_paper"].astype(bool)]
        primary_subset.to_csv(primary_dir / f"{safe_group}_primary_selected.csv", index=False)


def write_metric_matrices(df: pd.DataFrame, output: Path, prefix: str) -> None:
    matrix_dir = output / "matrices" / prefix
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for split in ["val", "test"]:
        for metric in METRIC_NAMES:
            column = f"{split}_{metric}"
            if column not in df.columns or df[column].notna().sum() == 0:
                continue
            pivot = df.pivot_table(
                index=["dataset_id", "arch", "profile_name", "context_id"],
                columns="method_matrix_label",
                values=column,
                aggfunc="max",
            ).reset_index()
            pivot.to_csv(matrix_dir / f"{split}_{metric}_by_context_method.csv", index=False)

    for group, subset in df.groupby("result_group", dropna=False):
        group_dir = matrix_dir / "by_group" / str(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        for split in ["val", "test"]:
            for metric in ["accuracy", "f1_macro", "balanced_accuracy", "auroc_ovr_macro", "auprc_ovr_macro"]:
                column = f"{split}_{metric}"
                if column not in subset.columns or subset[column].notna().sum() == 0:
                    continue
                pivot = subset.pivot_table(
                    index=["dataset_id", "arch", "profile_name", "context_id"],
                    columns="method_matrix_label",
                    values=column,
                    aggfunc="max",
                ).reset_index()
                pivot.to_csv(group_dir / f"{split}_{metric}_by_context_method.csv", index=False)


def write_summary_tables(df: pd.DataFrame, output: Path) -> None:
    coverage = (
        df.groupby(
            [
                "source_experiment",
                "result_group",
                "paper_comparison_group",
                "method_origin",
                "method_id",
                "method_display",
                "is_primary_for_paper",
            ],
            dropna=False,
        )
        .agg(
            rows=("context_id", "size"),
            contexts=("context_id", "nunique"),
            datasets=("dataset_id", "nunique"),
            architectures=("arch", "nunique"),
            brain_tumor_contexts=("is_brain_tumor_dataset", "sum"),
            probability_metric_rows=("probability_metrics_available", "sum"),
        )
        .reset_index()
        .sort_values(["is_primary_for_paper", "result_group", "method_id"], ascending=[False, True, True])
    )
    coverage.to_csv(output / "coverage_by_method.csv", index=False)

    availability_rows = []
    for split in ["val", "test"]:
        for metric in METRIC_NAMES:
            column = f"{split}_{metric}"
            if column not in df:
                continue
            availability_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "available_rows": int(df[column].notna().sum()),
                    "primary_available_rows": int(df.loc[df["is_primary_for_paper"].astype(bool), column].notna().sum()),
                    "total_rows": int(len(df)),
                }
            )
    pd.DataFrame(availability_rows).to_csv(output / "metric_availability.csv", index=False)

    primary = df[df["is_primary_for_paper"].astype(bool)].copy()
    if not primary.empty and "test_f1_macro" in primary:
        best_idx = (
            primary.sort_values(["context_id", "test_f1_macro", "test_accuracy"], ascending=[True, False, False])
            .groupby("context_id", dropna=False)
            .head(1)
            .index
        )
        best = primary.loc[best_idx].copy()
        best.to_csv(output / "primary_best_method_by_context_test_f1.csv", index=False)
        win_counts = (
            best.groupby(["result_group", "paper_comparison_group", "method_origin", "method_id", "method_display"], dropna=False)
            .size()
            .reset_index(name="context_wins_by_test_f1_macro")
            .sort_values("context_wins_by_test_f1_macro", ascending=False)
        )
        win_counts.to_csv(output / "primary_win_counts_by_test_f1_macro.csv", index=False)

        brain = primary[primary["is_brain_tumor_dataset"].astype(bool)]
        if not brain.empty:
            best_brain_idx = (
                brain.sort_values(["context_id", "test_f1_macro", "test_accuracy"], ascending=[True, False, False])
                .groupby("context_id", dropna=False)
                .head(1)
                .index
            )
            best_brain = brain.loc[best_brain_idx].copy()
            best_brain.to_csv(output / "primary_best_method_by_context_test_f1_brain_only.csv", index=False)
            (
                best_brain.groupby(["result_group", "paper_comparison_group", "method_origin", "method_id", "method_display"], dropna=False)
                .size()
                .reset_index(name="brain_context_wins_by_test_f1_macro")
                .sort_values("brain_context_wins_by_test_f1_macro", ascending=False)
                .to_csv(output / "primary_win_counts_by_test_f1_macro_brain_only.csv", index=False)
            )

    group_mean = (
        df.groupby(["is_primary_for_paper", "is_brain_tumor_dataset", "result_group", "paper_comparison_group", "method_origin", "method_id", "method_display"], dropna=False)
        .agg(
            contexts=("context_id", "nunique"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_f1_macro=("test_f1_macro", "mean"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_test_auroc_ovr_macro=("test_auroc_ovr_macro", "mean"),
            mean_test_auprc_ovr_macro=("test_auprc_ovr_macro", "mean"),
        )
        .reset_index()
        .sort_values(["is_primary_for_paper", "is_brain_tumor_dataset", "mean_test_f1_macro"], ascending=[False, False, False])
    )
    group_mean.to_csv(output / "summary_means_by_method.csv", index=False)


def write_matrix_pathbook(output: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "matrices").rglob("*.csv")):
        rel = path.relative_to(output)
        parts = rel.parts
        scope = parts[1] if len(parts) > 1 else ""
        group = ""
        if "by_group" in parts:
            idx = parts.index("by_group")
            if idx + 1 < len(parts):
                group = parts[idx + 1]
        name = path.stem
        split = ""
        metric = ""
        suffix = "_by_context_method"
        if name.endswith(suffix):
            core = name[: -len(suffix)]
            if "_" in core:
                split, metric = core.split("_", 1)
        try:
            shape = pd.read_csv(path, nrows=5).shape
            full_rows = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
            columns = shape[1]
        except Exception:
            full_rows = None
            columns = None
        rows.append(
            {
                "scope": scope,
                "group": group,
                "split": split,
                "metric": metric,
                "path": str(path),
                "relative_path": str(rel),
                "rows": full_rows,
                "columns": columns,
            }
        )
    pathbook = output / "matrix_pathbook.csv"
    pd.DataFrame(rows).to_csv(pathbook, index=False)
    return pathbook


def write_readme(output: Path, selected: pd.DataFrame, candidate: pd.DataFrame) -> None:
    readme = f"""# Paper result matrices

Generated by:

`{ROOT / 'scripts' / 'build_result_matrices_for_paper.py'}`

This folder consolidates already-computed outputs. It does not train models and
does not choose hyperparameters from test data.

## Main files

- `selected_methods_harmonized.csv`: all selected rows from softmax, experiment
  61, experiment 78, and experiment 79, with normalized columns.
- `primary_selected_methods_harmonized.csv`: preferred paper matrix. This keeps
  softmax baselines, final-embedding classifiers from 61, raw literature
  competitors plus WinMax from 78, and post-hoc prototype competitors from 79.
- `candidate_methods_harmonized.csv`: validation candidates from experiment 61,
  useful for appendices and sanity checks.
- `metric_long.csv`: long-format metrics, one row per context/method/split/metric.
- `matrices/`: wide matrices by split and metric for direct plotting.
- `coverage_by_method.csv`: coverage by source, group, and method.
- `metric_availability.csv`: documents which metrics are available.
- `matrix_pathbook.csv`: index of every wide matrix CSV, with scope, group,
  split, metric and full path.

## Result groups

- `softmax`: fine-tuned softmax head.
- `last_layer`: classifiers or prototype methods using the final embedding or
  final spatial representation.
- `multilayer`: methods that fuse or compare multiple CNN layers/views.
- `proposed`: current proposed method, labeled as `Proposed method` when coming
  from the WinMax reference in experiment 78. It is also tagged as multilayer in
  `paper_comparison_group`.
- `diagnostic`: layer probes from experiment 61. These are not main baselines.

## Metric provenance

Metrics ending in AUROC/AUPRC, log-loss, Brier, confidence and ECE require
probability artifacts. They are available for softmax, experiment 78, and
experiment 79. Experiment 61 only stores aggregate accuracy/F1/balanced
accuracy/precision/recall; its probability-derived metrics remain blank.

## Current row counts

- selected rows: {len(selected)}
- primary selected rows: {int(selected['is_primary_for_paper'].sum()) if 'is_primary_for_paper' in selected else 0}
- candidate rows: {len(candidate)}

## Important reporting caveats

- HAM10000 is marked as `is_external_domain_control=True` and should not be
  mixed into global brain-tumor averages unless explicitly stated.
- `brain_tumor_mri_14c` is labeled `Brain tumor MRI 14/15C` because the audit
  found 15 classes including `Normal`.
- Legacy/internal variants from experiment 61 are preserved for traceability
  but are not marked as primary paper competitors.
- Test metrics are copied or recomputed after validation selection. Do not use
  test metrics to re-select methods.
"""
    (output / "README_RESULT_MATRICES.md").write_text(readme, encoding="utf-8")


def save_manifest(output: Path, files: list[Path]) -> None:
    manifest = {
        "root": str(ROOT),
        "output": str(output),
        "generated_files": [str(path) for path in files],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrices").mkdir(parents=True, exist_ok=True)

    selected_parts = [
        normalize_softmax(args.softmax_root),
        normalize_exp61_selected(args.exp61),
        normalize_exp78_selected(args.exp78),
        normalize_exp79_selected(args.exp79),
    ]
    selected_parts = [part for part in selected_parts if not part.empty]
    if not selected_parts:
        raise FileNotFoundError("No selected result tables were found.")

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    selected = selected.sort_values(
        [
            "is_primary_for_paper",
            "dataset_id",
            "arch",
            "result_group",
            "source_experiment",
            "method_display",
        ],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)

    candidate = normalize_exp61_candidates(args.exp61)
    if candidate.empty:
        candidate = pd.DataFrame()

    primary = selected[selected["is_primary_for_paper"].astype(bool)].copy()
    long = metric_long(selected)
    primary_long = metric_long(primary)

    selected_path = output / "selected_methods_harmonized.csv"
    primary_path = output / "primary_selected_methods_harmonized.csv"
    candidate_path = output / "candidate_methods_harmonized.csv"
    long_path = output / "metric_long.csv"
    primary_long_path = output / "primary_metric_long.csv"

    selected.to_csv(selected_path, index=False)
    primary.to_csv(primary_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    long.to_csv(long_path, index=False)
    primary_long.to_csv(primary_long_path, index=False)

    write_group_tables(selected, output)
    write_metric_matrices(selected, output, prefix="all_selected")
    write_metric_matrices(primary, output, prefix="primary_selected")
    matrix_pathbook_path = write_matrix_pathbook(output)
    write_summary_tables(selected, output)
    write_readme(output, selected, candidate)

    generated = [
        selected_path,
        primary_path,
        candidate_path,
        long_path,
        primary_long_path,
        output / "coverage_by_method.csv",
        output / "metric_availability.csv",
        output / "summary_means_by_method.csv",
        matrix_pathbook_path,
        output / "README_RESULT_MATRICES.md",
    ]
    save_manifest(output, generated)

    print("Result matrix pack written to:", output)
    print("Selected rows:", len(selected))
    print("Primary selected rows:", len(primary))
    print("Candidate rows:", len(candidate))
    print("Metric-long rows:", len(long))
    print("Probability rows:", int(selected["probability_metrics_available"].fillna(False).astype(bool).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
