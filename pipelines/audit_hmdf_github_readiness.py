#!/usr/bin/env python
"""Audit HMDF-kNN paper claims and frozen experimental artifacts.

This script is intentionally CPU-only. It does not train models, change
experiment outputs, or select configurations using test data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "publication"
DEFAULT_PAPER = (
    ROOT
    / "audits"
    / "github_preparation_2026_06_12"
    / "manuscript_from_zip"
)
DEFAULT_OUTPUT = ROOT / "audits" / "github_preparation_2026_06_12"
DEFAULT_FAIR_MATRIX = (
    ROOT
    / "experiments"
    / "61_final_paper_fair_matrix"
    / "fast_all_candidate_results.csv"
)

EXPECTED_BRAIN_DATASETS = {
    "brain_tumor_mri_14c",
    "brain_tumor_mri_17c",
    "brain_tumor_mri_44c",
    "brain_tumor_mri_4c",
    "sciencedb_brain_tumor_3c",
}
EXPECTED_BACKBONES = {
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "convnext_tiny",
}
EXPECTED_K = {1, 3, 5, 7, 11}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fair-matrix", type=Path, default=DEFAULT_FAIR_MATRIX)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def close(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return bool(math.isfinite(actual) and abs(actual - expected) <= tolerance)


def add_claim(
    rows: list[dict[str, Any]],
    claim: str,
    reported: Any,
    found: Any,
    source: str,
    tolerance: float | None = None,
    comment: str = "",
) -> None:
    if reported is None:
        status = "verified" if bool(found) else "missing"
    elif isinstance(reported, (int, float)) and isinstance(found, (int, float)):
        limit = 5e-7 if tolerance is None else tolerance
        status = "verified" if close(float(found), float(reported), limit) else "mismatch"
    else:
        status = "verified" if str(found) == str(reported) else "mismatch"
    rows.append(
        {
            "claim": claim,
            "reported_value": reported,
            "found_value": found,
            "source": source,
            "status": status,
            "comment": comment,
        }
    )


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if "weights" in normalized:
        normalized["weights"] = [round(float(value), 7) for value in normalized["weights"]]
    for key in ("layer_indices", "dims"):
        if key in normalized:
            normalized[key] = [int(value) for value in normalized[key]]
    if "k" in normalized:
        normalized["k"] = int(normalized["k"])
    return normalized


def config_from_candidate(row: pd.Series) -> dict[str, Any]:
    return json.loads(str(row["config_json"]))


def selected_candidate(candidates: pd.DataFrame) -> pd.Series:
    frame = candidates.copy()
    for column in [
        "val_f1_macro",
        "val_balanced_accuracy",
        "val_accuracy",
        "final_dim",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(
        ["val_f1_macro", "val_balanced_accuracy", "val_accuracy", "final_dim"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return frame.iloc[0]


def audit_hmdf_contexts(master: pd.DataFrame) -> pd.DataFrame:
    proposed = master[
        as_bool(master["is_brain_tumor_dataset"])
        & master["method_id"].eq("proposed_method")
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, source_row in proposed.sort_values(["dataset_id", "backbone"]).iterrows():
        selected_path = Path(str(source_row["selected_result_path"]))
        prediction_path = Path(str(source_row["prediction_artifact_path"]))
        method_dir = selected_path.parent
        candidate_path = method_dir / "validation_candidates.csv"
        record: dict[str, Any] = {
            "dataset_id": source_row["dataset_id"],
            "backbone": source_row["backbone"],
            "selected_result_path": str(selected_path),
            "prediction_artifact_path": str(prediction_path),
            "selected_result_exists": selected_path.exists(),
            "prediction_artifact_exists": prediction_path.exists(),
            "validation_candidates_exists": candidate_path.exists(),
        }
        if not (selected_path.exists() and prediction_path.exists() and candidate_path.exists()):
            rows.append(record)
            continue

        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        config = selected["selected_config"]
        candidates = pd.read_csv(candidate_path)
        best = selected_candidate(candidates)
        best_config = config_from_candidate(best)
        payload = np.load(prediction_path, allow_pickle=True)
        y_test = np.asarray(payload["y_test"])
        pred_test = np.asarray(payload["pred_test"])
        accuracy = float(accuracy_score(y_test, pred_test))
        macro_f1 = float(f1_score(y_test, pred_test, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(y_test, pred_test))
        weights = np.asarray(config["weights"], dtype=float)
        test_columns = [column for column in candidates.columns if column.startswith("test_")]

        record.update(
            {
                "n_validation_candidates": len(candidates),
                "candidate_table_has_test_columns": bool(test_columns),
                "candidate_test_columns": "|".join(test_columns),
                "selected_config_matches_validation_argmax": (
                    normalize_config(config) == normalize_config(best_config)
                ),
                "uses_test_for_selection": bool(selected.get("uses_test_for_selection")),
                "test_evaluations_after_selection": selected.get(
                    "test_evaluations_after_selection"
                ),
                "selected_n_layers": len(config["layer_indices"]),
                "selected_k": int(config["k"]),
                "weights_nonnegative": bool(np.all(weights >= 0)),
                "weights_sum": float(weights.sum()),
                "weights_sum_to_one": close(float(weights.sum()), 1.0, 2e-6),
                "recomputed_test_accuracy": accuracy,
                "stored_test_accuracy": float(selected["test_accuracy"]),
                "master_test_accuracy": float(source_row["test_accuracy"]),
                "accuracy_matches": (
                    close(accuracy, float(selected["test_accuracy"]))
                    and close(accuracy, float(source_row["test_accuracy"]))
                ),
                "recomputed_test_f1_macro": macro_f1,
                "stored_test_f1_macro": float(selected["test_f1_macro"]),
                "master_test_f1_macro": float(source_row["test_f1_macro"]),
                "macro_f1_matches": (
                    close(macro_f1, float(selected["test_f1_macro"]))
                    and close(macro_f1, float(source_row["test_f1_macro"]))
                ),
                "recomputed_test_balanced_accuracy": balanced,
                "stored_test_balanced_accuracy": float(
                    selected["test_balanced_accuracy"]
                ),
                "master_test_balanced_accuracy": float(
                    source_row["test_balanced_accuracy"]
                ),
                "balanced_accuracy_matches": (
                    close(balanced, float(selected["test_balanced_accuracy"]))
                    and close(balanced, float(source_row["test_balanced_accuracy"]))
                ),
                "test_sample_count": len(y_test),
                "expected_test_sample_count": int(source_row["n_test"]),
                "test_sample_count_matches": len(y_test) == int(source_row["n_test"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def audit_method_artifacts(master: pd.DataFrame) -> pd.DataFrame:
    brain = master[
        as_bool(master["is_brain_tumor_dataset"])
        & as_bool(master["complete_54_context_method"])
    ].copy()
    rows: list[dict[str, Any]] = []
    for (method_id, method_label), group in brain.groupby(
        ["method_id", "method_label"], dropna=False
    ):
        selected_paths = group["selected_result_path"].dropna().map(Path)
        prediction_paths = group["prediction_artifact_path"].dropna().map(Path)
        candidate_paths = selected_paths.map(
            lambda path: path.parent / "validation_candidates.csv"
        )
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "brain_context_rows": len(group),
                "unique_dataset_backbone_contexts": len(
                    group[["dataset_id", "backbone"]].drop_duplicates()
                ),
                "selected_result_paths_recorded": len(selected_paths),
                "selected_result_files_existing": sum(
                    path.exists() for path in selected_paths
                ),
                "prediction_paths_recorded": len(prediction_paths),
                "prediction_files_existing": sum(
                    path.exists() for path in prediction_paths
                ),
                "validation_candidate_files_existing": sum(
                    path.exists() for path in candidate_paths
                ),
                "uses_test_for_selection_true_rows": int(
                    as_bool(group["uses_test_for_selection"]).sum()
                ),
                "mean_test_accuracy": pd.to_numeric(
                    group["test_accuracy"], errors="coerce"
                ).mean(),
                "mean_test_f1_macro": pd.to_numeric(
                    group["test_f1_macro"], errors="coerce"
                ).mean(),
                "mean_test_balanced_accuracy": pd.to_numeric(
                    group["test_balanced_accuracy"], errors="coerce"
                ).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["method_id"]).reset_index(drop=True)


def load_prediction_labels(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if path.is_dir():
        csv_path = path / "test_predictions.csv"
        if not csv_path.exists():
            return None
        frame = pd.read_csv(csv_path)
        if not {"y_true", "y_pred"}.issubset(frame.columns):
            return None
        return frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    if not path.exists() or path.suffix.lower() != ".npz":
        return None
    payload = np.load(path, allow_pickle=True)
    if "y_test" not in payload.files or "pred_test" not in payload.files:
        return None
    return np.asarray(payload["y_test"]), np.asarray(payload["pred_test"])


def audit_method_predictions(master: pd.DataFrame) -> pd.DataFrame:
    brain = master[
        as_bool(master["is_brain_tumor_dataset"])
        & as_bool(master["complete_54_context_method"])
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, source_row in brain.sort_values(
        ["method_id", "dataset_id", "backbone"]
    ).iterrows():
        value = source_row.get("prediction_artifact_path")
        path = None if pd.isna(value) else Path(str(value))
        loaded = None if path is None else load_prediction_labels(path)
        record: dict[str, Any] = {
            "method_id": source_row["method_id"],
            "method_label": source_row["method_label"],
            "dataset_id": source_row["dataset_id"],
            "backbone": source_row["backbone"],
            "prediction_artifact_path": "" if path is None else str(path),
            "prediction_available": loaded is not None,
        }
        if loaded is None:
            rows.append(record)
            continue
        y_test, pred_test = loaded
        accuracy = float(accuracy_score(y_test, pred_test))
        macro_f1 = float(f1_score(y_test, pred_test, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(y_test, pred_test))
        record.update(
            {
                "n_predictions": len(y_test),
                "expected_n_test": source_row["n_test"],
                "n_predictions_matches": (
                    pd.isna(source_row["n_test"])
                    or len(y_test) == int(source_row["n_test"])
                ),
                "recomputed_test_accuracy": accuracy,
                "stored_test_accuracy": source_row["test_accuracy"],
                "accuracy_matches": close(accuracy, float(source_row["test_accuracy"])),
                "recomputed_test_f1_macro": macro_f1,
                "stored_test_f1_macro": source_row["test_f1_macro"],
                "macro_f1_matches": close(macro_f1, float(source_row["test_f1_macro"])),
                "recomputed_test_balanced_accuracy": balanced,
                "stored_test_balanced_accuracy": source_row[
                    "test_balanced_accuracy"
                ],
                "balanced_accuracy_matches": close(
                    balanced, float(source_row["test_balanced_accuracy"])
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def audit_datasets(master: pd.DataFrame) -> pd.DataFrame:
    relevant = master[
        master["method_id"].eq("proposed_method")
        | master["method_id"].eq("softmax_full_finetuned")
    ].copy()
    rows: list[dict[str, Any]] = []
    for dataset_id, group in relevant.groupby("dataset_id"):
        complete = group.dropna(subset=["n_train", "n_val", "n_test"])
        row = complete.iloc[0] if len(complete) else group.iloc[0]
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": row["dataset_label"],
                "is_brain_tumor_dataset": bool(row["is_brain_tumor_dataset"]),
                "is_external_domain_control": bool(row["is_external_domain_control"]),
                "n_classes": row["n_classes"],
                "n_train": row["n_train"],
                "n_val": row["n_val"],
                "n_test": row["n_test"],
                "total": sum(
                    int(row[column]) for column in ["n_train", "n_val", "n_test"]
                ),
                "n_backbones": group["backbone"].nunique(),
                "backbones": "|".join(sorted(group["backbone"].dropna().unique())),
                "split_level": row["split_level"],
                "sample_identity_audit": row["sample_identity_audit"],
                "cross_split_duplicate_count": row["cross_split_duplicate_count"],
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def audit_final_classifiers(
    fair_matrix_path: Path,
    complete_methods: pd.DataFrame,
) -> pd.DataFrame:
    fair = pd.read_csv(fair_matrix_path, low_memory=False)
    final = fair[
        fair["method_block"].eq("final_embedding_classifiers")
        & ~fair["dataset_id"].astype(str).str.contains("ham", case=False, na=False)
        & fair["status"].eq("ok")
    ].copy()
    final = final.sort_values(
        ["val_f1_macro", "val_balanced_accuracy", "val_accuracy"],
        ascending=False,
        kind="mergesort",
    ).drop_duplicates(["dataset_id", "profile_name", "classifier"])
    aggregated = (
        final.groupby("classifier", as_index=False)
        .agg(
            n=("test_f1_macro", "count"),
            accuracy=("test_accuracy", "mean"),
            macro_f1=("test_f1_macro", "mean"),
            balanced_accuracy=("test_balanced_accuracy", "mean"),
        )
        .sort_values("classifier")
    )
    reported = complete_methods[
        complete_methods["method_id"].astype(str).str.startswith("final_")
        & ~complete_methods["method_id"].eq("last_layer_selected_classifier")
    ].copy()
    reported["classifier"] = reported["method_id"].str.replace(
        r"^final_", "", regex=True
    )
    merged = aggregated.merge(
        reported[
            [
                "classifier",
                "n",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
            ]
        ],
        on="classifier",
        how="outer",
        suffixes=("_recomputed", "_reported"),
    )
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
        merged[f"{metric}_matches"] = np.isclose(
            merged[f"{metric}_recomputed"],
            merged[f"{metric}_reported"],
            atol=5e-7,
            rtol=0,
            equal_nan=False,
        )
    merged["n_matches"] = merged["n_recomputed"].eq(merged["n_reported"])
    return merged


def audit_paper_assets(paper_dir: Path) -> pd.DataFrame:
    expected = [
        ("Figure 1", "figures/fig01_method_pipeline.png", "method diagram"),
        (
            "Figure 2",
            "figures/fig02_internal_layer_diagnostics.png",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Figure 3",
            "figures/fig03_selected_configuration.png",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Figure 4",
            "figures/fig04_block_macro_f1.png",
            "prepare_hmdf_results_section.py",
        ),
        ("Table I", "tables/table_datasets.tex", "dataset audit"),
        ("Table II", "tables/table_method_groups.tex", "method inventory"),
        (
            "Table III",
            "tables/table_ablation_summary.tex",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Table IV",
            "tables/table_complete_methods.tex",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Supplement Figure S1",
            "supplementary/figures/figS01_context_delta_heatmap.png",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Supplement Figure S2",
            "supplementary/figures/figS02_winning_context_confusion.png",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Supplement Figure S3",
            "supplementary/figures/figS03_fused_distance_geometry.png",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Supplement Table S1",
            "supplementary/tables/tableS01_dataset_summary.tex",
            "prepare_hmdf_results_section.py",
        ),
        (
            "Supplement Table S2",
            "supplementary/tables/tableS02_context_outcomes.tex",
            "prepare_hmdf_results_section.py",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "asset": asset,
                "relative_path": relative,
                "source": source,
                "exists": (paper_dir / relative).exists(),
                "size_bytes": (
                    (paper_dir / relative).stat().st_size
                    if (paper_dir / relative).exists()
                    else None
                ),
            }
            for asset, relative, source in expected
        ]
    )


def audit_stale_zip_files(paper_dir: Path) -> pd.DataFrame:
    candidates = [
        ("main (2).tex", "obsolete manuscript with old 15/26/4 counts"),
        ("tables/table_top_methods.tex", "legacy results table not used by main.tex"),
        ("tables/table_context_outcomes.tex", "legacy main-paper table"),
        ("tables/table_dataset_summary.tex", "legacy main-paper table"),
        ("figures/fig04_ablation_summary.png", "legacy ablation figure"),
        ("figures/fig05_context_delta_heatmap.png", "legacy context figure"),
        ("figures/fig06_winning_context_confusion.png", "legacy context figure"),
        (
            "Hierarchical_Multilayer_Distance_Fusion_kNN_for_Brain_Tumor_MRI_Classification_from_Internal_CNN_Embeddings.zip",
            "nested archive",
        ),
        ("main.aux", "LaTeX build artifact"),
        ("main.bbl", "LaTeX build artifact"),
        ("main.blg", "LaTeX build artifact"),
        ("main.log", "LaTeX build artifact"),
        ("main.out", "LaTeX build artifact"),
    ]
    return pd.DataFrame(
        [
            {
                "relative_path": relative,
                "exists": (paper_dir / relative).exists(),
                "reason": reason,
            }
            for relative, reason in candidates
        ]
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.results_dir / "master_experiment_table.csv"
    master = pd.read_csv(master_path, low_memory=False)
    blocks = pd.read_csv(args.results_dir / "block_comparison.csv")
    ablations = pd.read_csv(
        args.results_dir / "ablation_summary_real_for_paper.csv"
    )
    complete_methods = pd.read_csv(
        args.results_dir / "complete_method_comparison.csv"
    )
    outcomes = pd.read_csv(args.results_dir / "context_outcome_summary.csv")
    depth = pd.read_csv(args.results_dir / "internal_layer_depth_diagnostics.csv")
    selected_depth = pd.read_csv(args.results_dir / "selected_layer_depth_counts.csv")

    datasets = audit_datasets(master)
    hmdf = audit_hmdf_contexts(master)
    methods = audit_method_artifacts(master)
    method_predictions = audit_method_predictions(master)
    final_classifiers = audit_final_classifiers(args.fair_matrix, complete_methods)
    assets = audit_paper_assets(args.paper_dir)
    stale = audit_stale_zip_files(args.paper_dir)

    claims: list[dict[str, Any]] = []
    brain = master[as_bool(master["is_brain_tumor_dataset"])].copy()
    ham = master[as_bool(master["is_external_domain_control"])].copy()
    proposed = brain[brain["method_id"].eq("proposed_method")]
    contexts = brain[["dataset_id", "backbone"]].drop_duplicates()
    add_claim(
        claims,
        "Brain-MRI datasets",
        5,
        contexts["dataset_id"].nunique(),
        str(master_path),
    )
    add_claim(
        claims,
        "Backbones",
        9,
        contexts["backbone"].nunique(),
        str(master_path),
    )
    add_claim(
        claims,
        "Brain-MRI contexts",
        45,
        len(contexts),
        str(master_path),
    )
    add_claim(
        claims,
        "HAM10000 contexts kept separate",
        9,
        len(ham[["dataset_id", "backbone"]].drop_duplicates()),
        str(master_path),
    )
    add_claim(
        claims,
        "HMDF-kNN mean test macro-F1",
        0.9616,
        proposed["test_f1_macro"].mean(),
        str(master_path),
        tolerance=5e-5,
    )
    add_claim(
        claims,
        "HMDF-kNN mean test accuracy",
        0.9683,
        proposed["test_accuracy"].mean(),
        str(master_path),
        tolerance=5e-5,
    )
    add_claim(
        claims,
        "HMDF-kNN mean test balanced accuracy",
        0.9597,
        proposed["test_balanced_accuracy"].mean(),
        str(master_path),
        tolerance=5e-5,
    )

    block_expected = {
        "softmax": 0.9326,
        "last_layer": 0.9470,
        "multilayer": 0.9550,
        "proposed": 0.9616,
    }
    macro_blocks = blocks[blocks["metric"].eq("test_f1_macro")].set_index("block_id")
    for block_id, expected in block_expected.items():
        add_claim(
            claims,
            f"Method-family mean macro-F1: {block_id}",
            expected,
            float(macro_blocks.loc[block_id, "mean"]),
            str(args.results_dir / "block_comparison.csv"),
            tolerance=5e-5,
        )

    method_expected = {
        "maxvar_gcca": 0.9515,
        "mvda": 0.9504,
        "final_knn": 0.9469,
    }
    methods_indexed = complete_methods.set_index("method_id")
    for method_id, expected in method_expected.items():
        add_claim(
            claims,
            f"Complete-method mean macro-F1: {method_id}",
            expected,
            float(methods_indexed.loc[method_id, "macro_f1"]),
            str(args.results_dir / "complete_method_comparison.csv"),
            tolerance=5e-5,
        )

    ablation_expected = {
        "top1_only": (0.9560, 1.00),
        "all_layers_uniform": (0.9582, 5.67),
        "ranked_prefix_uniform": (0.9599, 2.49),
        "ranked_prefix_score_power": (0.9599, 2.49),
        "greedy_forward_score_power": (0.9604, 2.56),
        "greedy_forward_uniform": (0.9606, 2.60),
        "proposed_method_reference": (0.9616, 2.78),
    }
    ablation_indexed = ablations.set_index("variant_id")
    for variant, (expected_f1, expected_layers) in ablation_expected.items():
        add_claim(
            claims,
            f"Ablation mean macro-F1: {variant}",
            expected_f1,
            float(ablation_indexed.loc[variant, "mean_test_f1_macro"]),
            str(args.results_dir / "ablation_summary_real_for_paper.csv"),
            tolerance=5e-5,
        )
        add_claim(
            claims,
            f"Ablation mean selected layers: {variant}",
            expected_layers,
            float(ablation_indexed.loc[variant, "mean_selected_n_layers"]),
            str(args.results_dir / "ablation_summary_real_for_paper.csv"),
            tolerance=0.005,
        )

    outcome_expected = {
        "Numerical sign": (45, 28, 0, 17),
        "Practical margin (0.005)": (45, 19, 24, 2),
        "Paired bootstrap 95% CI": (38, 10, 28, 0),
        "Paired bootstrap + Holm": (38, 2, 36, 0),
    }
    outcome_indexed = outcomes.set_index("criterion")
    for criterion, expected in outcome_expected.items():
        found = tuple(
            int(outcome_indexed.loc[criterion, column])
            for column in [
                "n_contexts",
                "hmdf_ahead",
                "indistinguishable",
                "reference_ahead",
            ]
        )
        add_claim(
            claims,
            f"Context outcomes: {criterion}",
            "/".join(map(str, expected)),
            "/".join(map(str, found)),
            str(args.results_dir / "context_outcome_summary.csv"),
        )

    depth_means = depth.groupby("depth_group")[
        "mean_single_layer_val_f1_macro"
    ].mean()
    for group, expected in {
        "Early": 0.8920,
        "Mid": 0.9459,
        "Deep": 0.9666,
        "Final": 0.9582,
    }.items():
        add_claim(
            claims,
            f"Single-layer validation macro-F1: {group}",
            expected,
            float(depth_means[group]),
            str(args.results_dir / "internal_layer_depth_diagnostics.csv"),
            tolerance=5e-5,
        )
    selected_depth_indexed = selected_depth.set_index("depth_group")
    for group, expected in {"Early": 9, "Mid": 27, "Deep": 49, "Final": 40}.items():
        add_claim(
            claims,
            f"Selected layer-view count: {group}",
            expected,
            int(selected_depth_indexed.loc[group, "selected_view_count"]),
            str(args.results_dir / "selected_layer_depth_counts.csv"),
        )

    claims_frame = pd.DataFrame(claims)
    datasets.to_csv(args.output_dir / "dataset_context_audit.csv", index=False)
    hmdf.to_csv(args.output_dir / "hmdf_context_recalculation.csv", index=False)
    methods.to_csv(args.output_dir / "method_artifact_audit.csv", index=False)
    method_predictions.to_csv(
        args.output_dir / "method_prediction_recalculation.csv", index=False
    )
    final_classifiers.to_csv(
        args.output_dir / "final_classifier_recalculation.csv", index=False
    )
    assets.to_csv(args.output_dir / "figure_table_audit.csv", index=False)
    stale.to_csv(args.output_dir / "stale_zip_artifacts.csv", index=False)
    claims_frame.to_csv(args.output_dir / "claim_verification.csv", index=False)

    summary = {
        "brain_dataset_ids_match": set(
            datasets.loc[datasets["is_brain_tumor_dataset"], "dataset_id"]
        )
        == EXPECTED_BRAIN_DATASETS,
        "brain_backbones_match": set(contexts["backbone"]) == EXPECTED_BACKBONES,
        "brain_context_count": len(contexts),
        "ham_context_count": len(
            ham[["dataset_id", "backbone"]].drop_duplicates()
        ),
        "claim_status_counts": claims_frame["status"].value_counts().to_dict(),
        "hmdf_contexts_audited": len(hmdf),
        "hmdf_all_candidate_counts_185": bool(
            hmdf["n_validation_candidates"].eq(185).all()
        ),
        "hmdf_all_selected_by_validation_argmax": bool(
            hmdf["selected_config_matches_validation_argmax"].all()
        ),
        "hmdf_all_metrics_recomputed": bool(
            hmdf[
                [
                    "accuracy_matches",
                    "macro_f1_matches",
                    "balanced_accuracy_matches",
                ]
            ]
            .all(axis=1)
            .all()
        ),
        "hmdf_all_test_selection_flags_false": bool(
            ~hmdf["uses_test_for_selection"].any()
        ),
        "hmdf_all_k_in_grid": bool(hmdf["selected_k"].isin(EXPECTED_K).all()),
        "hmdf_all_weights_valid": bool(
            hmdf[["weights_nonnegative", "weights_sum_to_one"]]
            .all(axis=1)
            .all()
        ),
        "all_active_assets_exist": bool(assets["exists"].all()),
        "method_prediction_rows_available": int(
            method_predictions["prediction_available"].sum()
        ),
        "method_prediction_rows_missing": int(
            (~method_predictions["prediction_available"]).sum()
        ),
        "all_available_method_predictions_recomputed": bool(
            method_predictions.loc[
                method_predictions["prediction_available"],
                [
                    "n_predictions_matches",
                    "accuracy_matches",
                    "macro_f1_matches",
                    "balanced_accuracy_matches",
                ],
            ]
            .all(axis=1)
            .all()
        ),
        "stale_zip_artifacts_present": stale.loc[stale["exists"], "relative_path"].tolist(),
        "final_classifier_aggregates_match": bool(
            final_classifiers[
                [
                    "n_matches",
                    "accuracy_matches",
                    "macro_f1_matches",
                    "balanced_accuracy_matches",
                ]
            ]
            .all(axis=1)
            .all()
        ),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if claims_frame["status"].eq("verified").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
