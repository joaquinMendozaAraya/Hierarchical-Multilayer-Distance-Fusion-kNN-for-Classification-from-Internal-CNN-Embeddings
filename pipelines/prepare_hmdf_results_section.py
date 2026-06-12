#!/usr/bin/env python
"""Build publication-ready HMDF-kNN Results assets from frozen artifacts.

This script is CPU-only and does not train or select HMDF-kNN using test data.
Reference methods are selected independently within each context using
validation macro-F1, then compared on the frozen test predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "publication"
DEFAULT_PAPER = ROOT / "paper_hmdf_knn_publication_v2" / "masactual"
DEFAULT_LAYER_DIAGNOSTICS = (
    ROOT
    / "experiments"
    / "84_winmax_internal_diagnostics"
    / "winmax_layer_ranking_long.csv"
)
DEFAULT_FAIR_MATRIX = (
    ROOT
    / "experiments"
    / "61_final_paper_fair_matrix"
    / "fast_all_candidate_results.csv"
)

PROPOSED_ID = "proposed_method"
PRACTICAL_MARGIN = 0.005
DATASET_ORDER = [
    "brain_tumor_mri_14c",
    "brain_tumor_mri_17c",
    "brain_tumor_mri_44c",
    "brain_tumor_mri_4c",
    "sciencedb_brain_tumor_3c",
]
DATASET_LABELS = {
    "brain_tumor_mri_14c": "Brain MRI 15C",
    "brain_tumor_mri_17c": "Brain MRI 17C",
    "brain_tumor_mri_44c": "Brain MRI 44C",
    "brain_tumor_mri_4c": "Brain MRI 4C",
    "sciencedb_brain_tumor_3c": "ScienceDB 3C",
}
BACKBONE_ORDER = [
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "convnext_tiny",
]
BACKBONE_LABELS = {
    "resnet18": "R18",
    "resnet34": "R34",
    "resnet50": "R50",
    "densenet121": "DN121",
    "efficientnet_b0": "EN-B0",
    "efficientnet_b2": "EN-B2",
    "efficientnet_b3": "EN-B3",
    "mobilenet_v3_large": "MNV3",
    "convnext_tiny": "CNX-T",
}
COLORS = {
    "softmax": "#6B7280",
    "last_layer": "#377EB8",
    "multilayer": "#E69F00",
    "proposed": "#009E73",
    "negative": "#B24C63",
    "neutral": "#B9BDC5",
    "text": "#202124",
    "grid": "#D9DCE1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-comparison",
        action="store_true",
        help="Reuse context_comparison_validation_selected.csv if it exists.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def context_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["dataset_id"].astype(str)
        + "__"
        + frame["backbone"].astype(str)
        + "__"
        + frame["profile_name"].fillna("").astype(str)
    )


def selected_layer_count(value: Any) -> int:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0
    return len([part for part in text.split("|") if part])


def parse_layer_indices(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    indices = []
    for part in text.split("|"):
        try:
            indices.append(int(float(part)))
        except ValueError:
            continue
    return indices


def depth_group(position: float) -> str:
    if position <= 0.25:
        return "Early"
    if position <= 0.50:
        return "Mid"
    if position <= 0.75:
        return "Deep"
    return "Final"


def select_protocol_aligned_reference(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    brain = master[
        ~as_bool(master["is_external_domain_control"])
        & as_bool(master["complete_54_context_method"])
    ].copy()
    brain["context_id"] = context_key(brain)
    sort_metrics = ["val_f1_macro", "val_balanced_accuracy", "val_accuracy"]
    for metric in sort_metrics + ["test_f1_macro"]:
        brain[metric] = pd.to_numeric(brain[metric], errors="coerce")

    proposed = brain[brain["method_id"].eq(PROPOSED_ID)].copy()
    references = brain[~brain["method_id"].eq(PROPOSED_ID)].copy()
    references = references.sort_values(
        ["context_id", *sort_metrics, "method_id"],
        ascending=[True, False, False, False, True],
    )
    # Preserve complete rows. groupby.first() can combine the first non-null
    # value of each column from different methods when an artifact path is
    # missing, which breaks paired prediction comparisons.
    references = references.drop_duplicates("context_id", keep="first")

    keep_ref = [
        "context_id",
        "method_id",
        "method_label",
        "method_group",
        "prediction_artifact_path",
        "val_f1_macro",
        "test_f1_macro",
    ]
    comparison = proposed.merge(
        references[keep_ref],
        on="context_id",
        how="left",
        suffixes=("_hmdf", "_reference"),
    )
    comparison["delta_test_f1_macro"] = (
        comparison["test_f1_macro_hmdf"] - comparison["test_f1_macro_reference"]
    )
    comparison["strict_outcome"] = np.select(
        [
            comparison["delta_test_f1_macro"] > 1e-12,
            comparison["delta_test_f1_macro"] < -1e-12,
        ],
        ["win", "loss"],
        default="tie",
    )
    comparison["practical_outcome"] = np.select(
        [
            comparison["delta_test_f1_macro"] >= PRACTICAL_MARGIN,
            comparison["delta_test_f1_macro"] <= -PRACTICAL_MARGIN,
        ],
        ["win", "loss"],
        default="tie",
    )
    return proposed, comparison


def load_predictions(path_value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value))
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


def encode_labels(*arrays: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    labels = np.unique(np.concatenate([np.asarray(array) for array in arrays]))
    mapping = {label: index for index, label in enumerate(labels.tolist())}
    encoded = [
        np.asarray([mapping[value] for value in np.asarray(array)], dtype=np.int64)
        for array in arrays
    ]
    return encoded, labels


def macro_f1_encoded(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    matrix = np.bincount(
        y_true * n_classes + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    return float(f1.mean())


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    proposed_pred: np.ndarray,
    reference_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    encoded, _ = encode_labels(y_true, proposed_pred, reference_pred)
    y, proposed, reference = encoded
    n_classes = int(max(y.max(), proposed.max(), reference.max()) + 1)
    point = macro_f1_encoded(y, proposed, n_classes) - macro_f1_encoded(
        y, reference, n_classes
    )
    class_indices = [np.flatnonzero(y == label) for label in np.unique(y)]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        samples[index] = macro_f1_encoded(
            y[sampled], proposed[sampled], n_classes
        ) - macro_f1_encoded(y[sampled], reference[sampled], n_classes)
    low, high = np.quantile(samples, [0.025, 0.975])
    lower_tail = (np.count_nonzero(samples <= 0) + 1) / (n_bootstrap + 1)
    upper_tail = (np.count_nonzero(samples >= 0) + 1) / (n_bootstrap + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "prediction_point_delta": point,
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "bootstrap_p_two_sided": float(p_value),
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return pd.Series(adjusted, index=p_values.index)
    ordered = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(ordered)
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return pd.Series(adjusted, index=p_values.index)


def add_statistical_comparison(
    comparison: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        proposed_payload = load_predictions(row["prediction_artifact_path_hmdf"])
        reference_payload = load_predictions(row["prediction_artifact_path_reference"])
        base = {
            "context_id": row["context_id"],
            "paired_prediction_order_verified": False,
            "statistical_comparison_available": False,
        }
        if proposed_payload is None or reference_payload is None:
            rows.append(base)
            continue
        y_proposed, pred_proposed = proposed_payload
        y_reference, pred_reference = reference_payload
        if len(y_proposed) != len(y_reference) or not np.array_equal(
            y_proposed, y_reference
        ):
            rows.append(base)
            continue
        stable_seed = seed + int(
            hashlib.sha256(row["context_id"].encode("utf-8")).hexdigest()[:8], 16
        )
        stats = paired_stratified_bootstrap(
            y_proposed,
            pred_proposed,
            pred_reference,
            n_bootstrap=n_bootstrap,
            seed=stable_seed,
        )
        rows.append(
            {
                **base,
                **stats,
                "paired_prediction_order_verified": True,
                "statistical_comparison_available": True,
                "n_test_samples": len(y_proposed),
                "n_bootstrap": n_bootstrap,
            }
        )
    stats = pd.DataFrame(rows)
    merged = comparison.merge(stats, on="context_id", how="left")
    merged["bootstrap_p_holm"] = holm_adjust(merged["bootstrap_p_two_sided"])
    available = merged["statistical_comparison_available"].fillna(False)
    merged["significant_outcome_unadjusted"] = np.select(
        [
            available & (merged["bootstrap_ci95_low"] > 0),
            available & (merged["bootstrap_ci95_high"] < 0),
            ~available,
        ],
        ["win", "loss", "not_available"],
        default="not_significant",
    )
    merged["significant_outcome_holm"] = np.select(
        [
            available
            & (merged["bootstrap_p_holm"] < 0.05)
            & (merged["prediction_point_delta"] > 0),
            available
            & (merged["bootstrap_p_holm"] < 0.05)
            & (merged["prediction_point_delta"] < 0),
            ~available,
        ],
        ["win", "loss", "not_available"],
        default="not_significant",
    )
    return merged


def outcome_counts(series: pd.Series, middle: str) -> tuple[int, int, int]:
    return (
        int((series == "win").sum()),
        int((series == middle).sum()),
        int((series == "loss").sum()),
    )


def build_outcome_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict = outcome_counts(comparison["strict_outcome"], "tie")
    practical = outcome_counts(comparison["practical_outcome"], "tie")
    unadjusted = outcome_counts(
        comparison["significant_outcome_unadjusted"], "not_significant"
    )
    corrected = outcome_counts(
        comparison["significant_outcome_holm"], "not_significant"
    )
    statistical_n = int(
        comparison["statistical_comparison_available"].fillna(False).sum()
    )
    for criterion, counts, middle_label, n_contexts in [
        ("Numerical sign", strict, "Exact tie", len(comparison)),
        ("Practical margin (0.005)", practical, "Practical tie", len(comparison)),
        ("Paired bootstrap 95% CI", unadjusted, "Not significant", statistical_n),
        ("Paired bootstrap + Holm", corrected, "Not significant", statistical_n),
    ]:
        rows.append(
            {
                "criterion": criterion,
                "n_contexts": n_contexts,
                "hmdf_ahead": counts[0],
                "middle_label": middle_label,
                "indistinguishable": counts[1],
                "reference_ahead": counts[2],
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id in DATASET_ORDER:
        part = comparison[comparison["dataset_id"].eq(dataset_id)].copy()
        if part.empty:
            continue
        practical = outcome_counts(part["practical_outcome"], "tie")
        significant = outcome_counts(part["significant_outcome_holm"], "not_significant")
        reference_counts = part["method_label_reference"].value_counts()
        max_count = int(reference_counts.max())
        most_frequent = "/".join(
            reference_counts[reference_counts.eq(max_count)].index.tolist()
        )
        selected_references = "; ".join(
            f"{method} ({count})" for method, count in reference_counts.items()
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": DATASET_LABELS[dataset_id],
                "classes": int(pd.to_numeric(part["n_classes"], errors="coerce").iloc[0]),
                "contexts": len(part),
                "hmdf_macro_f1": part["test_f1_macro_hmdf"].mean(),
                "reference_macro_f1": part["test_f1_macro_reference"].mean(),
                "mean_delta": part["delta_test_f1_macro"].mean(),
                "most_frequent_reference": f"{most_frequent} ({max_count})",
                "selected_references": selected_references,
                "practical_w_t_l": f"{practical[0]}/{practical[1]}/{practical[2]}",
                "statistical_w_ns_l": (
                    f"{significant[0]}/{significant[1]}/{significant[2]}"
                ),
                "statistical_n": int(
                    part["statistical_comparison_available"].fillna(False).sum()
                ),
                "mean_selected_layers": part["selected_layers"].map(
                    selected_layer_count
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=320)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_internal_layer_diagnostics(
    proposed: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = pd.read_csv(DEFAULT_LAYER_DIAGNOSTICS)
    ranking = ranking[ranking["dataset_id"].isin(DATASET_ORDER)].copy()
    ranking["n_candidate_layers"] = (
        ranking.groupby("context_id")["layer_index"].transform("max") + 1
    )
    ranking["normalized_depth"] = (
        ranking["layer_index"] + 1
    ) / ranking["n_candidate_layers"]
    ranking["depth_group"] = ranking["normalized_depth"].map(depth_group)

    depth_order = ["Early", "Mid", "Deep", "Final"]
    context_depth = (
        ranking.groupby(["context_id", "depth_group"], as_index=False)[
            "val_f1_macro"
        ]
        .mean()
        .rename(columns={"val_f1_macro": "mean_single_layer_val_f1_macro"})
    )
    context_depth["depth_group"] = pd.Categorical(
        context_depth["depth_group"],
        categories=depth_order,
        ordered=True,
    )
    context_depth = context_depth.sort_values(["depth_group", "context_id"])
    context_depth.to_csv(
        results_dir / "internal_layer_depth_diagnostics.csv",
        index=False,
    )

    layer_count_lookup = (
        ranking.groupby("context_id")["n_candidate_layers"].first().to_dict()
    )
    selected_rows = []
    for _, row in proposed.iterrows():
        context_id = row["context_id"]
        n_layers = int(layer_count_lookup[context_id])
        raw_indices = row.get("selected_layer_indices", row.get("selected_layers"))
        for layer_index in parse_layer_indices(raw_indices):
            selected_rows.append(
                {
                    "context_id": context_id,
                    "layer_index": layer_index,
                    "n_candidate_layers": n_layers,
                    "normalized_depth": (layer_index + 1) / n_layers,
                    "depth_group": depth_group((layer_index + 1) / n_layers),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected_counts = (
        selected["depth_group"]
        .value_counts()
        .reindex(depth_order, fill_value=0)
        .rename_axis("depth_group")
        .reset_index(name="selected_view_count")
    )
    selected.to_csv(
        results_dir / "selected_layer_depth_assignments.csv",
        index=False,
    )
    selected_counts.to_csv(
        results_dir / "selected_layer_depth_counts.csv",
        index=False,
    )
    return context_depth, selected_counts


def make_internal_layer_figure(
    context_depth: pd.DataFrame,
    selected_counts: pd.DataFrame,
    figure_dir: Path,
) -> None:
    depth_order = ["Early", "Mid", "Deep", "Final"]
    values = [
        context_depth.loc[
            context_depth["depth_group"].eq(group),
            "mean_single_layer_val_f1_macro",
        ].to_numpy(dtype=float)
        for group in depth_order
    ]

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    box = left.boxplot(
        values,
        positions=np.arange(len(depth_order)),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["text"], "linewidth": 1.4},
        whiskerprops={"color": "#6B7280"},
        capprops={"color": "#6B7280"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#8EC9C0")
        patch.set_alpha(0.72)
        patch.set_edgecolor(COLORS["proposed"])

    rng = np.random.default_rng(42)
    for index, group_values in enumerate(values):
        jitter = rng.uniform(-0.12, 0.12, size=len(group_values))
        left.scatter(
            np.full(len(group_values), index) + jitter,
            group_values,
            s=12,
            alpha=0.30,
            color=COLORS["text"],
            linewidths=0,
        )
        mean_value = float(np.mean(group_values))
        left.scatter(
            index,
            mean_value,
            marker="D",
            s=34,
            color=COLORS["proposed"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        left.text(
            index,
            min(1.005, mean_value + 0.014),
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=COLORS["text"],
        )
    left.set_xticks(np.arange(len(depth_order)))
    left.set_xticklabels(depth_order)
    left.set_xlabel("Normalized CNN depth group")
    left.set_ylabel("Single-layer validation macro-F1")
    left.set_title("A. Standalone layer diagnostics", loc="left", weight="bold")
    left.set_ylim(
        max(0.0, min(float(np.min(group)) for group in values) - 0.035),
        1.015,
    )

    counts = (
        selected_counts.set_index("depth_group")
        .reindex(depth_order)["selected_view_count"]
        .to_numpy(dtype=int)
    )
    x = np.arange(len(depth_order))
    bars = right.bar(x, counts, color=COLORS["proposed"], width=0.65)
    right.set_xticks(x)
    right.set_xticklabels(depth_order)
    right.set_xlabel("Normalized CNN depth group")
    right.set_ylabel("Selected layer views")
    right.set_title("B. Frequency in selected prefixes", loc="left", weight="bold")
    right.set_ylim(0, counts.max() + max(4, int(counts.max() * 0.16)))
    for bar, value in zip(bars, counts):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    save_figure(fig, figure_dir / "fig02_internal_layer_diagnostics")


def make_block_figure(blocks: pd.DataFrame, figure_dir: Path) -> None:
    plot = blocks[blocks["metric"].eq("test_f1_macro")].sort_values("block_order")
    labels = {
        "softmax": "Softmax\nhead",
        "last_layer": "Final-embedding\nclassifiers",
        "multilayer": "Multilayer\nreference methods",
        "proposed": "HMDF-kNN",
    }
    x = np.arange(len(plot))
    means = plot["mean"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            means - plot["bootstrap_ci95_low"].to_numpy(dtype=float),
            plot["bootstrap_ci95_high"].to_numpy(dtype=float) - means,
        ]
    )
    colors = [COLORS.get(block, "#777777") for block in plot["block_id"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(x, means, yerr=errors, capsize=4, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[value] for value in plot["block_id"]])
    ax.set_ylabel("Mean test macro-F1")
    ax.set_xlabel("Method family")
    ax.set_title("Aggregate performance by method family", loc="left", weight="bold")
    ax.set_ylim(max(0.0, means.min() - 0.025), min(1.0, means.max() + 0.025))
    upper_bounds = plot["bootstrap_ci95_high"].to_numpy(dtype=float)
    for bar, value, upper in zip(bars, means, upper_bounds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            upper + 0.0015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, figure_dir / "fig04_block_macro_f1")


def make_context_heatmap(comparison: pd.DataFrame, figure_dir: Path) -> None:
    grid = comparison.pivot(
        index="dataset_id",
        columns="backbone",
        values="delta_test_f1_macro",
    ).reindex(index=DATASET_ORDER, columns=BACKBONE_ORDER)
    context_lookup = comparison.set_index(["dataset_id", "backbone"])
    finite = np.abs(grid.to_numpy(dtype=float))
    vmax = max(0.02, float(np.nanmax(finite)))
    cmap = LinearSegmentedColormap.from_list(
        "reference_to_hmdf",
        [COLORS["negative"], "#FAFAFA", COLORS["proposed"]],
    )
    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    image = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(np.arange(len(BACKBONE_ORDER)))
    ax.set_xticklabels([BACKBONE_LABELS[value] for value in BACKBONE_ORDER])
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABELS[value] for value in DATASET_ORDER])
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Brain-MRI dataset")
    ax.set_title(
        "Test macro-F1 delta vs validation-selected reference method",
        loc="left",
        weight="bold",
    )
    for row_index, dataset_id in enumerate(DATASET_ORDER):
        for column_index, backbone in enumerate(BACKBONE_ORDER):
            value = grid.loc[dataset_id, backbone]
            if pd.isna(value):
                continue
            record = context_lookup.loc[(dataset_id, backbone)]
            symbol = {"win": "+", "tie": "=", "loss": "-"}[
                record["practical_outcome"]
            ]
            star = "*" if record["significant_outcome_holm"] != "not_significant" else ""
            text_color = "white" if abs(value) > 0.58 * vmax else COLORS["text"]
            ax.text(
                column_index,
                row_index,
                f"{value:+.3f}{star}\n{symbol}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Delta test macro-F1")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figS01_context_delta_heatmap")


def make_ablation_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    label_map = {
        "top1_only": "Best single layer",
        "all_layers_uniform": "All-layer uniform fusion",
        "ranked_prefix_uniform": "Ranked-prefix uniform fusion",
        "ranked_prefix_score_power": "Ranked-prefix score-weighted fusion",
        "greedy_forward_score_power": "Greedy score-weighted fusion",
        "greedy_forward_uniform": "Greedy uniform fusion",
        "proposed_method_reference": "HMDF-kNN",
    }
    plot = summary[summary["variant_id"].isin(label_map)].sort_values("variant_order")
    labels = [label_map[value] for value in plot["variant_id"]]
    y = np.arange(len(plot))
    absolute = plot["mean_test_f1_macro"].to_numpy(dtype=float)
    ablations = plot[~plot["variant_id"].eq("proposed_method_reference")].copy()
    ablation_labels = [label_map[value] for value in ablations["variant_id"]]
    ablation_y = np.arange(len(ablations))
    losses = -ablations["mean_delta_test_f1_macro_vs_proposed"].to_numpy(dtype=float)

    fig, (left, middle, right) = plt.subplots(
        1,
        3,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": [1.15, 0.95, 1.0]},
    )
    bar_colors = [
        COLORS["proposed"] if label == "HMDF-kNN" else "#4C78A8"
        for label in labels
    ]
    left.barh(y, absolute, color=bar_colors)
    left.set_yticks(y)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlabel("Mean test macro-F1")
    left.set_title("A. Absolute performance", loc="left", weight="bold")
    left.set_xlim(max(0.0, absolute.min() - 0.004), min(1.0, absolute.max() + 0.002))
    for index, value in enumerate(absolute):
        left.text(value + 0.00012, index, f"{value:.4f}", ha="left", va="center")

    middle.axvspan(
        0,
        PRACTICAL_MARGIN,
        color=COLORS["neutral"],
        alpha=0.23,
    )
    middle.axvline(PRACTICAL_MARGIN, color=COLORS["text"], linewidth=1)
    middle.barh(ablation_y, losses, color="#4C78A8")
    middle.set_yticks(ablation_y)
    middle.set_yticklabels([])
    middle.invert_yaxis()
    middle.set_xlabel("Mean macro-F1 loss")
    middle.set_title("B. Relative loss", loc="left", weight="bold")
    middle.set_xlim(0, max(0.0062, losses.max() + 0.00055))
    for index, value in enumerate(losses):
        middle.text(value + 0.00008, index, f"{value:.4f}", ha="left", va="center")

    variant_ahead = ablations["practical_wins_vs_proposed"].to_numpy(dtype=int)
    ties = ablations["practical_ties_vs_proposed"].to_numpy(dtype=int)
    hmdf_ahead = ablations["practical_losses_vs_proposed"].to_numpy(dtype=int)
    right.barh(
        ablation_y,
        variant_ahead,
        color=COLORS["negative"],
        label="Variant ahead",
    )
    right.barh(
        ablation_y,
        ties,
        left=variant_ahead,
        color=COLORS["neutral"],
        label="Practical tie",
    )
    right.barh(
        ablation_y,
        hmdf_ahead,
        left=variant_ahead + ties,
        color=COLORS["proposed"],
        label="HMDF-kNN ahead",
    )
    right.set_yticks(ablation_y)
    right.set_yticklabels([])
    right.invert_yaxis()
    right.set_xlabel("Number of contexts")
    right.set_title("C. Practical outcomes", loc="left", weight="bold")
    right.set_xlim(0, 45)
    right.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout()
    save_figure(fig, figure_dir / "fig04_ablation_summary")


def weight_family(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("dirichlet"):
        return "Dirichlet-sampled"
    if text.startswith("val_power") or text.startswith("score"):
        return "Score-based"
    if text == "uniform":
        return "Uniform"
    return "Other"


def build_complete_method_summary(master: pd.DataFrame) -> pd.DataFrame:
    brain = master[
        ~as_bool(master["is_external_domain_control"])
        & as_bool(master["complete_54_context_method"])
    ].copy()
    aggregate = (
        brain.groupby(["method_id"], as_index=False)
        .agg(
            n=("test_f1_macro", "count"),
            accuracy=("test_accuracy", "mean"),
            macro_f1=("test_f1_macro", "mean"),
            balanced_accuracy=("test_balanced_accuracy", "mean"),
        )
        .set_index("method_id")
    )

    fair = pd.read_csv(DEFAULT_FAIR_MATRIX, low_memory=False)
    final = fair[
        fair["method_block"].eq("final_embedding_classifiers")
        & ~fair["dataset_id"].astype(str).str.contains("ham", case=False, na=False)
        & fair["status"].eq("ok")
    ].copy()
    final = final.sort_values(
        ["val_f1_macro", "val_balanced_accuracy", "val_accuracy"],
        ascending=False,
    ).drop_duplicates(["dataset_id", "profile_name", "classifier"])
    final_aggregate = (
        final.groupby("classifier", as_index=False)
        .agg(
            n=("test_f1_macro", "count"),
            accuracy=("test_accuracy", "mean"),
            macro_f1=("test_f1_macro", "mean"),
            balanced_accuracy=("test_balanced_accuracy", "mean"),
        )
        .set_index("classifier")
    )

    rows: list[dict[str, Any]] = []

    def add_master(
        family: str,
        method_id: str,
        method_tex: str,
        source_tex: str,
        input_representation: str,
    ) -> None:
        metrics = aggregate.loc[method_id]
        rows.append(
            {
                "family": family,
                "method_id": method_id,
                "method_tex": method_tex,
                "source_tex": source_tex,
                "input_representation": input_representation,
                **metrics.to_dict(),
            }
        )

    def add_final(
        classifier: str,
        method_tex: str,
        source_tex: str,
    ) -> None:
        metrics = final_aggregate.loc[classifier]
        rows.append(
            {
                "family": "Final-embedding classifiers",
                "method_id": f"final_{classifier}",
                "method_tex": method_tex,
                "source_tex": source_tex,
                "input_representation": "Final CNN embedding",
                **metrics.to_dict(),
            }
        )

    add_master(
        "Direct CNN baseline",
        "softmax_full_finetuned",
        "Softmax head",
        "CNN classifier head",
        "Native class logits",
    )
    add_master(
        "Final-embedding classifiers",
        "last_layer_selected_classifier",
        "Validation-selected final classifier",
        "This study",
        "Final CNN embedding",
    )
    add_final("knn", "kNN", r"Cover and Hart (1967) \cite{cover1967nearest}")
    add_final(
        "linear_svm",
        "Linear SVM",
        r"Cortes and Vapnik (1995) \cite{cortes1995svm}",
    )
    add_final(
        "random_forest",
        "Random forest",
        r"Breiman (2001) \cite{breiman2001random}",
    )
    add_final("logreg", "Logistic regression", "Classical classifier")
    add_final(
        "gmm_diag",
        "Diagonal GMM",
        r"EM/GMM \cite{dempster1977em,chopin2024gmm}",
    )
    add_final(
        "xgboost",
        "XGBoost",
        r"Chen and Guestrin (2016) \cite{chen2016xgboost}",
    )
    add_final("gaussian_nb", "Gaussian naive Bayes", "Classical classifier")

    add_master(
        "Multilayer controls",
        "raw_concat_linear",
        "Raw concatenation + linear",
        "This study",
        "All layer embeddings",
    )
    add_master(
        "Multilayer controls",
        "concat_pca_linear",
        "Concatenation + PCA + linear",
        "This study",
        "All layer embeddings",
    )
    add_master(
        "Multilayer controls",
        "uniform_layer_softvote",
        "Uniform layer soft vote",
        "This study",
        "Layer-level decisions",
    )
    add_master(
        "Multilayer controls",
        "uniform_kernel_svm",
        "Uniform kernel SVM",
        r"SVM \cite{cortes1995svm}",
        "Layer-wise kernels",
    )

    add_master(
        "Literature multilayer references",
        "fradi_mlcff",
        (
            r"MLCFF-style fusion (Fradi et al., 2021)$^\dagger$ "
            r"\cite{fradi2021multilayer}"
        ),
        r"Fradi et al. (2021) \cite{fradi2021multilayer}",
        "Stage embeddings",
    )
    add_master(
        "Literature multilayer references",
        "head2toe",
        (
            r"Head2Toe-style fusion (Evci et al., 2022)$^\dagger$ "
            r"\cite{evci2022head2toe}"
        ),
        r"Evci et al. (2022) \cite{evci2022head2toe}",
        "Stage embeddings",
    )
    add_master(
        "Literature multilayer references",
        "easymkl",
        r"EasyMKL$^\dagger$",
        r"Aiolli and Donini (2015) \cite{aiolli2015easymkl}",
        "Layer-wise kernels",
    )

    add_master(
        "Multiview references",
        "maxvar_gcca",
        r"MAXVAR-GCCA$^\dagger$",
        r"Carroll (1968); Kettenring (1971) \cite{carroll1968gca,kettenring1971canonical}",
        "Paired layer views",
    )
    add_master(
        "Multiview references",
        "gmlda",
        r"GMLDA$^\dagger$",
        r"Sharma et al. (2012) \cite{sharma2012gma}",
        "Paired layer views",
    )
    add_master(
        "Multiview references",
        "mvda",
        r"MvDA$^\dagger$",
        r"Kan et al. (2016) \cite{kan2016mvda}",
        "Paired layer views",
    )

    add_master(
        "Metric/prototype references",
        "concat_nca_knn",
        r"NCA + kNN$^\dagger$",
        r"Goldberger et al. (2004) \cite{goldberger2004nca}",
        "Concatenated layer views",
    )
    add_master(
        "Metric/prototype references",
        "kmex_final_embedding",
        r"KMEx (Gautam et al., 2024) \cite{gautam2024kmex}",
        r"Gautam et al. (2024) \cite{gautam2024kmex}",
        "Final CNN embedding",
    )

    add_master(
        "This work",
        "proposed_method",
        "HMDF-kNN",
        "This work",
        "Selected layer distances",
    )
    result = pd.DataFrame(rows)
    numeric_columns = ["n", "accuracy", "macro_f1", "balanced_accuracy"]
    result[numeric_columns] = result[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return result


def make_configuration_figure(proposed: pd.DataFrame, figure_dir: Path) -> None:
    layer_counts = proposed["selected_layers"].map(selected_layer_count).value_counts()
    k_counts = pd.to_numeric(proposed["k"], errors="coerce").dropna().astype(int).value_counts()
    family_counts = proposed["fusion_weighting"].map(weight_family).value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    panels = [
        (
            axes[0],
            [1, 2, 3, 4],
            [int(layer_counts.get(value, 0)) for value in [1, 2, 3, 4]],
            "Selected layers",
            "A. Layer-prefix size",
        ),
        (
            axes[1],
            [1, 3, 5, 7, 11],
            [int(k_counts.get(value, 0)) for value in [1, 3, 5, 7, 11]],
            "Selected k",
            "B. kNN neighborhood",
        ),
        (
            axes[2],
            ["Uniform", "Score-based", "Dirichlet"],
            [
                int(family_counts.get(value, 0))
                for value in ["Uniform", "Score-based", "Dirichlet-sampled"]
            ],
            "Selected weight family",
            "C. Distance weights",
        ),
    ]
    for ax, categories, values, xlabel, title in panels:
        x = np.arange(len(categories))
        bars = ax.bar(x, values, color=COLORS["proposed"], width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of contexts")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(0, max(values) + max(2, int(max(values) * 0.18)))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.4,
                str(value),
                ha="center",
                va="bottom",
            )
    fig.tight_layout()
    save_figure(fig, figure_dir / "fig03_selected_configuration")


def confusion_matrix_encoded(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels.tolist())}
    true_encoded = np.asarray([mapping[value] for value in y_true], dtype=int)
    pred_encoded = np.asarray([mapping[value] for value in y_pred], dtype=int)
    n_classes = len(labels)
    return np.bincount(
        true_encoded * n_classes + pred_encoded,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)


def per_class_metrics(matrix: np.ndarray, names: list[str], method: str) -> pd.DataFrame:
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return pd.DataFrame(
        {
            "method": method,
            "class": names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": matrix.sum(axis=1).astype(int),
        }
    )


def select_winning_context(comparison: pd.DataFrame) -> pd.Series | None:
    candidates = comparison[
        comparison["significant_outcome_holm"].eq("win")
        & comparison["practical_outcome"].eq("win")
    ].sort_values("delta_test_f1_macro", ascending=False)
    if candidates.empty:
        candidates = comparison[
            comparison["practical_outcome"].eq("win")
        ].sort_values("delta_test_f1_macro", ascending=False)
    if candidates.empty:
        return None
    return candidates.iloc[0]


def resolve_profile_path(row: pd.Series) -> Path | None:
    direct = row.get("profile_path")
    if pd.notna(direct):
        candidate = Path(str(direct))
        if candidate.exists():
            return candidate
    selected_result = row.get("selected_result_path")
    if pd.notna(selected_result):
        result_path = Path(str(selected_result))
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            candidate = Path(str(payload.get("profile_path", "")))
            if candidate.exists():
                return candidate
    return None


def context_class_names(row: pd.Series, labels: np.ndarray) -> list[str]:
    profile_path = resolve_profile_path(row)
    if profile_path is None:
        return [str(value) for value in labels]
    split_info = profile_path / "split_info.json"
    if split_info.exists():
        payload = json.loads(split_info.read_text(encoding="utf-8"))
        names = payload.get("classes", [])
        if len(names) == len(labels):
            return [str(value) for value in names]
    return [str(value) for value in labels]


def make_winning_context_confusion(
    comparison: pd.DataFrame,
    figure_dir: Path,
    results_dir: Path,
) -> None:
    selected = select_winning_context(comparison)
    if selected is None:
        return
    row = selected
    proposed_payload = load_predictions(row["prediction_artifact_path_hmdf"])
    reference_payload = load_predictions(row["prediction_artifact_path_reference"])
    if proposed_payload is None or reference_payload is None:
        return
    y_true, proposed_pred = proposed_payload
    y_reference, reference_pred = reference_payload
    if not np.array_equal(y_true, y_reference):
        return
    labels = np.unique(np.concatenate([y_true, proposed_pred, reference_pred]))
    class_names = context_class_names(row, labels)
    proposed_matrix = confusion_matrix_encoded(y_true, proposed_pred, labels)
    reference_matrix = confusion_matrix_encoded(y_true, reference_pred, labels)
    normalized = [
        np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix, dtype=float),
            where=matrix.sum(axis=1, keepdims=True) > 0,
        )
        for matrix in [proposed_matrix, reference_matrix]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=True, sharey=True)
    titles = [
        f"HMDF-kNN\nmacro-F1={row['test_f1_macro_hmdf']:.4f}",
        (
            f"{row['method_label_reference']}\n"
            f"macro-F1={row['test_f1_macro_reference']:.4f}"
        ),
    ]
    for ax, matrix, title in zip(axes, normalized, titles):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted class")
        ax.set_title(title, weight="bold")
        for row_index in range(len(class_names)):
            for column_index in range(len(class_names)):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else COLORS["text"],
                    fontsize=8,
                )
    axes[0].set_ylabel("True class")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.18, top=0.80, wspace=0.28)
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.018, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Row-normalized proportion")
    fig.suptitle(
        (
            f"Holm-significant practical win: {DATASET_LABELS[row['dataset_id']]} "
            f"({BACKBONE_LABELS.get(row['backbone'], row['backbone'])})"
        ),
        weight="bold",
    )
    save_figure(fig, figure_dir / "figS02_winning_context_confusion")

    metrics = pd.concat(
        [
            per_class_metrics(proposed_matrix, class_names, "HMDF-kNN"),
            per_class_metrics(
                reference_matrix, class_names, str(row["method_label_reference"])
            ),
        ],
        ignore_index=True,
    )
    metrics.insert(0, "dataset_id", row["dataset_id"])
    metrics.insert(1, "backbone", row["backbone"])
    metrics.to_csv(results_dir / "winning_context_per_class.csv", index=False)
    metadata = {
        "dataset_id": row["dataset_id"],
        "selection_rule": (
            "largest macro-F1 delta among Holm-corrected significant practical wins"
        ),
        "backbone": row["backbone"],
        "reference_method": row["method_label_reference"],
        "hmdf_test_macro_f1": row["test_f1_macro_hmdf"],
        "reference_test_macro_f1": row["test_f1_macro_reference"],
        "delta_test_macro_f1": row["delta_test_f1_macro"],
        "bootstrap_p_holm": row["bootstrap_p_holm"],
        "class_order": class_names,
    }
    (results_dir / "winning_context.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def pca_coordinates(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular_values[:2]


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    squared_norms = np.sum(values * values, axis=1, keepdims=True)
    squared = squared_norms + squared_norms.T - 2.0 * values @ values.T
    return np.sqrt(np.maximum(squared, 0.0))


def classical_mds(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(distances, dtype=np.float64)
    n_samples = distances.shape[0]
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order[:2]], 0.0)
    coordinates = eigenvectors[:, order[:2]] * np.sqrt(positive)
    return coordinates, eigenvalues[order]


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float64),
        where=norms > 0,
    )


def make_fused_distance_geometry(
    comparison: pd.DataFrame,
    figure_dir: Path,
    results_dir: Path,
) -> None:
    selected = select_winning_context(comparison)
    if selected is None:
        return
    row = selected
    profile_path = resolve_profile_path(row)
    if profile_path is None:
        return
    test_dir = profile_path / "embeddings" / "multicapa_norm" / "test"
    if not test_dir.exists():
        return

    layer_indices = parse_layer_indices(row["selected_layer_indices"])
    layer_dims = parse_layer_indices(row["selected_layer_dims"])
    weights = [
        float(value)
        for value in str(row["fusion_weights"]).split("|")
        if str(value).strip()
    ]
    all_dims = parse_layer_indices(row["view_dims"])
    if not layer_indices or len(layer_dims) != len(weights):
        return

    labels = np.load(test_dir / "labels.npy")
    selected_views = [
        l2_normalize(np.load(test_dir / f"z_dim_{dimension}.npy"))
        for dimension in layer_dims
    ]
    final_view = l2_normalize(np.load(test_dir / f"z_dim_{all_dims[-1]}.npy"))
    best_view = selected_views[0]
    fused_distances = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for weight, view in zip(weights, selected_views):
        fused_distances += weight * pairwise_euclidean(view)

    final_coordinates = pca_coordinates(final_view)
    best_coordinates = pca_coordinates(best_view)
    fused_coordinates, eigenvalues = classical_mds(fused_distances)
    class_names = context_class_names(row, np.unique(labels))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    panels = [
        (axes[0], final_coordinates, "A. Final embedding (PCA)"),
        (axes[1], best_coordinates, "B. Best single layer (PCA)"),
        (axes[2], fused_coordinates, "C. Fused distance metric (MDS)"),
    ]
    for ax, coordinates, title in panels:
        for class_index, class_name in enumerate(class_names):
            mask = labels == class_index
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=14,
                alpha=0.55,
                color=palette[class_index % len(palette)],
                label=class_name,
                linewidths=0,
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(alpha=0.25)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(class_names),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, figure_dir / "figS03_fused_distance_geometry")

    positive = eigenvalues[eigenvalues > 0]
    metadata = {
        "dataset_id": row["dataset_id"],
        "backbone": row["backbone"],
        "selection_rule": (
            "same Holm-corrected significant practical-win context as the "
            "class-level error analysis"
        ),
        "selected_layer_indices": layer_indices,
        "selected_layer_dims": layer_dims,
        "fusion_weights": weights,
        "final_embedding_dim": all_dims[-1],
        "best_single_layer_dim": layer_dims[0],
        "n_test_samples": int(len(labels)),
        "fused_projection": "classical MDS over the HMDF-kNN weighted fused test-test distance matrix",
        "final_and_single_layer_projection": "two-component PCA by centered SVD",
        "mds_positive_eigenvalue_fraction_first_two": (
            float(positive[:2].sum() / positive.sum()) if len(positive) else None
        ),
        "use": "qualitative visualization only; not used for model selection or evaluation",
    }
    (results_dir / "fused_distance_geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def short_reference_label(value: str) -> str:
    replacements = {
        "Last-layer selected classifier": "Final emb.",
        "MLCFF-style fusion": "MLCFF",
        "Head2Toe-style fusion": "Head2Toe",
        "Uniform kernel SVM": "Uniform SVM",
        "Softmax head": "Softmax",
        "NCA + kNN": "NCA+kNN",
    }
    return replacements.get(str(value), str(value))


def write_complete_method_table(summary: pd.DataFrame, table_dir: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Complete aggregate method comparison grouped by method family.}",
        r"\label{tab:complete_methods}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{p{3.1cm}p{7.1cm}rrrr}",
        r"\toprule",
        r"Method family & Method & $n$ & Acc. & Macro-F1 & Bal. acc. \\",
        r"\midrule",
    ]
    for family, part in summary.groupby("family", sort=False):
        for row_index, (_, row) in enumerate(part.iterrows()):
            family_cell = latex_escape(family) if row_index == 0 else ""
            lines.append(
                "{} & {} & {} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                    family_cell,
                    row["method_tex"],
                    int(row["n"]),
                    row["accuracy"],
                    row["macro_f1"],
                    row["balanced_accuracy"],
                )
            )
        if family != summary["family"].iloc[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        (
            r"\scriptsize Each row averages 45 brain-MRI dataset-backbone contexts. "
            r"Hyperparameters are selected by validation macro-F1 within each "
            r"method and context. Individual final-embedding rows select the "
            r"best hyperparameter setting for that classifier. "
            r"$^\dagger$Stage-level adaptation over the saved CNN embedding views; "
            r"complete configurations and per-context outputs are provided in the "
            r"supplementary artifacts."
        ),
        r"\end{table*}",
    ]
    (table_dir / "table_complete_methods.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_ablation_table(summary: pd.DataFrame, table_dir: Path) -> None:
    label_map = {
        "top1_only": "Best single layer",
        "all_layers_uniform": "All-layer uniform distance fusion",
        "ranked_prefix_uniform": "Ranked-prefix uniform fusion",
        "ranked_prefix_score_power": "Ranked-prefix score-weighted fusion",
        "greedy_forward_score_power": "Greedy score-weighted fusion",
        "greedy_forward_uniform": "Greedy uniform fusion",
        "proposed_method_reference": "HMDF-kNN",
    }
    aggregation_map = {
        "top1_only": "None",
        "all_layers_uniform": "Uniform distance aggregation",
        "ranked_prefix_uniform": "Uniform distance aggregation",
        "ranked_prefix_score_power": "Score-based distance aggregation",
        "greedy_forward_score_power": "Score-based distance aggregation",
        "greedy_forward_uniform": "Uniform distance aggregation",
        "proposed_method_reference": (
            "Validation-selected weighted distance aggregation"
        ),
    }
    selection_map = {
        "top1_only": "Best validation-ranked layer",
        "all_layers_uniform": "All layers",
        "ranked_prefix_uniform": "Validation-ranked prefix",
        "ranked_prefix_score_power": "Validation-ranked prefix",
        "greedy_forward_score_power": "Greedy selected subset",
        "greedy_forward_uniform": "Greedy selected subset",
        "proposed_method_reference": "Validation-ranked prefix",
    }
    plot = summary[summary["variant_id"].isin(label_map)].sort_values("variant_order")
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Design ablation of HMDF-kNN aggregation strategy.}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{p{3.5cm}p{4.7cm}p{3.8cm}rr}",
        r"\toprule",
        (
            r"Variant & Aggregation strategy & Layer selection & "
            r"Mean layers & Mean test macro-F1 \\"
        ),
        r"\midrule",
    ]
    for _, row in plot.iterrows():
        lines.append(
            "{} & {} & {} & {:.2f} & {:.4f} \\\\".format(
                label_map[row["variant_id"]],
                aggregation_map[row["variant_id"]],
                selection_map[row["variant_id"]],
                row["mean_selected_n_layers"],
                row["mean_test_f1_macro"],
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        (
            r"\footnotesize Results are averaged over 45 brain-MRI contexts. "
            r"Rows summarize complete validation-selected design variants and "
            r"may jointly differ in layer selection, aggregation, and $k$; they "
            r"are not an isolated test of the weight family."
        ),
        r"\end{table*}",
    ]
    (table_dir / "table_ablation_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_dataset_table(summary: pd.DataFrame, table_dir: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Dataset-level comparison against context-wise validation-selected non-proposed reference methods.}",
        r"\label{tab:dataset_summary}",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{p{1.8cm}rr rrr p{2.4cm}p{4.2cm}ll}",
        r"\toprule",
        (
            r"Dataset & C & $n$ & HMDF-kNN & Val.-selected ref. & Delta & "
            r"Most frequent ref. & Reference methods selected & Practical W/T/L & Stat. W/NS/L ($n$) \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        selected_counts = []
        for item in str(row["selected_references"]).split("; "):
            if " (" in item:
                method, count = item.rsplit(" (", 1)
                selected_counts.append(
                    f"{short_reference_label(method)} ({count}"
                )
            else:
                selected_counts.append(short_reference_label(item))
        most_frequent = str(row["most_frequent_reference"])
        for long_name in [
            "Last-layer selected classifier",
            "MLCFF-style fusion",
            "Head2Toe-style fusion",
            "Uniform kernel SVM",
            "Softmax head",
            "NCA + kNN",
        ]:
            most_frequent = most_frequent.replace(
                long_name,
                short_reference_label(long_name),
            )
        lines.append(
            "{} & {} & {} & {:.4f} & {:.4f} & {:+.4f} & {} & {} & {} & {} ({}) \\\\".format(
                latex_escape(row["dataset"]),
                int(row["classes"]),
                int(row["contexts"]),
                row["hmdf_macro_f1"],
                row["reference_macro_f1"],
                row["mean_delta"],
                latex_escape(most_frequent),
                latex_escape("; ".join(selected_counts)),
                row["practical_w_t_l"],
                row["statistical_w_ns_l"],
                int(row["statistical_n"]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        (
            r"\tiny Within each dataset-backbone context, the non-proposed "
            r"reference is selected by validation macro-F1 before test comparison; "
            r"the displayed reference macro-F1 is the mean of those context-wise "
            r"selections. Practical W/T/L uses a 0.005 margin. Statistical W/NS/L "
            r"uses only contexts with paired predictions and applies Holm correction."
        ),
        r"\end{table}",
    ]
    (table_dir / "tableS01_dataset_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_outcome_table(summary: pd.DataFrame, table_dir: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Context-level outcomes against validation-selected reference methods.}",
        r"\label{tab:context_outcomes}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Criterion & $n$ & HMDF ahead & Tie/NS & Ref. ahead \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["criterion"]),
                int(row["n_contexts"]),
                int(row["hmdf_ahead"]),
                int(row["indistinguishable"]),
                int(row["reference_ahead"]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        (
            r"\scriptsize Numerical and practical outcomes use all 45 contexts. "
            r"NS denotes not statistically significant among contexts with paired "
            r"predictions. Practical ties are defined only by the 0.005 margin and "
            r"are not equivalent to $p>0.05$. Holm correction controls family-wise "
            r"error over the available paired tests."
        ),
        r"\end{table}",
    ]
    (table_dir / "tableS02_context_outcomes.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_protocol_note(
    output: Path,
    comparison: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> None:
    statistical_n = int(
        comparison["statistical_comparison_available"].fillna(False).sum()
    )
    oracle_path = output / "delta_vs_best_nonproposed_by_context.csv"
    oracle_note = ""
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        strict_wins = int(
            (oracle["delta_vs_best_nonproposed_test_f1_macro"] > 1e-12).sum()
        )
        strict_losses = int(
            (oracle["delta_vs_best_nonproposed_test_f1_macro"] < -1e-12).sum()
        )
        practical = oracle["practical_outcome_margin_0p005"].value_counts()
        oracle_note = f"""
## Retrospective test-envelope analysis

The legacy context envelope chooses the highest observed non-proposed test
macro-F1 after evaluation. It gives {strict_wins} numerical wins and
{strict_losses} numerical losses for HMDF-kNN; with the 0.005 practical margin,
the counts are {int(practical.get('practical_win', 0))}/{int(practical.get('practical_tie', 0))}/{int(practical.get('practical_loss', 0))}.

This envelope is retained as a conservative descriptive analysis only. It is
not used for inferential claims because the reference method is chosen using
the same test outcomes being compared.
"""
    markdown_rows = [
        "| Criterion | HMDF ahead | Tie/NS | Reference ahead |",
        "|---|---:|---:|---:|",
    ]
    for _, row in outcome_summary.iterrows():
        markdown_rows.append(
            f"| {row['criterion']} | {int(row['hmdf_ahead'])} | "
            f"{int(row['indistinguishable'])} | {int(row['reference_ahead'])} |"
        )
    note = f"""# HMDF-kNN Results comparison protocol

Generated from frozen artifacts. No model was retrained.

## Main comparison

- Contexts: {len(comparison)} brain-MRI dataset-backbone pairs.
- Proposed method: HMDF-kNN.
- Reference selection: highest validation macro-F1 among complete non-proposed
  methods in the same context; validation balanced accuracy and validation
  accuracy are tie-breakers.
- Test use: one frozen comparison after reference selection.
- Practical equivalence margin: {PRACTICAL_MARGIN:.3f} macro-F1.
- Statistical analysis: paired class-stratified bootstrap over identical test
  samples, {n_bootstrap} replicates, base seed {seed}. Paired predictions were
  available for {statistical_n} of {len(comparison)} contexts.
- Multiple comparisons: Holm correction over the {statistical_n} available
  context-level p-values.

## Outcome definitions

1. Numerical: sign of the test macro-F1 delta.
2. Practical: win/loss only when the absolute delta reaches 0.005.
3. Statistical, unadjusted: paired bootstrap 95% CI excludes zero.
4. Statistical, corrected: Holm-adjusted p < 0.05 among contexts with paired
   prediction artifacts; the delta sign determines which method is ahead.

{chr(10).join(markdown_rows)}
{oracle_note}
"""
    (output / "results_section_statistical_protocol.md").write_text(
        note,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configure_style()
    figure_dir = args.paper_dir / "figures"
    table_dir = args.paper_dir / "tables"
    supplementary_dir = args.paper_dir / "supplementary"
    supplementary_figure_dir = supplementary_dir / "figures"
    supplementary_table_dir = supplementary_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    supplementary_figure_dir.mkdir(parents=True, exist_ok=True)
    supplementary_table_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(
        args.results_dir / "master_experiment_table.csv",
        low_memory=False,
    )
    blocks = pd.read_csv(args.results_dir / "block_comparison.csv")
    ablations = pd.read_csv(
        args.results_dir / "ablation_summary_real_for_paper.csv"
    )

    proposed, comparison = select_protocol_aligned_reference(master)
    cached_comparison = (
        args.results_dir / "context_comparison_validation_selected.csv"
    )
    if args.reuse_comparison and cached_comparison.exists():
        comparison = pd.read_csv(cached_comparison, low_memory=False)
    else:
        comparison = add_statistical_comparison(
            comparison,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    outcome_summary = build_outcome_summary(comparison)
    dataset_summary = build_dataset_summary(comparison)
    complete_methods = build_complete_method_summary(master)
    context_depth, selected_depth_counts = build_internal_layer_diagnostics(
        proposed,
        args.results_dir,
    )

    comparison.to_csv(
        args.results_dir / "context_comparison_validation_selected.csv",
        index=False,
    )
    outcome_summary.to_csv(
        args.results_dir / "context_outcome_summary.csv",
        index=False,
    )
    dataset_summary.to_csv(
        args.results_dir / "dataset_level_summary.csv",
        index=False,
    )
    complete_methods.to_csv(
        args.results_dir / "complete_method_comparison.csv",
        index=False,
    )

    make_internal_layer_figure(context_depth, selected_depth_counts, figure_dir)
    make_configuration_figure(proposed, figure_dir)
    make_block_figure(blocks, figure_dir)
    make_context_heatmap(comparison, supplementary_figure_dir)
    make_winning_context_confusion(
        comparison,
        supplementary_figure_dir,
        args.results_dir,
    )
    make_fused_distance_geometry(
        comparison,
        supplementary_figure_dir,
        args.results_dir,
    )
    write_ablation_table(ablations, table_dir)
    write_complete_method_table(complete_methods, table_dir)
    write_dataset_table(dataset_summary, supplementary_table_dir)
    write_outcome_table(outcome_summary, supplementary_table_dir)
    write_protocol_note(
        args.results_dir,
        comparison,
        outcome_summary,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print(outcome_summary.to_string(index=False))
    print()
    print(dataset_summary.to_string(index=False))
    print()
    print("Figures:", figure_dir)
    print("Tables:", table_dir)
    print("Supplementary artifacts:", supplementary_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
