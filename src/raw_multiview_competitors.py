"""Independent post-hoc multi-view competitors for frozen CNN embeddings.

The functions in this module intentionally do not import any VCHMF/WinMax
operator. Each external method receives only train/validation/test layer
embeddings and labels. Hyperparameters are selected on validation, and test is
evaluated only for the frozen selected candidate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC, SVC


EPS = 1e-10
_CANDIDATE_REPORTER: Callable[[dict[str, Any]], None] | None = None


def set_candidate_reporter(
    reporter: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Register a lightweight callback invoked after each validation candidate."""
    global _CANDIDATE_REPORTER
    _CANDIDATE_REPORTER = reporter


@dataclass
class MultiViewData:
    train_views: list[np.ndarray]
    val_views: list[np.ndarray]
    test_views: list[np.ndarray]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    view_names: list[str]
    view_dims: list[int]
    classes: np.ndarray


@dataclass
class MethodOutput:
    method_id: str
    display_name: str
    fidelity: str
    source: str
    candidates: list[dict[str, Any]]
    selected_config: dict[str, Any]
    val_pred: np.ndarray
    test_pred: np.ndarray
    val_proba: np.ndarray | None
    test_proba: np.ndarray | None
    diagnostics: dict[str, Any]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc_ovr_macro": float("nan"),
    }
    if proba is not None:
        try:
            labels = np.asarray(classes if classes is not None else np.unique(y_true))
            out["auroc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    proba,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except (ValueError, TypeError):
            pass
    return out


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)).astype(np.float32)


def estimator_proba(estimator: Any, x: Any, classes: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(x), dtype=np.float32)
        est_classes = np.asarray(getattr(estimator, "classes_", classes))
        out = np.zeros((len(raw), len(classes)), dtype=np.float32)
        for j, label in enumerate(est_classes):
            idx = np.flatnonzero(classes == label)
            if len(idx):
                out[:, idx[0]] = raw[:, j]
        return out
    if hasattr(estimator, "decision_function"):
        try:
            raw = np.asarray(estimator.decision_function(x))
            if raw.ndim == 2 and raw.shape[1] != len(classes):
                return None
            return _softmax(raw)
        except Exception:
            return None
    return None


def _candidate_row(
    config: dict[str, Any],
    y_val: np.ndarray,
    pred_val: np.ndarray,
    proba_val: np.ndarray | None,
    classes: np.ndarray,
    fit_seconds: float,
    final_dim: int,
    n_train_fit: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_json": json.dumps(config, sort_keys=True),
        "fit_seconds": float(fit_seconds),
        "final_dim": int(final_dim),
        "n_train_fit": int(n_train_fit),
    }
    row.update({f"val_{k}": v for k, v in classification_metrics(y_val, pred_val, proba_val, classes).items()})
    if _CANDIDATE_REPORTER is not None:
        _CANDIDATE_REPORTER(row)
    return row


def _is_better(row: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current = (
        row["val_f1_macro"],
        row["val_balanced_accuracy"],
        row["val_accuracy"],
        -row["final_dim"],
        -row["fit_seconds"],
    )
    previous = (
        best["val_f1_macro"],
        best["val_balanced_accuracy"],
        best["val_accuracy"],
        -best["final_dim"],
        -best["fit_seconds"],
    )
    return current > previous


def l2_views(views: list[np.ndarray]) -> list[np.ndarray]:
    return [normalize(np.asarray(x, dtype=np.float32), norm="l2").astype(np.float32) for x in views]


def concat_l2(views: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(l2_views(views), axis=1).astype(np.float32)


def _logreg(c: float, seed: int, max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(
        C=float(c),
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )


def run_raw_concat_linear(
    data: MultiViewData,
    *,
    c_grid: list[float],
    seed: int,
) -> MethodOutput:
    xtr, xva, xte = map(
        concat_l2,
        [data.train_views, data.val_views, data.test_views],
    )
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_model = None
    for c in c_grid:
        start = time.perf_counter()
        model = _logreg(c, seed).fit(xtr, data.y_train)
        pred = model.predict(xva)
        proba = estimator_proba(model, xva, data.classes)
        row = _candidate_row(
            {"C": c},
            data.y_val,
            pred,
            proba,
            data.classes,
            time.perf_counter() - start,
            xtr.shape[1],
            len(data.y_train),
        )
        candidates.append(row)
        if _is_better(row, best_row):
            best_row, best_model = row, model
    assert best_model is not None and best_row is not None
    return MethodOutput(
        method_id="raw_concat_linear",
        display_name="Raw concatenation + linear classifier",
        fidelity="mandatory_control",
        source="Standard feature-level fusion control",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=best_model.predict(xva),
        test_pred=best_model.predict(xte),
        val_proba=estimator_proba(best_model, xva, data.classes),
        test_proba=estimator_proba(best_model, xte, data.classes),
        diagnostics={"input_dim": int(xtr.shape[1]), "uses_all_views": True},
    )


def run_concat_pca_linear(
    data: MultiViewData,
    *,
    pca_dims: list[int],
    c_grid: list[float],
    seed: int,
) -> MethodOutput:
    xtr, xva, xte = map(
        concat_l2,
        [data.train_views, data.val_views, data.test_views],
    )
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_pack = None
    feasible = sorted(set(min(int(d), xtr.shape[0] - 1, xtr.shape[1]) for d in pca_dims))
    for dim in feasible:
        if dim < 2:
            continue
        start_pca = time.perf_counter()
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
        ztr = pca.fit_transform(xtr).astype(np.float32)
        zva = pca.transform(xva).astype(np.float32)
        pca_seconds = time.perf_counter() - start_pca
        for c in c_grid:
            start = time.perf_counter()
            model = _logreg(c, seed).fit(ztr, data.y_train)
            pred = model.predict(zva)
            proba = estimator_proba(model, zva, data.classes)
            row = _candidate_row(
                {"pca_dim": dim, "C": c},
                data.y_val,
                pred,
                proba,
                data.classes,
                pca_seconds + time.perf_counter() - start,
                dim,
                len(data.y_train),
            )
            candidates.append(row)
            if _is_better(row, best_row):
                best_row, best_pack = row, (pca, model, zva)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible PCA candidate")
    pca, model, zva = best_pack
    zte = pca.transform(xte).astype(np.float32)
    return MethodOutput(
        method_id="concat_pca_linear",
        display_name="Concatenation + PCA + linear classifier",
        fidelity="mandatory_control",
        source="Standard dimensionality-control baseline",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "input_dim": int(xtr.shape[1]),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "uses_all_views": True,
        },
    )


def run_uniform_layer_softvote(
    data: MultiViewData,
    *,
    c_grid: list[float],
    seed: int,
) -> MethodOutput:
    tr_views, va_views, te_views = map(
        l2_views,
        [data.train_views, data.val_views, data.test_views],
    )
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_pack = None
    for c in c_grid:
        start = time.perf_counter()
        models = [_logreg(c, seed).fit(x, data.y_train) for x in tr_views]
        val_prob = np.mean(
            [estimator_proba(m, x, data.classes) for m, x in zip(models, va_views)],
            axis=0,
        )
        pred = data.classes[np.argmax(val_prob, axis=1)]
        row = _candidate_row(
            {"C": c, "vote": "uniform_probability_mean", "n_views": len(models)},
            data.y_val,
            pred,
            val_prob,
            data.classes,
            time.perf_counter() - start,
            max(data.view_dims),
            len(data.y_train),
        )
        candidates.append(row)
        if _is_better(row, best_row):
            best_row, best_pack = row, (models, val_prob)
    assert best_pack is not None and best_row is not None
    models, val_prob = best_pack
    test_prob = np.mean(
        [estimator_proba(m, x, data.classes) for m, x in zip(models, te_views)],
        axis=0,
    )
    return MethodOutput(
        method_id="uniform_layer_softvote",
        display_name="Uniform layer soft vote",
        fidelity="mandatory_control",
        source="Standard decision-level ensemble control",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=data.classes[np.argmax(val_prob, axis=1)],
        test_pred=data.classes[np.argmax(test_prob, axis=1)],
        val_proba=val_prob,
        test_proba=test_prob,
        diagnostics={"uses_all_views": True, "learned_view_weights": False},
    )


def run_uniform_kernel_svm(
    data: MultiViewData,
    *,
    c_grid: list[float],
    seed: int,
) -> MethodOutput:
    scale = math.sqrt(len(data.train_views))
    xtr = np.concatenate([x / scale for x in l2_views(data.train_views)], axis=1)
    xva = np.concatenate([x / scale for x in l2_views(data.val_views)], axis=1)
    xte = np.concatenate([x / scale for x in l2_views(data.test_views)], axis=1)
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_model = None
    for c in c_grid:
        start = time.perf_counter()
        model = LinearSVC(C=float(c), dual="auto", random_state=seed).fit(xtr, data.y_train)
        pred = model.predict(xva)
        proba = estimator_proba(model, xva, data.classes)
        row = _candidate_row(
            {"C": c, "kernel": "uniform_mean_linear"},
            data.y_val,
            pred,
            proba,
            data.classes,
            time.perf_counter() - start,
            xtr.shape[1],
            len(data.y_train),
        )
        candidates.append(row)
        if _is_better(row, best_row):
            best_row, best_model = row, model
    assert best_model is not None and best_row is not None
    return MethodOutput(
        method_id="uniform_kernel_svm",
        display_name="Uniform multi-view linear kernel SVM",
        fidelity="mandatory_control",
        source="Uniform multiple-kernel control",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=best_model.predict(xva),
        test_pred=best_model.predict(xte),
        val_proba=estimator_proba(best_model, xva, data.classes),
        test_proba=estimator_proba(best_model, xte, data.classes),
        diagnostics={
            "uses_all_views": True,
            "kernel_equivalence": "mean of L2-normalized linear kernels",
        },
    )


def run_mlcff(
    data: MultiViewData,
    *,
    pca_dims: list[int],
    c_grid: list[float],
    seed: int,
) -> MethodOutput:
    """Fradi et al. 2021: per-layer L2, concat, PCA, LDA, linear OvO SVM."""
    xtr, xva, xte = map(
        concat_l2,
        [data.train_views, data.val_views, data.test_views],
    )
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_pack = None
    feasible = sorted(set(min(int(d), xtr.shape[0] - 1, xtr.shape[1]) for d in pca_dims))
    for pca_dim in feasible:
        if pca_dim < 2:
            continue
        pca_start = time.perf_counter()
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
        ptr = pca.fit_transform(xtr)
        pva = pca.transform(xva)
        lda_dim = min(len(data.classes) - 1, pca_dim)
        lda = LinearDiscriminantAnalysis(solver="svd", n_components=lda_dim)
        ztr = lda.fit_transform(ptr, data.y_train)
        zva = lda.transform(pva)
        transform_seconds = time.perf_counter() - pca_start
        for c in c_grid:
            start = time.perf_counter()
            model = OneVsOneClassifier(
                LinearSVC(C=float(c), dual="auto", random_state=seed)
            ).fit(ztr, data.y_train)
            pred = model.predict(zva)
            proba = estimator_proba(model, zva, data.classes)
            row = _candidate_row(
                {"pca_dim": pca_dim, "lda_dim": lda_dim, "svm_C": c},
                data.y_val,
                pred,
                proba,
                data.classes,
                transform_seconds + time.perf_counter() - start,
                ztr.shape[1],
                len(data.y_train),
            )
            candidates.append(row)
            if _is_better(row, best_row):
                best_row, best_pack = row, (pca, lda, model, zva)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible MLCFF candidate")
    pca, lda, model, zva = best_pack
    zte = lda.transform(pca.transform(xte))
    return MethodOutput(
        method_id="fradi_mlcff",
        display_name="MLCFF (Fradi et al., 2021)",
        fidelity="exact_algorithm_input_adapted",
        source="Fradi, Fradi and Dugelay, VISAPP 2021, DOI 10.5220/0010388105740581",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "paper_steps": "GAP vectors; per-layer L2; concat; PCA; LDA; linear OvO SVM",
            "input_adaptation": "Available stage endpoints replace every convolutional layer; GAP was performed during extraction.",
            "uses_all_views": True,
        },
    )


def _head2toe_group_lasso_scores(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    lam: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Head2Toe requires PyTorch") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
    )
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
    model = torch.nn.Linear(x.shape[1], n_classes, bias=True).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    generator = torch.Generator().manual_seed(seed)
    n = len(y)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=generator)
        xb = x_tensor[idx].to(target_device)
        yb = y_tensor[idx].to(target_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        group_penalty = torch.linalg.vector_norm(model.weight.T, ord=2, dim=1).sum()
        loss = functional.cross_entropy(logits, yb) + float(lam) * group_penalty
        loss.backward()
        optimizer.step()
        scheduler.step()
    scores = torch.linalg.vector_norm(model.weight.T, ord=2, dim=1).detach().cpu().numpy()
    return scores.astype(np.float32)


def run_head2toe(
    data: MultiViewData,
    *,
    lambdas: list[float],
    keep_fractions: list[float],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> MethodOutput:
    """Head2Toe selection: group L2,1 linear probe, top features, retrained head."""
    xtr, xva, xte = map(
        concat_l2,
        [data.train_views, data.val_views, data.test_views],
    )
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_pack = None
    label_to_index = {label: idx for idx, label in enumerate(data.classes)}
    encoded_train = np.asarray(
        [label_to_index[label] for label in data.y_train], dtype=np.int64
    )
    for lam in lambdas:
        start_group = time.perf_counter()
        scores = _head2toe_group_lasso_scores(
            xtr,
            encoded_train,
            len(data.classes),
            lam=lam,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        group_seconds = time.perf_counter() - start_group
        order = np.argsort(scores)[::-1]
        for fraction in keep_fractions:
            keep = min(xtr.shape[1], max(1, int(math.ceil(float(fraction) * xtr.shape[1]))))
            selected = np.sort(order[:keep])
            start = time.perf_counter()
            head = _logreg(1e6, seed, max_iter=1500).fit(xtr[:, selected], data.y_train)
            pred = head.predict(xva[:, selected])
            proba = estimator_proba(head, xva[:, selected], data.classes)
            row = _candidate_row(
                {
                    "group_lasso_lambda": lam,
                    "keep_fraction": fraction,
                    "n_selected": keep,
                    "steps": steps,
                    "final_head_C": 1e6,
                },
                data.y_val,
                pred,
                proba,
                data.classes,
                group_seconds + time.perf_counter() - start,
                keep,
                len(data.y_train),
            )
            candidates.append(row)
            if _is_better(row, best_row):
                best_row, best_pack = row, (head, selected, scores)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible Head2Toe candidate")
    head, selected, scores = best_pack
    return MethodOutput(
        method_id="head2toe",
        display_name="Head2Toe (Evci et al., 2022)",
        fidelity="exact_selection_algorithm_input_adapted",
        source="Evci et al., ICML 2022, PMLR 162:6003-6023",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=head.predict(xva[:, selected]),
        test_pred=head.predict(xte[:, selected]),
        val_proba=estimator_proba(head, xva[:, selected], data.classes),
        test_proba=estimator_proba(head, xte[:, selected], data.classes),
        diagnostics={
            "paper_steps": "unit-normalized layer features; concat; L2,1 linear probe; top feature fraction; retrained unregularized linear head",
            "input_adaptation": "Stage-level GAP vectors replace all block/window-pooled activations.",
            "selected_feature_indices": selected.tolist(),
            "selected_feature_scores": scores[selected].astype(float).tolist(),
            "uses_all_views_before_selection": True,
            "label_encoding": {
                str(label): int(index) for label, index in label_to_index.items()
            },
        },
    )


def _stratified_cap_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y))
    idx = np.arange(len(y))
    selected, _ = train_test_split(
        idx,
        train_size=max_samples,
        random_state=seed,
        stratify=y,
    )
    return np.sort(selected)


def _linear_kernels(
    train_views: list[np.ndarray],
    query_views: list[np.ndarray],
) -> list[Any]:
    import torch

    return [
        # MKLpy's margin solvers return float64 dual coefficients. Keeping the
        # Gram matrices in float64 avoids mixed-dtype products in EasyMKL.
        torch.from_numpy(np.asarray(q @ tr.T, dtype=np.float64))
        for tr, q in zip(train_views, query_views)
    ]


def _extract_mkl_weights(model: Any) -> Any:
    try:
        solution = model.solution
        if isinstance(solution, dict):
            return {
                str(k): np.asarray(v.weights).astype(float).tolist()
                for k, v in solution.items()
            }
        return np.asarray(solution.weights).astype(float).tolist()
    except Exception:
        return "not_exposed_by_wrapper"


def run_easymkl(
    data: MultiViewData,
    *,
    lambda_grid: list[float],
    c_grid: list[float],
    max_train: int,
    seed: int,
) -> MethodOutput:
    """EasyMKL with one L2-normalized linear kernel per layer."""
    try:
        from MKLpy.algorithms import EasyMKL
    except ImportError as exc:
        raise RuntimeError("EasyMKL requires MKLpy and cvxopt") from exc

    tr_views, va_views, te_views = map(
        l2_views,
        [data.train_views, data.val_views, data.test_views],
    )
    fit_idx = _stratified_cap_indices(data.y_train, max_train, seed)
    fit_views = [x[fit_idx] for x in tr_views]
    y_fit = data.y_train[fit_idx]
    train_kernels = _linear_kernels(fit_views, fit_views)
    val_kernels = _linear_kernels(fit_views, va_views)
    candidates: list[dict[str, Any]] = []
    best_row = None
    best_model = None
    for lam in lambda_grid:
        for c in c_grid:
            start = time.perf_counter()
            learner = SVC(C=float(c), kernel="precomputed")
            model = EasyMKL(
                learner=learner,
                lam=float(lam),
                solver="auto",
                multiclass_strategy="ovr",
            ).fit(train_kernels, y_fit)
            pred = np.asarray(model.predict(val_kernels))
            proba = estimator_proba(model, val_kernels, data.classes)
            row = _candidate_row(
                {
                    "lambda": lam,
                    "svm_C": c,
                    "kernel": "per_layer_l2_normalized_linear",
                    "max_train": max_train,
                },
                data.y_val,
                pred,
                proba,
                data.classes,
                time.perf_counter() - start,
                len(fit_views),
                len(fit_idx),
            )
            candidates.append(row)
            if _is_better(row, best_row):
                best_row, best_model = row, model
    if best_model is None or best_row is None:
        raise RuntimeError("No feasible EasyMKL candidate")
    test_kernels = _linear_kernels(fit_views, te_views)
    return MethodOutput(
        method_id="easymkl",
        display_name="EasyMKL (Aiolli and Donini, 2015)",
        fidelity="exact_algorithm_linear_kernel_adapter",
        source="Aiolli and Donini, Neurocomputing 2015, DOI 10.1016/j.neucom.2014.11.078",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=np.asarray(best_model.predict(val_kernels)),
        test_pred=np.asarray(best_model.predict(test_kernels)),
        val_proba=estimator_proba(best_model, val_kernels, data.classes),
        test_proba=estimator_proba(best_model, test_kernels, data.classes),
        diagnostics={
            "uses_all_views": True,
            "n_train_total": len(data.y_train),
            "n_train_mkl": len(fit_idx),
            "train_cap_is_computational_adapter": len(fit_idx) < len(data.y_train),
            "learned_kernel_weights": _extract_mkl_weights(best_model),
        },
    )


@dataclass
class ViewProjection:
    scalers: list[StandardScaler]
    pcas: list[PCA]
    weights: list[np.ndarray]

    def transform_views(self, views: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for x, scaler, pca, weight in zip(views, self.scalers, self.pcas, self.weights):
            z = pca.transform(scaler.transform(x))
            out.append(np.asarray(z @ weight, dtype=np.float32))
        return out

    def transform_mean(self, views: list[np.ndarray]) -> np.ndarray:
        return np.mean(self.transform_views(views), axis=0).astype(np.float32)


def _preprocess_views(
    train_views: list[np.ndarray],
    other_views: list[list[np.ndarray]],
    q: int,
    seed: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[StandardScaler], list[PCA]]:
    train_out: list[np.ndarray] = []
    other_out: list[list[np.ndarray]] = [[] for _ in other_views]
    scalers: list[StandardScaler] = []
    pcas: list[PCA] = []
    for view_idx, xtr in enumerate(train_views):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(xtr)
        dim = min(int(q), scaled.shape[0] - 1, scaled.shape[1])
        if dim < 2:
            raise ValueError("View is too small for projection")
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
        train_out.append(pca.fit_transform(scaled).astype(np.float64))
        for group_idx, group in enumerate(other_views):
            other_out[group_idx].append(
                pca.transform(scaler.transform(group[view_idx])).astype(np.float64)
            )
        scalers.append(scaler)
        pcas.append(pca)
    return train_out, other_out, scalers, pcas


def _fit_knn_candidates(
    ztr: np.ndarray,
    ytr: np.ndarray,
    zva: np.ndarray,
    yva: np.ndarray,
    classes: np.ndarray,
    k_grid: list[int],
    base_config: dict[str, Any],
    fit_seconds: float,
    n_train_fit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, KNeighborsClassifier | None]:
    rows = []
    best_row = None
    best_model = None
    for k in k_grid:
        if k >= len(ytr):
            continue
        start = time.perf_counter()
        model = KNeighborsClassifier(n_neighbors=int(k), weights="distance", n_jobs=-1).fit(ztr, ytr)
        pred = model.predict(zva)
        proba = estimator_proba(model, zva, classes)
        config = {**base_config, "k": int(k), "knn_weights": "distance"}
        row = _candidate_row(
            config,
            yva,
            pred,
            proba,
            classes,
            fit_seconds + time.perf_counter() - start,
            ztr.shape[1],
            n_train_fit,
        )
        rows.append(row)
        if _is_better(row, best_row):
            best_row, best_model = row, model
    return rows, best_row, best_model


def _fit_gcca_projection(
    train_views: list[np.ndarray],
    q: int,
    latent_dim: int,
    ridge: float,
    seed: int,
) -> tuple[ViewProjection, np.ndarray]:
    z_views, _, scalers, pcas = _preprocess_views(train_views, [], q, seed)
    q_bases = [np.linalg.qr(z, mode="reduced")[0] for z in z_views]
    stacked = np.concatenate(q_bases, axis=1)
    u, _, _ = np.linalg.svd(stacked, full_matrices=False)
    dim = min(int(latent_dim), u.shape[1])
    shared = u[:, :dim]
    weights = []
    for z in z_views:
        gram = z.T @ z
        scale = max(float(np.trace(gram) / max(gram.shape[0], 1)), EPS)
        weights.append(
            linalg.solve(
                gram + float(ridge) * scale * np.eye(gram.shape[0]),
                z.T @ shared,
                assume_a="pos",
            )
        )
    projection = ViewProjection(scalers, pcas, weights)
    return projection, projection.transform_mean(train_views)


def run_maxvar_gcca(
    data: MultiViewData,
    *,
    q_grid: list[int],
    latent_dims: list[int],
    ridge_grid: list[float],
    k_grid: list[int],
    seed: int,
) -> MethodOutput:
    candidates = []
    best_row = None
    best_pack = None
    for q in q_grid:
        for latent in latent_dims:
            for ridge in ridge_grid:
                start = time.perf_counter()
                projection, ztr = _fit_gcca_projection(
                    data.train_views, q, latent, ridge, seed
                )
                zva = projection.transform_mean(data.val_views)
                fit_seconds = time.perf_counter() - start
                rows, local_best, model = _fit_knn_candidates(
                    ztr,
                    data.y_train,
                    zva,
                    data.y_val,
                    data.classes,
                    k_grid,
                    {"view_pca_dim": q, "latent_dim": latent, "ridge": ridge},
                    fit_seconds,
                    len(data.y_train),
                )
                candidates.extend(rows)
                if local_best is not None and _is_better(local_best, best_row):
                    best_row, best_pack = local_best, (projection, model, zva, ztr)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible MAXVAR-GCCA candidate")
    projection, model, zva, _ = best_pack
    zte = projection.transform_mean(data.test_views)
    return MethodOutput(
        method_id="maxvar_gcca",
        display_name="Regularized MAXVAR-GCCA",
        fidelity="exact_objective_with_out_of_sample_ridge_maps",
        source="MAXVAR generalized CCA; Carroll 1968 / Kettenring 1971",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "uses_all_views": True,
            "fusion": "mean of per-view projections into the learned common space",
        },
    )


def _class_scatter(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = z.mean(axis=0)
    between = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    within = np.zeros_like(between)
    means = []
    for label in np.unique(y):
        group = z[y == label]
        class_mean = group.mean(axis=0)
        means.append(class_mean)
        centered = group - class_mean
        within += centered.T @ centered
        delta = class_mean - mean
        between += len(group) * np.outer(delta, delta)
    return between, within, np.column_stack(means)


def _generalized_projection(
    numerator: np.ndarray,
    denominator: np.ndarray,
    latent_dim: int,
    regularization: float,
) -> np.ndarray:
    numerator = (numerator + numerator.T) * 0.5
    denominator = (denominator + denominator.T) * 0.5
    scale = max(float(np.trace(denominator) / max(denominator.shape[0], 1)), EPS)
    regularized = denominator + float(regularization) * scale * np.eye(denominator.shape[0])
    count = min(int(latent_dim), numerator.shape[0])
    values, vectors = linalg.eigh(
        numerator,
        regularized,
        subset_by_index=[numerator.shape[0] - count, numerator.shape[0] - 1],
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    return np.asarray(vectors[:, order], dtype=np.float64)


def _fit_gmlda_projection(
    train_views: list[np.ndarray],
    y: np.ndarray,
    q: int,
    latent_dim: int,
    alpha: float,
    regularization: float,
    seed: int,
) -> tuple[ViewProjection, np.ndarray]:
    z_views, _, scalers, pcas = _preprocess_views(train_views, [], q, seed)
    blocks_a = []
    blocks_b = []
    exemplars = []
    for z in z_views:
        between, within, means = _class_scatter(z, y)
        blocks_a.append(between)
        blocks_b.append(within)
        exemplars.append(means)
    dims = [z.shape[1] for z in z_views]
    offsets = np.cumsum([0] + dims)
    total = offsets[-1]
    numerator = np.zeros((total, total), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    trace0 = max(float(np.trace(blocks_b[0])), EPS)
    for i in range(len(z_views)):
        si = slice(offsets[i], offsets[i + 1])
        numerator[si, si] = blocks_a[i]
        gamma = trace0 / max(float(np.trace(blocks_b[i])), EPS)
        denominator[si, si] = gamma * blocks_b[i]
        for j in range(i + 1, len(z_views)):
            sj = slice(offsets[j], offsets[j + 1])
            cross = float(alpha) * exemplars[i] @ exemplars[j].T
            numerator[si, sj] = cross
            numerator[sj, si] = cross.T
    eigenvectors = _generalized_projection(
        numerator, denominator, latent_dim, regularization
    )
    weights = [
        eigenvectors[offsets[i] : offsets[i + 1]]
        for i in range(len(z_views))
    ]
    projection = ViewProjection(scalers, pcas, weights)
    return projection, projection.transform_mean(train_views)


def run_gmlda(
    data: MultiViewData,
    *,
    q_grid: list[int],
    latent_dims: list[int],
    alpha_grid: list[float],
    regularization_grid: list[float],
    k_grid: list[int],
    seed: int,
) -> MethodOutput:
    candidates = []
    best_row = None
    best_pack = None
    for q in q_grid:
        for latent in latent_dims:
            for alpha in alpha_grid:
                for reg in regularization_grid:
                    start = time.perf_counter()
                    projection, ztr = _fit_gmlda_projection(
                        data.train_views,
                        data.y_train,
                        q,
                        latent,
                        alpha,
                        reg,
                        seed,
                    )
                    zva = projection.transform_mean(data.val_views)
                    rows, local_best, model = _fit_knn_candidates(
                        ztr,
                        data.y_train,
                        zva,
                        data.y_val,
                        data.classes,
                        k_grid,
                        {
                            "view_pca_dim": q,
                            "latent_dim": latent,
                            "alpha": alpha,
                            "regularization": reg,
                        },
                        time.perf_counter() - start,
                        len(data.y_train),
                    )
                    candidates.extend(rows)
                    if local_best is not None and _is_better(local_best, best_row):
                        best_row, best_pack = local_best, (projection, model, zva)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible GMLDA candidate")
    projection, model, zva = best_pack
    zte = projection.transform_mean(data.test_views)
    return MethodOutput(
        method_id="gmlda",
        display_name="GMLDA (Sharma et al., 2012)",
        fidelity="exact_gma_gmlda_projection_input_adapted",
        source="Sharma et al., CVPR 2012, DOI 10.1109/CVPR.2012.6247923",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "uses_all_views": True,
            "paper_equations": "GMA Eq. 7-10 with LDA A/B matrices and class-mean exemplars",
            "input_adaptation": "CNN layers are paired views; projected views are averaged for same-image classification.",
        },
    )


def _fit_mvda_projection(
    train_views: list[np.ndarray],
    y: np.ndarray,
    q: int,
    latent_dim: int,
    regularization: float,
    seed: int,
) -> tuple[ViewProjection, np.ndarray]:
    z_views, _, scalers, pcas = _preprocess_views(train_views, [], q, seed)
    labels = np.unique(y)
    n_views = len(z_views)
    n_samples = len(y)
    dims = [z.shape[1] for z in z_views]
    offsets = np.cumsum([0] + dims)
    total = offsets[-1]
    within = np.zeros((total, total), dtype=np.float64)
    between = np.zeros_like(within)
    class_means = [
        {label: z[y == label].mean(axis=0) for label in labels}
        for z in z_views
    ]
    weighted_sums = [
        sum(int(np.sum(y == label)) * class_means[j][label] for label in labels)
        for j in range(n_views)
    ]
    total_multiview_n = n_views * n_samples
    for j in range(n_views):
        sj = slice(offsets[j], offsets[j + 1])
        for r in range(n_views):
            sr = slice(offsets[r], offsets[r + 1])
            class_cross = np.zeros((dims[j], dims[r]), dtype=np.float64)
            for label in labels:
                count = int(np.sum(y == label))
                class_cross += (count / n_views) * np.outer(
                    class_means[j][label], class_means[r][label]
                )
            between[sj, sr] = class_cross - np.outer(
                weighted_sums[j], weighted_sums[r]
            ) / total_multiview_n
            if j == r:
                within[sj, sr] = z_views[j].T @ z_views[j] - class_cross
            else:
                within[sj, sr] = -class_cross
    eigenvectors = _generalized_projection(
        between, within, latent_dim, regularization
    )
    weights = [
        eigenvectors[offsets[i] : offsets[i + 1]]
        for i in range(n_views)
    ]
    projection = ViewProjection(scalers, pcas, weights)
    return projection, projection.transform_mean(train_views)


def run_mvda(
    data: MultiViewData,
    *,
    q_grid: list[int],
    latent_dims: list[int],
    regularization_grid: list[float],
    k_grid: list[int],
    seed: int,
) -> MethodOutput:
    candidates = []
    best_row = None
    best_pack = None
    for q in q_grid:
        for latent in latent_dims:
            for reg in regularization_grid:
                start = time.perf_counter()
                projection, ztr = _fit_mvda_projection(
                    data.train_views,
                    data.y_train,
                    q,
                    latent,
                    reg,
                    seed,
                )
                zva = projection.transform_mean(data.val_views)
                rows, local_best, model = _fit_knn_candidates(
                    ztr,
                    data.y_train,
                    zva,
                    data.y_val,
                    data.classes,
                    k_grid,
                    {
                        "view_pca_dim": q,
                        "latent_dim": latent,
                        "regularization": reg,
                    },
                    time.perf_counter() - start,
                    len(data.y_train),
                )
                candidates.extend(rows)
                if local_best is not None and _is_better(local_best, best_row):
                    best_row, best_pack = local_best, (projection, model, zva)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible MvDA candidate")
    projection, model, zva = best_pack
    zte = projection.transform_mean(data.test_views)
    return MethodOutput(
        method_id="mvda",
        display_name="MvDA (Kan et al., 2016)",
        fidelity="exact_scatter_formulation_input_adapted",
        source="Kan et al., IEEE TPAMI 2016, DOI 10.1109/TPAMI.2015.2435740",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "uses_all_views": True,
            "paper_equations": "MvDA Eq. 7-12, generalized eigenproblem D w = lambda S w",
            "input_adaptation": "CNN layers are paired views; projected views are averaged for same-image classification.",
        },
    )


def run_nca_knn(
    data: MultiViewData,
    *,
    pca_dims: list[int],
    nca_dims: list[int],
    k_grid: list[int],
    max_fit_samples: int,
    max_iter: int,
    seed: int,
) -> MethodOutput:
    xtr, xva, xte = map(
        concat_l2,
        [data.train_views, data.val_views, data.test_views],
    )
    fit_idx = _stratified_cap_indices(data.y_train, max_fit_samples, seed)
    candidates = []
    best_row = None
    best_pack = None
    for pca_dim in pca_dims:
        dim = min(int(pca_dim), xtr.shape[0] - 1, xtr.shape[1])
        if dim < 2:
            continue
        pca_start = time.perf_counter()
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
        ptr = pca.fit_transform(xtr)
        pva = pca.transform(xva)
        pca_seconds = time.perf_counter() - pca_start
        for nca_dim in nca_dims:
            out_dim = min(int(nca_dim), dim)
            start = time.perf_counter()
            nca = NeighborhoodComponentsAnalysis(
                n_components=out_dim,
                max_iter=max_iter,
                random_state=seed,
                init="pca",
            ).fit(ptr[fit_idx], data.y_train[fit_idx])
            ztr = nca.transform(ptr).astype(np.float32)
            zva = nca.transform(pva).astype(np.float32)
            rows, local_best, model = _fit_knn_candidates(
                ztr,
                data.y_train,
                zva,
                data.y_val,
                data.classes,
                k_grid,
                {
                    "pca_dim": dim,
                    "nca_dim": out_dim,
                    "nca_max_iter": max_iter,
                    "nca_max_fit_samples": max_fit_samples,
                },
                pca_seconds + time.perf_counter() - start,
                len(fit_idx),
            )
            candidates.extend(rows)
            if local_best is not None and _is_better(local_best, best_row):
                best_row, best_pack = local_best, (pca, nca, model, zva)
    if best_pack is None or best_row is None:
        raise RuntimeError("No feasible NCA candidate")
    pca, nca, model, zva = best_pack
    zte = nca.transform(pca.transform(xte)).astype(np.float32)
    return MethodOutput(
        method_id="concat_nca_knn",
        display_name="Concatenation + NCA + kNN",
        fidelity="established_composed_baseline",
        source="Goldberger et al., NeurIPS 2004; NCA applied after fixed all-layer concatenation",
        candidates=candidates,
        selected_config=json.loads(best_row["config_json"]),
        val_pred=model.predict(zva),
        test_pred=model.predict(zte),
        val_proba=estimator_proba(model, zva, data.classes),
        test_proba=estimator_proba(model, zte, data.classes),
        diagnostics={
            "uses_all_views": True,
            "n_train_total": len(data.y_train),
            "n_train_nca_fit": len(fit_idx),
            "train_cap_is_computational_adapter": len(fit_idx) < len(data.y_train),
        },
    )


METHOD_FUNCTIONS: dict[str, Callable[..., MethodOutput]] = {
    "raw_concat_linear": run_raw_concat_linear,
    "concat_pca_linear": run_concat_pca_linear,
    "uniform_layer_softvote": run_uniform_layer_softvote,
    "uniform_kernel_svm": run_uniform_kernel_svm,
    "fradi_mlcff": run_mlcff,
    "head2toe": run_head2toe,
    "easymkl": run_easymkl,
    "maxvar_gcca": run_maxvar_gcca,
    "gmlda": run_gmlda,
    "mvda": run_mvda,
    "concat_nca_knn": run_nca_knn,
}
