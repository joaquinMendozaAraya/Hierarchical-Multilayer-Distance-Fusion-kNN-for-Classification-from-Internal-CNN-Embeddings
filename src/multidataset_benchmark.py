"""Multi-dataset embedding benchmark utilities for Tesis_2026.

This module works on already extracted embeddings from the Colab suite.  It is
designed to be restartable: every expensive stage writes CSV/NPZ artifacts and
can skip finished work on later runs.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


EXPECTED_DIMS = [64, 128, 256, 512, 1280]
SPLITS = ["train", "val", "test"]


@dataclass
class EmbeddingProfile:
    dataset_id: str
    run_id: str
    profile_name: str
    profile_path: str
    n_classes: int
    classes: list[str]
    available_dims: list[int]
    missing_dims: list[int]
    split_counts: dict[str, int]
    split_class_counts: dict[str, dict[str, int]]
    has_checkpoint: bool
    has_existing_baselines: bool
    has_existing_gmm: bool
    has_existing_knn: bool
    usable: bool
    issues: list[str]

    @property
    def key(self) -> str:
        return f"{self.dataset_id}__{self.run_id}__{self.profile_name}"

    @property
    def path(self) -> Path:
        return Path(self.profile_path)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def macro_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def softmax_np(x, axis=1):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)


def entropy_probs(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def top_margin(p):
    p = np.asarray(p, dtype=np.float32)
    if p.shape[1] == 1:
        return np.ones(p.shape[0], dtype=np.float32)
    top2 = np.sort(p, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def maybe_read_classes(profile_path: Path, labels=None) -> list[str]:
    candidates = [
        profile_path / "splits" / "split_info.json",
        profile_path.parents[1] / "run_config.json" if len(profile_path.parents) > 1 else None,
        profile_path.parents[2] / "run_config.json" if len(profile_path.parents) > 2 else None,
    ]
    for path in candidates:
        if path is None:
            continue
        info = load_json(path, default=None)
        if isinstance(info, dict):
            classes = info.get("classes")
            if classes:
                return [str(c) for c in classes]
            base = info.get("base_split_info", {})
            if isinstance(base, dict) and base.get("classes"):
                return [str(c) for c in base["classes"]]
    if labels is not None:
        return [str(i) for i in sorted(np.unique(labels).tolist())]
    return []


def infer_dataset_id(profile_path: Path) -> str:
    text = str(profile_path).replace("\\", "/").lower()
    for known in ["brain_tumor_mri_44c", "brain_tumor_mri_4c", "ham10000_skin_7c"]:
        if known in text:
            return known
    try:
        artifacts_idx = [part.lower() for part in profile_path.parts].index("artifacts")
        return safe_name(profile_path.parts[artifacts_idx + 1])
    except Exception:
        return safe_name(profile_path.parents[2].name if len(profile_path.parents) > 2 else profile_path.name)


def discover_embedding_profiles(
    search_roots: list[str | Path],
    expected_dims: list[int] | None = None,
) -> list[EmbeddingProfile]:
    expected_dims = expected_dims or EXPECTED_DIMS
    profiles: list[EmbeddingProfile] = []
    seen = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for emb_norm in root.rglob("embeddings/multicapa_norm"):
            profile_path = emb_norm.parents[1]
            if profile_path in seen:
                continue
            seen.add(profile_path)
            issues = []
            split_counts: dict[str, int] = {}
            split_class_counts: dict[str, dict[str, int]] = {}
            dims_by_split = {}
            labels_by_split = {}
            for split in SPLITS:
                split_dir = emb_norm / split
                dims = sorted(
                    int(p.stem.split("_")[-1])
                    for p in split_dir.glob("z_dim_*.npy")
                    if p.stem.split("_")[-1].isdigit()
                )
                dims_by_split[split] = dims
                labels_path = split_dir / "labels.npy"
                if labels_path.exists():
                    y = np.load(labels_path)
                    labels_by_split[split] = y
                    split_counts[split] = int(len(y))
                    vc = pd.Series(y).value_counts().sort_index()
                    split_class_counts[split] = {str(int(k)): int(v) for k, v in vc.items()}
                else:
                    issues.append(f"missing labels.npy in {split}")
                    split_counts[split] = 0
                    split_class_counts[split] = {}
                for dim in dims:
                    z_path = split_dir / f"z_dim_{dim}.npy"
                    if labels_path.exists() and z_path.exists():
                        x = np.load(z_path, mmap_mode="r")
                        if x.shape[0] != split_counts[split]:
                            issues.append(f"shape mismatch {split} dim {dim}: {x.shape[0]} vs {split_counts[split]}")
            available_dims = sorted(set(dims_by_split["train"]) & set(dims_by_split["val"]) & set(dims_by_split["test"]))
            missing_dims = [d for d in expected_dims if d not in available_dims]
            if missing_dims:
                issues.append(f"missing common dims: {missing_dims}")
            if any(split_counts.get(s, 0) <= 0 for s in SPLITS):
                issues.append("one or more splits are empty")
            y_train = labels_by_split.get("train")
            classes = maybe_read_classes(profile_path, y_train)
            n_classes = len(classes) if classes else int(len(np.unique(y_train))) if y_train is not None else 0
            if y_train is not None and n_classes != len(np.unique(y_train)):
                # Split_info classes may contain names for all labels; this is fine, but record it.
                pass
            has_checkpoint = (profile_path / "checkpoints" / "triplet_best.pt").exists()
            has_existing_baselines = (profile_path / "results" / "baselines" / "benchmark_by_dim.csv").exists()
            has_existing_gmm = (profile_path / "evidence" / "gmm_summaries" / "gmm_metrics_by_dim.csv").exists()
            has_existing_knn = (profile_path / "evidence" / "knn_summaries" / "knn_metrics_by_dim_k.csv").exists()
            dataset_id = infer_dataset_id(profile_path)
            run_id = profile_path.parents[1].name if profile_path.parent.name == "profiles" else profile_path.parent.name
            profile_name = profile_path.name
            usable = len(available_dims) > 0 and all(split_counts.get(s, 0) > 0 for s in SPLITS)
            profiles.append(
                EmbeddingProfile(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    profile_name=profile_name,
                    profile_path=str(profile_path.resolve()),
                    n_classes=n_classes,
                    classes=classes,
                    available_dims=available_dims,
                    missing_dims=missing_dims,
                    split_counts=split_counts,
                    split_class_counts=split_class_counts,
                    has_checkpoint=has_checkpoint,
                    has_existing_baselines=has_existing_baselines,
                    has_existing_gmm=has_existing_gmm,
                    has_existing_knn=has_existing_knn,
                    usable=usable,
                    issues=issues,
                )
            )
    profiles.sort(key=lambda p: (p.dataset_id, p.run_id, p.profile_name))
    return profiles


def profiles_to_frame(profiles: list[EmbeddingProfile]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append(
            {
                "dataset_id": p.dataset_id,
                "run_id": p.run_id,
                "profile": p.profile_name,
                "usable": p.usable,
                "n_classes": p.n_classes,
                "dims": ",".join(map(str, p.available_dims)),
                "missing_dims": ",".join(map(str, p.missing_dims)),
                "n_train": p.split_counts.get("train", 0),
                "n_val": p.split_counts.get("val", 0),
                "n_test": p.split_counts.get("test", 0),
                "has_checkpoint": p.has_checkpoint,
                "has_existing_baselines": p.has_existing_baselines,
                "has_existing_gmm": p.has_existing_gmm,
                "has_existing_knn": p.has_existing_knn,
                "issues": " | ".join(p.issues),
                "profile_path": p.profile_path,
            }
        )
    return pd.DataFrame(rows)


def write_profile_registry(profiles: list[EmbeddingProfile], output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = profiles_to_frame(profiles)
    df.to_csv(output_dir / "input_profile_registry.csv", index=False)
    save_json([asdict(p) for p in profiles], output_dir / "input_profile_registry.json")
    return df


def profile_output_dir(output_root: str | Path, profile: EmbeddingProfile) -> Path:
    out = Path(output_root) / "profiles" / safe_name(profile.dataset_id) / safe_name(profile.run_id) / safe_name(profile.profile_name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_split(profile: EmbeddingProfile, split: str, dim: int | None = None):
    base = profile.path / "embeddings" / "multicapa_norm" / split
    y = np.load(base / "labels.npy")
    if dim is None:
        return y
    x = np.load(base / f"z_dim_{dim}.npy")
    return x.astype(np.float32), y.astype(int)


def ensure_prediction_dir(out_dir: Path) -> Path:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir


def existing_result_keys(path: Path, key_cols: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(tuple(row[col] for col in key_cols) for _, row in df.iterrows())


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def log_error(output_root: Path, profile: EmbeddingProfile, stage: str, exc: BaseException) -> None:
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_id": profile.dataset_id,
        "run_id": profile.run_id,
        "profile": profile.profile_name,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    append_rows(output_root / "error_log.csv", [row])


def fit_predict_classifier(model_name: str, params: dict, x_train, y_train, x_val, x_test):
    scaler = None
    use_scaler = model_name in {"logreg", "linear_svm"}
    if use_scaler:
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x_train)
        xva = scaler.transform(x_val)
        xte = scaler.transform(x_test)
    else:
        xtr, xva, xte = x_train, x_val, x_test

    if model_name == "logreg":
        clf = LogisticRegression(**params)
    elif model_name == "linear_svm":
        clf = LinearSVC(**params)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_name == "gaussian_nb":
        clf = GaussianNB(**params)
    elif model_name == "knn":
        clf = KNeighborsClassifier(**params)
    else:
        raise ValueError(model_name)

    clf.fit(xtr, y_train)
    return clf.predict(x_val), clf.predict(x_test), {"model": clf, "scaler": scaler}


def run_baseline_benchmark(
    profile: EmbeddingProfile,
    output_root: str | Path,
    dims: list[int] | None = None,
    force: bool = False,
    random_state: int = 42,
):
    dims = dims or profile.available_dims
    out_dir = profile_output_dir(output_root, profile)
    result_path = out_dir / "tables" / "baseline_results.csv"
    pred_dir = ensure_prediction_dir(out_dir)
    done = existing_result_keys(result_path, ["method", "dimension"]) if not force else set()

    model_space = {
        "knn": [{"n_neighbors": k, "metric": "euclidean"} for k in [1, 2, 3, 5, 7, 10, 15, 25, 50]],
        "logreg": [
            {"C": c, "max_iter": 5000, "class_weight": "balanced", "solver": "lbfgs"}
            for c in [0.01, 0.1, 1.0, 10.0]
        ],
        "linear_svm": [
            {"C": c, "max_iter": 8000, "class_weight": "balanced", "random_state": random_state}
            for c in [0.1, 1.0, 10.0]
        ],
        "random_forest": [
            {"n_estimators": 350, "max_depth": None, "class_weight": "balanced_subsample", "random_state": random_state, "n_jobs": -1}
        ],
        "gaussian_nb": [{"var_smoothing": v} for v in [1e-9, 1e-8, 1e-7]],
    }

    rows = []
    grid_rows = []
    for dim in dims:
        x_train, y_train = load_split(profile, "train", dim)
        x_val, y_val = load_split(profile, "val", dim)
        x_test, y_test = load_split(profile, "test", dim)
        for model_name, params_list in model_space.items():
            method = f"{model_name}_dim{dim}"
            if (method, dim) in done:
                continue
            valid_params = []
            for params in params_list:
                if model_name == "knn" and params["n_neighbors"] >= len(y_train):
                    continue
                valid_params.append(params)
            best = None
            t0 = time.time()
            for params in valid_params:
                pred_val, pred_test, pack = fit_predict_classifier(
                    model_name, params, x_train, y_train, x_val, x_test
                )
                val_m = macro_metrics(y_val, pred_val)
                grid_rows.append(
                    {
                        "dataset_id": profile.dataset_id,
                        "run_id": profile.run_id,
                        "profile": profile.profile_name,
                        "dimension": dim,
                        "model": model_name,
                        "params": json.dumps(params),
                        **{f"val_{k}": v for k, v in val_m.items()},
                    }
                )
                if best is None or val_m["f1_macro"] > best["val_metrics"]["f1_macro"]:
                    best = {"params": params, "pred_val": pred_val, "pred_test": pred_test, "val_metrics": val_m, "pack": pack}
            if best is None:
                continue
            test_m = macro_metrics(y_test, best["pred_test"])
            rows.append(
                {
                    "dataset_id": profile.dataset_id,
                    "run_id": profile.run_id,
                    "profile": profile.profile_name,
                    "method_family": model_name,
                    "method": method,
                    "dimension": dim,
                    "uses_knn": model_name == "knn",
                    "uses_gmm": False,
                    "uses_all_dims": False,
                    "best_params": json.dumps(best["params"]),
                    "seconds": time.time() - t0,
                    **{f"val_{k}": v for k, v in best["val_metrics"].items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
            )
            np.savez_compressed(
                pred_dir / f"{method}.npz",
                y_val=y_val,
                pred_val=best["pred_val"],
                y_test=y_test,
                pred_test=best["pred_test"],
                classes=np.array(profile.classes, dtype=object),
            )

    append_rows(result_path, rows)
    append_rows(out_dir / "tables" / "baseline_validation_grid.csv", grid_rows)
    return pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()


def gmm_variants_for_dim(dim: int, x_train_shape, pca_candidates=(64, 128)):
    variants = [{"name": "direct", "use_pca": False, "pca_dim": None}]
    n_samples, n_features = x_train_shape
    for pca_dim in pca_candidates:
        if pca_dim < dim and pca_dim < n_samples and pca_dim < n_features:
            variants.append({"name": f"pca{pca_dim}", "use_pca": True, "pca_dim": pca_dim})
    return variants


def fit_gmm_pack(x_train, y_train, dim: int, variant: dict, random_state=42):
    pca = None
    x_work = x_train
    if variant["use_pca"]:
        pca = PCA(n_components=variant["pca_dim"], random_state=random_state)
        x_work = pca.fit_transform(x_train).astype(np.float32)
    models = {}
    bic_rows = []
    for class_id in sorted(np.unique(y_train)):
        x_class = x_work[y_train == class_id]
        candidates = [k for k in [1, 2, 3] if k <= max(1, len(x_class) // 25)]
        if not candidates:
            candidates = [1]
        best_model = None
        best_bic = np.inf
        best_k = None
        for k in candidates:
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-5,
                max_iter=250,
                n_init=1,
                random_state=random_state,
            )
            model.fit(x_class)
            bic = model.bic(x_class)
            bic_rows.append(
                {
                    "dimension": dim,
                    "variant": variant["name"],
                    "class_id": int(class_id),
                    "k": int(k),
                    "bic": float(bic),
                    "aic": float(model.aic(x_class)),
                    "converged": bool(model.converged_),
                    "n_iter": int(model.n_iter_),
                }
            )
            if bic < best_bic:
                best_model = model
                best_bic = bic
                best_k = k
        models[int(class_id)] = best_model
        for row in bic_rows:
            if row["dimension"] == dim and row["variant"] == variant["name"] and row["class_id"] == int(class_id):
                row["best_k_selected"] = int(best_k)
    return {"dim": dim, "variant": variant, "pca": pca, "models_by_class": models}, bic_rows


def predict_gmm_pack(x, pack, y_train):
    x_work = pack["pca"].transform(x).astype(np.float32) if pack["pca"] is not None else x
    class_ids = sorted(pack["models_by_class"].keys())
    loglik = np.column_stack([pack["models_by_class"][c].score_samples(x_work) for c in class_ids]).astype(np.float32)
    counts = np.array([(y_train == c).sum() for c in class_ids], dtype=np.float32)
    priors = counts / np.clip(counts.sum(), 1e-12, None)
    posterior = softmax_np(loglik + np.log(np.clip(priors, 1e-12, None))[None, :], axis=1).astype(np.float32)
    pred = np.array([class_ids[i] for i in np.argmax(posterior, axis=1)], dtype=int)
    return {"loglik": loglik, "posterior": posterior, "pred": pred}


def run_gmm_benchmark(
    profile: EmbeddingProfile,
    output_root: str | Path,
    dims: list[int] | None = None,
    force: bool = False,
    random_state: int = 42,
):
    dims = dims or profile.available_dims
    out_dir = profile_output_dir(output_root, profile)
    result_path = out_dir / "tables" / "gmm_results.csv"
    pred_dir = ensure_prediction_dir(out_dir)
    model_dir = out_dir / "models" / "gmm"
    model_dir.mkdir(parents=True, exist_ok=True)
    done = existing_result_keys(result_path, ["method", "dimension"]) if not force else set()
    rows = []
    bic_rows_all = []
    packs: dict[int, dict] = {}

    for dim in dims:
        method = f"gmm_dim{dim}"
        if (method, dim) in done:
            pack_path = model_dir / f"gmm_dim{dim}.joblib"
            if pack_path.exists():
                packs[dim] = joblib.load(pack_path)
            continue
        x_train, y_train = load_split(profile, "train", dim)
        x_val, y_val = load_split(profile, "val", dim)
        x_test, y_test = load_split(profile, "test", dim)
        best = None
        t0 = time.time()
        for variant in gmm_variants_for_dim(dim, x_train.shape):
            pack, bic_rows = fit_gmm_pack(x_train, y_train, dim, variant, random_state=random_state)
            bic_rows_all.extend(bic_rows)
            pred_val = predict_gmm_pack(x_val, pack, y_train)["pred"]
            val_m = macro_metrics(y_val, pred_val)
            if best is None or val_m["f1_macro"] > best["val_metrics"]["f1_macro"]:
                best = {"pack": pack, "val_metrics": val_m, "variant": variant}
        out_val = predict_gmm_pack(x_val, best["pack"], y_train)
        out_test = predict_gmm_pack(x_test, best["pack"], y_train)
        test_m = macro_metrics(y_test, out_test["pred"])
        rows.append(
            {
                "dataset_id": profile.dataset_id,
                "run_id": profile.run_id,
                "profile": profile.profile_name,
                "method_family": "gmm",
                "method": method,
                "dimension": dim,
                "uses_knn": False,
                "uses_gmm": True,
                "uses_all_dims": False,
                "best_params": json.dumps(best["variant"]),
                "seconds": time.time() - t0,
                **{f"val_{k}": v for k, v in best["val_metrics"].items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        joblib.dump(best["pack"], model_dir / f"gmm_dim{dim}.joblib")
        packs[dim] = best["pack"]
        np.savez_compressed(
            pred_dir / f"{method}.npz",
            y_val=y_val,
            pred_val=out_val["pred"],
            y_test=y_test,
            pred_test=out_test["pred"],
            val_posterior=out_val["posterior"],
            test_posterior=out_test["posterior"],
            classes=np.array(profile.classes, dtype=object),
        )

    append_rows(result_path, rows)
    append_rows(out_dir / "tables" / "gmm_bic_grid.csv", bic_rows_all)
    return (pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()), packs


def neighbor_query(nn, x_query, kmax, is_train):
    n_req = kmax + 1 if is_train else kmax
    dist, idx = nn.kneighbors(x_query, n_neighbors=n_req)
    if not is_train:
        return dist[:, :kmax].astype(np.float32), idx[:, :kmax].astype(np.int32)
    clean_dist = []
    clean_idx = []
    for i in range(idx.shape[0]):
        keep = [(d, j) for d, j in zip(dist[i], idx[i]) if j != i][:kmax]
        clean_dist.append([d for d, _ in keep])
        clean_idx.append([j for _, j in keep])
    return np.asarray(clean_dist, dtype=np.float32), np.asarray(clean_idx, dtype=np.int32)


def hist_from_neighbor_labels(labels, n_classes):
    hist = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    for c in range(n_classes):
        hist[:, c] = (labels == c).mean(axis=1)
    return hist


def build_router_features(
    x_train,
    y_train,
    x_query,
    gmm_post_train,
    gmm_post_query,
    k_values: list[int],
    is_train: bool,
):
    n_classes = int(len(np.unique(y_train)))
    kmax = max(k_values)
    nn = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean")
    nn.fit(x_train)
    dist, idx = neighbor_query(nn, x_query, kmax, is_train=is_train)
    blocks = []
    names = []
    for k in k_values:
        labels = y_train[idx[:, :k]]
        hist = hist_from_neighbor_labels(labels, n_classes)
        neigh_post = gmm_post_train[idx[:, :k]].mean(axis=1).astype(np.float32)
        d = dist[:, :k]
        stats = np.column_stack(
            [
                d[:, 0],
                d.mean(axis=1),
                d.std(axis=1),
                d.max(axis=1),
                entropy_probs(hist),
                top_margin(hist),
                hist.max(axis=1),
                entropy_probs(neigh_post),
                top_margin(neigh_post),
                (np.argmax(hist, axis=1) == np.argmax(gmm_post_query, axis=1)).astype(np.float32),
                np.abs(hist - gmm_post_query).sum(axis=1),
                np.abs(neigh_post - gmm_post_query).sum(axis=1),
            ]
        ).astype(np.float32)
        focused_hist = (hist * gmm_post_query).astype(np.float32)
        focused_post = (neigh_post * gmm_post_query).astype(np.float32)
        blocks.extend([stats, hist, neigh_post, focused_hist, focused_post])
        names.extend([f"k{k}_stats", f"k{k}_hist", f"k{k}_neigh_post", f"k{k}_focused_hist", f"k{k}_focused_post"])
    gmm_stats = np.column_stack(
        [
            entropy_probs(gmm_post_query),
            top_margin(gmm_post_query),
            gmm_post_query.max(axis=1),
        ]
    ).astype(np.float32)
    blocks.extend([gmm_post_query.astype(np.float32), gmm_stats])
    names.extend(["gmm_posterior", "gmm_stats"])
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def train_router_meta_classifier(x_train_f, y_train, x_val_f, y_val, random_state=42):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train_f)
    xva = scaler.transform(x_val_f)
    best = None
    for c in [0.01, 0.1, 1.0, 3.0, 10.0]:
        clf = LogisticRegression(
            C=c,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(xtr, y_train)
        pred_val = clf.predict(xva)
        val_m = macro_metrics(y_val, pred_val)
        if best is None or val_m["f1_macro"] > best["val_metrics"]["f1_macro"]:
            best = {"clf": clf, "scaler": scaler, "C": c, "val_metrics": val_m, "pred_val": pred_val}
    return best


def run_gmm_focused_knn_router(
    profile: EmbeddingProfile,
    output_root: str | Path,
    dims: list[int] | None = None,
    force: bool = False,
    random_state: int = 42,
):
    dims = dims or profile.available_dims
    out_dir = profile_output_dir(output_root, profile)
    result_path = out_dir / "tables" / "proposed_router_results.csv"
    pred_dir = ensure_prediction_dir(out_dir)
    feature_dir = out_dir / "router_features"
    model_dir = out_dir / "models" / "gmm_focused_knn_router"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    done = existing_result_keys(result_path, ["method", "dimension"]) if not force else set()
    rows = []
    all_dim_features = {"train": [], "val": [], "test": []}
    all_dim_feature_labels = None

    y_train_ref = None
    y_val_ref = None
    y_test_ref = None
    for dim in dims:
        x_train, y_train = load_split(profile, "train", dim)
        x_val, y_val = load_split(profile, "val", dim)
        x_test, y_test = load_split(profile, "test", dim)
        y_train_ref, y_val_ref, y_test_ref = y_train, y_val, y_test
        gmm_pack_path = out_dir / "models" / "gmm" / f"gmm_dim{dim}.joblib"
        if gmm_pack_path.exists() and not force:
            gmm_pack = joblib.load(gmm_pack_path)
        else:
            gmm_pack, _ = fit_gmm_pack(
                x_train,
                y_train,
                dim,
                {"name": "direct", "use_pca": False, "pca_dim": None},
                random_state=random_state,
            )
        gmm_train = predict_gmm_pack(x_train, gmm_pack, y_train)["posterior"]
        gmm_val = predict_gmm_pack(x_val, gmm_pack, y_train)["posterior"]
        gmm_test = predict_gmm_pack(x_test, gmm_pack, y_train)["posterior"]
        k_values = [k for k in [1, 2, 3, 5, 7, 10, 15] if k < len(y_train)]
        f_train, feature_blocks = build_router_features(x_train, y_train, x_train, gmm_train, gmm_train, k_values, is_train=True)
        f_val, _ = build_router_features(x_train, y_train, x_val, gmm_train, gmm_val, k_values, is_train=False)
        f_test, _ = build_router_features(x_train, y_train, x_test, gmm_train, gmm_test, k_values, is_train=False)
        all_dim_features["train"].append(f_train)
        all_dim_features["val"].append(f_val)
        all_dim_features["test"].append(f_test)
        all_dim_feature_labels = feature_blocks
        np.savez_compressed(
            feature_dir / f"router_features_dim{dim}.npz",
            x_train=f_train,
            x_val=f_val,
            x_test=f_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            k_values=np.array(k_values),
            feature_blocks=np.array(feature_blocks, dtype=object),
        )
        method = f"gmm_focused_knn_router_dim{dim}"
        if (method, dim) not in done:
            t0 = time.time()
            best = train_router_meta_classifier(f_train, y_train, f_val, y_val, random_state=random_state)
            pred_test = best["clf"].predict(best["scaler"].transform(f_test))
            test_m = macro_metrics(y_test, pred_test)
            rows.append(
                {
                    "dataset_id": profile.dataset_id,
                    "run_id": profile.run_id,
                    "profile": profile.profile_name,
                    "method_family": "gmm_focused_knn_router",
                    "method": method,
                    "dimension": dim,
                    "uses_knn": True,
                    "uses_gmm": True,
                    "uses_all_dims": False,
                    "best_params": json.dumps({"C": best["C"], "k_values": k_values, "feature_blocks": feature_blocks}),
                    "seconds": time.time() - t0,
                    **{f"val_{k}": v for k, v in best["val_metrics"].items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
            )
            joblib.dump(best, model_dir / f"router_dim{dim}.joblib")
            np.savez_compressed(
                pred_dir / f"{method}.npz",
                y_val=y_val,
                pred_val=best["pred_val"],
                y_test=y_test,
                pred_test=pred_test,
                classes=np.array(profile.classes, dtype=object),
            )
    # Proposed final: concatenate evidence from all available dimensions.
    method = "gmm_focused_knn_router_all_dims"
    if (method, -1) not in done and all_dim_features["train"]:
        t0 = time.time()
        f_train = np.concatenate(all_dim_features["train"], axis=1)
        f_val = np.concatenate(all_dim_features["val"], axis=1)
        f_test = np.concatenate(all_dim_features["test"], axis=1)
        best = train_router_meta_classifier(f_train, y_train_ref, f_val, y_val_ref, random_state=random_state)
        pred_test = best["clf"].predict(best["scaler"].transform(f_test))
        test_m = macro_metrics(y_test_ref, pred_test)
        rows.append(
            {
                "dataset_id": profile.dataset_id,
                "run_id": profile.run_id,
                "profile": profile.profile_name,
                "method_family": "gmm_focused_knn_router",
                "method": method,
                "dimension": -1,
                "uses_knn": True,
                "uses_gmm": True,
                "uses_all_dims": True,
                "best_params": json.dumps({"C": best["C"], "dims": dims, "feature_blocks_per_dim": all_dim_feature_labels}),
                "seconds": time.time() - t0,
                **{f"val_{k}": v for k, v in best["val_metrics"].items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        joblib.dump(best, model_dir / "router_all_dims.joblib")
        np.savez_compressed(
            pred_dir / f"{method}.npz",
            y_val=y_val_ref,
            pred_val=best["pred_val"],
            y_test=y_test_ref,
            pred_test=pred_test,
            classes=np.array(profile.classes, dtype=object),
        )

    append_rows(result_path, rows)
    return pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()


def run_profile_benchmark(
    profile: EmbeddingProfile,
    output_root: str | Path,
    dims: list[int] | None = None,
    force: bool = False,
    run_baselines: bool = True,
    run_gmm: bool = True,
    run_proposed: bool = True,
    random_state: int = 42,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        if run_baselines:
            frames.append(run_baseline_benchmark(profile, output_root, dims=dims, force=force, random_state=random_state))
    except Exception as exc:
        log_error(output_root, profile, "baselines", exc)
        print("ERROR baselines", profile.key, exc)
    try:
        if run_gmm:
            gmm_df, _ = run_gmm_benchmark(profile, output_root, dims=dims, force=force, random_state=random_state)
            frames.append(gmm_df)
    except Exception as exc:
        log_error(output_root, profile, "gmm", exc)
        print("ERROR gmm", profile.key, exc)
    try:
        if run_proposed:
            frames.append(run_gmm_focused_knn_router(profile, output_root, dims=dims, force=force, random_state=random_state))
    except Exception as exc:
        log_error(output_root, profile, "proposed_router", exc)
        print("ERROR proposed_router", profile.key, exc)
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_all_benchmarks(
    profiles: list[EmbeddingProfile],
    output_root: str | Path,
    dims: list[int] | None = None,
    force: bool = False,
    run_baselines: bool = True,
    run_gmm: bool = True,
    run_proposed: bool = True,
    random_state: int = 42,
):
    output_root = Path(output_root)
    all_frames = []
    for profile in profiles:
        if not profile.usable:
            print("SKIP unusable:", profile.key, profile.issues)
            continue
        print("\n=== Benchmark", profile.key, "===")
        use_dims = [d for d in (dims or profile.available_dims) if d in profile.available_dims]
        df = run_profile_benchmark(
            profile,
            output_root,
            dims=use_dims,
            force=force,
            run_baselines=run_baselines,
            run_gmm=run_gmm,
            run_proposed=run_proposed,
            random_state=random_state,
        )
        if not df.empty:
            all_frames.append(df)
            aggregate_results(output_root)
    return aggregate_results(output_root)


def aggregate_results(output_root: str | Path) -> pd.DataFrame:
    output_root = Path(output_root)
    tables = []
    for path in (output_root / "profiles").rglob("*_results.csv"):
        if path.name in {"baseline_results.csv", "gmm_results.csv", "proposed_router_results.csv"}:
            try:
                tables.append(pd.read_csv(path))
            except Exception:
                pass
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)
    df = df.drop_duplicates(subset=["dataset_id", "run_id", "profile", "method", "dimension"], keep="last")
    out = output_root / "summary_tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_method_results.csv", index=False)
    sort_cols = ["dataset_id", "test_f1_macro", "test_accuracy"]
    df.sort_values(sort_cols, ascending=[True, False, False]).to_csv(out / "all_method_results_sorted.csv", index=False)
    best = df.sort_values(["dataset_id", "profile", "test_f1_macro", "test_accuracy"], ascending=[True, True, False, False])
    best.groupby(["dataset_id", "profile"]).head(10).to_csv(out / "top10_methods_by_dataset.csv", index=False)
    return df


def load_prediction(output_root: str | Path, row: pd.Series):
    pred_dir = profile_output_dir(
        output_root,
        EmbeddingProfile(
            dataset_id=row["dataset_id"],
            run_id=row["run_id"],
            profile_name=row["profile"],
            profile_path=".",
            n_classes=0,
            classes=[],
            available_dims=[],
            missing_dims=[],
            split_counts={},
            split_class_counts={},
            has_checkpoint=False,
            has_existing_baselines=False,
            has_existing_gmm=False,
            has_existing_knn=False,
            usable=True,
            issues=[],
        ),
    ) / "predictions"
    path = pred_dir / f"{row['method']}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def plot_class_distributions(profiles: list[EmbeddingProfile], output_root: str | Path):
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        classes = profile.classes or [str(i) for i in range(profile.n_classes)]
        fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, min(9, profile.n_classes * 0.22))))
        for ax, split in zip(axes, SPLITS):
            counts = profile.split_class_counts.get(split, {})
            values = [counts.get(str(i), 0) for i in range(len(classes))]
            ax.barh(range(len(classes)), values)
            ax.set_title(f"{profile.dataset_id} {split}")
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=7)
            ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig_dir / f"class_distribution_{safe_name(profile.dataset_id)}_{safe_name(profile.profile_name)}.png", dpi=180)
        plt.close(fig)


def plot_benchmark_figures(output_root: str | Path, profiles: list[EmbeddingProfile]):
    output_root = Path(output_root)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = aggregate_results(output_root)
    if df.empty:
        return
    # Best method per dataset/profile.
    best = df.sort_values(["dataset_id", "profile", "test_f1_macro", "test_accuracy"], ascending=[True, True, False, False])
    best1 = best.groupby(["dataset_id", "profile"]).head(1).copy()
    best1.to_csv(output_root / "summary_tables" / "best_method_by_dataset.csv", index=False)

    # Bar chart top methods per dataset.
    for dataset_id, group in df.groupby("dataset_id"):
        top = group.sort_values(["test_f1_macro", "test_accuracy"], ascending=False).head(20).copy()
        labels = [f"{m}\\nD{int(d) if int(d) != -1 else 'all'}" for m, d in zip(top["method_family"], top["dimension"])]
        fig, ax = plt.subplots(figsize=(max(10, len(top) * 0.55), 5))
        ax.bar(range(len(top)), top["test_f1_macro"], color=["#4477AA" if not u else "#CC6677" for u in top["uses_all_dims"]])
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.set_ylim(0, min(1.0, max(0.1, top["test_f1_macro"].max() + 0.05)))
        ax.set_ylabel("Test macro F1")
        ax.set_title(f"Top methods - {dataset_id}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"top_methods_{safe_name(dataset_id)}.png", dpi=180)
        plt.close(fig)

    # Heatmap-like method family x dataset using best test macro F1.
    fam = df.groupby(["dataset_id", "method_family"])["test_f1_macro"].max().unstack()
    fig, ax = plt.subplots(figsize=(max(7, fam.shape[1] * 1.3), max(3, fam.shape[0] * 0.7)))
    im = ax.imshow(fam.fillna(0).values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(fam.shape[1]))
    ax.set_xticklabels(fam.columns, rotation=45, ha="right")
    ax.set_yticks(range(fam.shape[0]))
    ax.set_yticklabels(fam.index)
    for i in range(fam.shape[0]):
        for j in range(fam.shape[1]):
            val = fam.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.65 else "black", fontsize=8)
    ax.set_title("Best test macro F1 by dataset and method family")
    fig.colorbar(im, ax=ax, label="Test macro F1")
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_best_f1_by_family_dataset.png", dpi=180)
    plt.close(fig)

    # Dimension curves.
    dim_df = df[df["dimension"] >= 0].copy()
    for dataset_id, group in dim_df.groupby("dataset_id"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for family, sub in group.groupby("method_family"):
            curve = sub.groupby("dimension")["test_f1_macro"].max().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=family)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(group["dimension"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel("Best test macro F1")
        ax.set_title(f"Dimension sensitivity - {dataset_id}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"dimension_curve_{safe_name(dataset_id)}.png", dpi=180)
        plt.close(fig)

    # Mean rank across datasets.
    rank_source = df.groupby(["dataset_id", "method_family"])["test_f1_macro"].max().reset_index()
    rank_source["rank"] = rank_source.groupby("dataset_id")["test_f1_macro"].rank(ascending=False, method="average")
    ranks = rank_source.groupby("method_family")["rank"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(ranks) * 0.45)))
    ax.barh(ranks.index[::-1], ranks.values[::-1])
    ax.set_xlabel("Mean rank lower is better")
    ax.set_title("Mean method-family rank across datasets")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_rank_by_method_family.png", dpi=180)
    plt.close(fig)

    # Confusion matrices for best per dataset/profile when predictions exist.
    for _, row in best1.iterrows():
        pred = load_prediction(output_root, row)
        if pred is None:
            continue
        y_test = pred["y_test"]
        pred_test = pred["pred_test"]
        classes = [str(c) for c in pred["classes"].tolist()]
        if not classes:
            classes = [str(i) for i in sorted(np.unique(y_test).tolist())]
        label_ids = list(range(len(classes)))
        cm = confusion_matrix(y_test, pred_test, labels=label_ids)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            output_root / "summary_tables" / f"confusion_best_{safe_name(row['dataset_id'])}_{safe_name(row['profile'])}.csv"
        )
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.35), max(5, len(classes) * 0.35)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Best confusion matrix\n{row['dataset_id']} - {row['method']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(label_ids)
        ax.set_yticks(label_ids)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_best_{safe_name(row['dataset_id'])}_{safe_name(row['profile'])}.png", dpi=180)
        plt.close(fig)

        report = classification_report(y_test, pred_test, labels=label_ids, target_names=classes, zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(
            output_root / "summary_tables" / f"classification_report_best_{safe_name(row['dataset_id'])}_{safe_name(row['profile'])}.csv"
        )

    plot_class_distributions(profiles, output_root)


def plot_pca_embedding_previews(
    profiles: list[EmbeddingProfile],
    output_root: str | Path,
    dim: int = 64,
    max_points: int = 2500,
    random_state: int = 42,
):
    fig_dir = Path(output_root) / "figures" / "pca_previews"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    for profile in profiles:
        if dim not in profile.available_dims:
            continue
        x_train, y_train = load_split(profile, "train", dim)
        x_test, y_test = load_split(profile, "test", dim)
        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        split = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
        if len(y) > max_points:
            idx = rng.choice(len(y), size=max_points, replace=False)
            x, y, split = x[idx], y[idx], split[idx]
        coords = PCA(n_components=2, random_state=random_state).fit_transform(x)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, s=8, cmap="tab20", alpha=0.75)
        ax.set_title(f"PCA preview dim {dim} - {profile.dataset_id}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("label")
        fig.tight_layout()
        fig.savefig(fig_dir / f"pca_dim{dim}_{safe_name(profile.dataset_id)}_{safe_name(profile.profile_name)}.png", dpi=180)
        plt.close(fig)
