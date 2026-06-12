from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "efficientnet_b0": {"image_size": 224, "batch_size": 32},
    "efficientnet_b2": {"image_size": 260, "batch_size": 20},
    "efficientnet_b3": {"image_size": 300, "batch_size": 12},
    "resnet18": {"image_size": 224, "batch_size": 48},
    "resnet34": {"image_size": 224, "batch_size": 32},
    "resnet50": {"image_size": 224, "batch_size": 24},
    "densenet121": {"image_size": 224, "batch_size": 24},
    "mobilenet_v3_large": {"image_size": 224, "batch_size": 40},
    "convnext_tiny": {"image_size": 224, "batch_size": 16},
    "vit_b_16": {"image_size": 224, "batch_size": 8},
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class CsvRelImageDataset(Dataset):
    def __init__(self, csv_path: Path, datasets_root: Path, transform):
        self.frame = pd.read_csv(csv_path)
        self.datasets_root = datasets_root
        self.transform = transform
        if "rel_path" not in self.frame.columns:
            raise ValueError(f"CSV must contain rel_path: {csv_path}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = self.datasets_root / str(row["rel_path"])
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        y = int(row["label"])
        return x, y, str(row["rel_path"])


@dataclass
class ModelBundle:
    model: nn.Module
    classifier: nn.Module
    feature_dim: int


def build_model(arch: str, n_classes: int, weights: str = "imagenet") -> ModelBundle:
    arch = arch.lower()
    use_weights = weights == "imagenet"

    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if use_weights else None
        model = models.efficientnet_b0(weights=w)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "efficientnet_b2":
        w = models.EfficientNet_B2_Weights.DEFAULT if use_weights else None
        model = models.efficientnet_b2(weights=w)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "efficientnet_b3":
        w = models.EfficientNet_B3_Weights.DEFAULT if use_weights else None
        model = models.efficientnet_b3(weights=w)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if use_weights else None
        model = models.resnet18(weights=w)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.fc, in_features)

    if arch == "resnet34":
        w = models.ResNet34_Weights.DEFAULT if use_weights else None
        model = models.resnet34(weights=w)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.fc, in_features)

    if arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if use_weights else None
        model = models.resnet50(weights=w)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.fc, in_features)

    if arch == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if use_weights else None
        model = models.densenet121(weights=w)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if use_weights else None
        model = models.mobilenet_v3_large(weights=w)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if use_weights else None
        model = models.convnext_tiny(weights=w)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.classifier, in_features)

    if arch == "vit_b_16":
        w = models.ViT_B_16_Weights.DEFAULT if use_weights else None
        model = models.vit_b_16(weights=w)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, n_classes)
        return ModelBundle(model, model.heads.head, in_features)

    raise ValueError(f"Unsupported architecture: {arch}")


def set_trainable(bundle: ModelBundle, freeze_mode: str) -> None:
    freeze_mode = freeze_mode.lower()
    if freeze_mode == "full":
        for p in bundle.model.parameters():
            p.requires_grad = True
        return
    for p in bundle.model.parameters():
        p.requires_grad = False
    if freeze_mode == "classifier":
        for p in bundle.classifier.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"This Colab runner supports freeze_mode full or classifier. Got {freeze_mode}")


def make_transforms(image_size: int, augment: bool):
    train_tfms = []
    if augment:
        train_tfms.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        train_tfms.append(transforms.Resize((image_size, image_size)))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(train_tfms), eval_tfms


def macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def balanced_acc_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recs = []
    for cls in range(n_classes):
        mask = y_true == cls
        recs.append(0.0 if int(mask.sum()) == 0 else float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recs))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def evaluate(model, loader, device, loss_fn, n_classes: int, amp: bool):
    model.eval()
    total_loss = 0.0
    total = 0
    ys, preds, paths = [], [], []
    probs_parts = []
    with torch.no_grad():
        for x, y, rel in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            prob = torch.softmax(logits.detach(), dim=1)
            pred = prob.argmax(1)
            total_loss += float(loss.detach().cpu()) * len(y)
            total += len(y)
            ys.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            probs_parts.append(prob.detach().cpu().numpy().astype(np.float32))
            paths.extend(list(rel))
    y_true = np.concatenate(ys) if ys else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)
    probs = np.concatenate(probs_parts) if probs_parts else np.zeros((0, n_classes), dtype=np.float32)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": macro_f1_np(y_true, y_pred, n_classes),
        "balanced_accuracy": balanced_acc_np(y_true, y_pred, n_classes),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "paths": paths,
    }


def train_epoch(model, loader, device, loss_fn, optimizer, scaler, grad_accum_steps: int, amp: bool):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = loss_fn(logits, y) / max(grad_accum_steps, 1)
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        full_loss = float(loss.detach().cpu()) * max(grad_accum_steps, 1)
        total_loss += full_loss * len(y)
        total += len(y)
        correct += int((logits.argmax(1) == y).sum().detach().cpu())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def write_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def class_weights_from_csv(train_csv: Path, n_classes: int, device):
    y = pd.read_csv(train_csv)["label"].to_numpy(dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts * n_classes, 1e-6, None)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loaders(pack_dir: Path, dataset_id: str, image_size: int, batch_size: int, num_workers: int, augment: bool):
    split_dir = pack_dir / "splits" / dataset_id
    datasets_root = pack_dir / "datasets"
    train_tfms, eval_tfms = make_transforms(image_size, augment)
    ds_train = CsvRelImageDataset(split_dir / "train.csv", datasets_root, train_tfms)
    ds_val = CsvRelImageDataset(split_dir / "val.csv", datasets_root, eval_tfms)
    ds_test = CsvRelImageDataset(split_dir / "test.csv", datasets_root, eval_tfms)
    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders


def save_predictions(job_dir: Path, split: str, result: dict[str, Any], classes: list[str]) -> None:
    pred_dir = job_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "rel_path": result["paths"],
        "y_true": result["y_true"].astype(int),
        "y_pred": result["y_pred"].astype(int),
        "class_true": [classes[int(i)] for i in result["y_true"]],
        "class_pred": [classes[int(i)] for i in result["y_pred"]],
        "max_prob": result["probs"].max(axis=1) if len(result["probs"]) else [],
    })
    df.to_csv(pred_dir / f"{split}_predictions.csv", index=False)
    np.save(pred_dir / f"{split}_probs.npy", result["probs"].astype(np.float32))
    cm = confusion_matrix_np(result["y_true"], result["y_pred"], len(classes))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(pred_dir / f"{split}_confusion_matrix.csv")


def job_id(dataset_id: str, arch: str, freeze_mode: str, seed: int, image_size: int) -> str:
    return safe_name(f"{dataset_id}__{arch}__{freeze_mode}__seed{seed}__img{image_size}")


def run_job(pack_dir: Path, dataset_id: str, model_cfg: dict[str, Any], global_cfg: dict[str, Any], args) -> dict[str, Any]:
    arch = model_cfg["arch"]
    weights = model_cfg.get("weights", global_cfg.get("weights", "imagenet"))
    seed = int(model_cfg.get("seed", global_cfg.get("seed", 42)))
    freeze_mode = model_cfg.get("freeze_mode", global_cfg.get("freeze_mode", "full"))
    image_size = int(model_cfg.get("image_size", MODEL_DEFAULTS.get(arch, {}).get("image_size", 224)))
    batch_size = int(args.batch_size or model_cfg.get("batch_size", MODEL_DEFAULTS.get(arch, {}).get("batch_size", global_cfg.get("batch_size", 16))))
    epochs = int(args.epochs or model_cfg.get("epochs", global_cfg.get("epochs", 15)))
    lr = float(model_cfg.get("lr", global_cfg.get("lr", 1e-4)))
    weight_decay = float(model_cfg.get("weight_decay", global_cfg.get("weight_decay", 1e-4)))
    num_workers = int(args.num_workers if args.num_workers is not None else global_cfg.get("num_workers", 2))
    patience = int(model_cfg.get("patience", global_cfg.get("patience", 6)))
    grad_accum_steps = int(model_cfg.get("grad_accum_steps", global_cfg.get("grad_accum_steps", 1)))
    amp = bool(global_cfg.get("amp", True))
    augment = bool(global_cfg.get("augment", True))

    split_info = read_json(pack_dir / "splits" / dataset_id / "dataset_info.json", {})
    classes = split_info.get("classes")
    if not classes:
        raise ValueError(f"Missing classes for {dataset_id}")
    n_classes = len(classes)

    jid = job_id(dataset_id, arch, freeze_mode, seed, image_size)
    job_dir = pack_dir / "results" / dataset_id / arch / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"
    if done_path.exists() and not args.force:
        print(f"SKIP done: {jid}")
        return {"status": "skipped_done", "job_id": jid, "job_dir": str(job_dir)}

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"START {jid} device={device} epochs={epochs} batch={batch_size} lr={lr}")
    save_json({
        "job_id": jid,
        "dataset_id": dataset_id,
        "arch": arch,
        "weights": weights,
        "freeze_mode": freeze_mode,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "classes": classes,
        "started_at": now(),
        "pack_dir": str(pack_dir),
    }, job_dir / "job_config.json")

    set_seed(seed)
    loaders = build_loaders(pack_dir, dataset_id, image_size, batch_size, num_workers, augment)
    bundle = build_model(arch, n_classes, weights=weights)
    set_trainable(bundle, freeze_mode)
    model = bundle.model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_from_csv(pack_dir / "splits" / dataset_id / "train.csv", n_classes, device))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    last_path = job_dir / "last.pt"
    best_path = job_dir / "best.pt"
    history_path = job_dir / "history.csv"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0

    if last_path.exists() and not args.restart:
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        history = ckpt.get("history", [])
        print(f"RESUME {jid} from epoch {start_epoch}, best_val_f1={best_val_f1:.6f}")
    elif history_path.exists() and not args.restart:
        history = pd.read_csv(history_path).to_dict("records")

    epochs_no_improve = 0
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, loaders["train"], device, loss_fn, optimizer, scaler, grad_accum_steps, amp)
        val_metrics = evaluate(model, loaders["val"], device, loss_fn, n_classes, amp)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_min": (time.time() - t0) / 60.0,
            "timestamp": now(),
        }
        history.append(row)
        improved = row["val_macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
            "job_config": read_json(job_dir / "job_config.json", {}),
        }
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        write_history(history, history_path)
        save_json({
            "job_id": jid,
            "status": "running",
            "epoch": epoch,
            "epochs": epochs,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "updated_at": now(),
        }, job_dir / "progress.json")
        print(f"{jid} epoch {epoch}/{epochs} train_acc={row['train_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f} best={best_val_f1:.4f}")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"EARLY STOP {jid}: no improvement for {patience} epochs")
            break

    if not best_path.exists():
        save_checkpoint(best_path, load_checkpoint(last_path, device))
    best_ckpt = load_checkpoint(best_path, device)
    model.load_state_dict(best_ckpt["model"])
    val_result = evaluate(model, loaders["val"], device, loss_fn, n_classes, amp)
    test_result = evaluate(model, loaders["test"], device, loss_fn, n_classes, amp)
    save_predictions(job_dir, "val", val_result, classes)
    save_predictions(job_dir, "test", test_result, classes)
    metrics = {
        "job_id": jid,
        "dataset_id": dataset_id,
        "arch": arch,
        "weights": weights,
        "freeze_mode": freeze_mode,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "best_epoch": int(best_ckpt.get("best_epoch", best_epoch)),
        "best_val_f1_from_training": float(best_ckpt.get("best_val_f1", best_val_f1)),
        "val_loss": val_result["loss"],
        "val_accuracy": val_result["accuracy"],
        "val_macro_f1": val_result["macro_f1"],
        "val_balanced_accuracy": val_result["balanced_accuracy"],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
        "test_balanced_accuracy": test_result["balanced_accuracy"],
        "finished_at": now(),
    }
    save_json(metrics, job_dir / "metrics.json")
    save_json({"status": "done", **metrics}, done_path)
    print(f"DONE {jid} test_f1={metrics['test_macro_f1']:.6f} test_acc={metrics['test_accuracy']:.6f}")
    del model, bundle, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "done", "job_id": jid, "job_dir": str(job_dir), **metrics}


def collect_results(pack_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in (pack_dir / "results").rglob("metrics.json"):
        rows.append(read_json(metrics_path, {}))
    df = pd.DataFrame(rows)
    out = pack_dir / "results" / "ALL_FINETUNE_RESULTS.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_cols = [
        "job_id", "dataset_id", "arch", "weights", "freeze_mode", "seed",
        "image_size", "batch_size", "best_epoch", "val_accuracy",
        "val_macro_f1", "val_balanced_accuracy", "test_accuracy",
        "test_macro_f1", "test_balanced_accuracy", "finished_at",
    ]
    if not df.empty:
        df.sort_values(["dataset_id", "arch", "test_macro_f1"], ascending=[True, True, False]).to_csv(out, index=False)
    else:
        pd.DataFrame(columns=expected_cols).to_csv(out, index=False)
    return df


def parse_csv_filter(text: str | None) -> set[str] | None:
    if not text:
        return None
    vals = {x.strip() for x in str(text).split(",") if x.strip()}
    return vals or None


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path, {})


def command_run(args) -> int:
    pack_dir = Path(args.pack_dir).resolve()
    config = load_config(Path(args.config).resolve() if args.config else pack_dir / "configs" / "full_finetune_config.json")
    datasets = config.get("datasets") or [p.name for p in sorted((pack_dir / "splits").iterdir()) if p.is_dir()]
    models_cfg = [m for m in config.get("models", []) if m.get("enabled", True)]
    dataset_filter = parse_csv_filter(args.datasets)
    model_filter = parse_csv_filter(args.models)
    if dataset_filter:
        datasets = [d for d in datasets if d in dataset_filter]
    if model_filter:
        models_cfg = [m for m in models_cfg if m.get("arch") in model_filter]

    print("PACK_DIR:", pack_dir)
    print("Datasets:", datasets)
    print("Models:", [m.get("arch") for m in models_cfg])
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        for d in datasets:
            for m in models_cfg:
                arch = m["arch"]
                image_size = int(m.get("image_size", MODEL_DEFAULTS.get(arch, {}).get("image_size", 224)))
                seed = int(m.get("seed", config.get("seed", 42)))
                freeze_mode = m.get("freeze_mode", config.get("freeze_mode", "full"))
                print(job_id(d, arch, freeze_mode, seed, image_size))
        return 0

    run_rows = []
    failed_rows = []
    jobs_done = 0
    for dataset_id in datasets:
        for model_cfg in models_cfg:
            if args.max_jobs and jobs_done >= args.max_jobs:
                print("Reached --max-jobs")
                collect_results(pack_dir)
                return 0
            try:
                row = run_job(pack_dir, dataset_id, model_cfg, config, args)
                run_rows.append(row)
                jobs_done += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                err = {"dataset_id": dataset_id, "arch": model_cfg.get("arch"), "error": str(e), "traceback": traceback.format_exc(), "time": now()}
                failed_rows.append(err)
                print("ERROR", err["dataset_id"], err["arch"], err["error"])
                save_json(err, pack_dir / "results" / "errors" / f"{safe_name(dataset_id)}__{safe_name(model_cfg.get('arch'))}.json")
                if args.stop_on_error:
                    raise
            except Exception as e:
                err = {"dataset_id": dataset_id, "arch": model_cfg.get("arch"), "error": str(e), "traceback": traceback.format_exc(), "time": now()}
                failed_rows.append(err)
                print("ERROR", err["dataset_id"], err["arch"], err["error"])
                save_json(err, pack_dir / "results" / "errors" / f"{safe_name(dataset_id)}__{safe_name(model_cfg.get('arch'))}.json")
                if args.stop_on_error:
                    raise
            finally:
                collect_results(pack_dir)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(pack_dir / "results" / "FAILED_JOBS.csv", index=False)
    collect_results(pack_dir)
    return 0


def command_collect(args) -> int:
    df = collect_results(Path(args.pack_dir).resolve())
    print(df.to_string(index=False) if not df.empty else "No metrics found yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab full fine-tuning runner with checkpoint resume")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--pack-dir", default=str(Path(__file__).resolve().parents[1]))
    r.add_argument("--config", default="")
    r.add_argument("--datasets", default="", help="Comma-separated dataset ids; empty = config/all")
    r.add_argument("--models", default="", help="Comma-separated model architectures; empty = config/all")
    r.add_argument("--epochs", type=int, default=0, help="Override epochs for every job")
    r.add_argument("--batch-size", type=int, default=0, help="Override batch size for every job")
    r.add_argument("--num-workers", type=int, default=None)
    r.add_argument("--device", default="auto")
    r.add_argument("--max-jobs", type=int, default=0)
    r.add_argument("--force", action="store_true", help="Rerun even if done.json exists")
    r.add_argument("--restart", action="store_true", help="Ignore last.pt and start a job from epoch 1")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--stop-on-error", action="store_true")
    r.set_defaults(func=command_run)

    c = sub.add_parser("collect")
    c.add_argument("--pack-dir", default=str(Path(__file__).resolve().parents[1]))
    c.set_defaults(func=command_collect)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
