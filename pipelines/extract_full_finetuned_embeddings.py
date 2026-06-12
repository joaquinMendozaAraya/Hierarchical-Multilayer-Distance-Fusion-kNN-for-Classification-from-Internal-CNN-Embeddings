from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "google_drive_full_finetuning_pack"
COLAB_SCRIPT = PACK_DIR / "scripts" / "full_finetune_colab.py"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "baselines_from_colab" / "results"
DEFAULT_OUTPUT = ROOT / "experiments" / "51_final_vchmf_all_scenarios"
SPLITS = ["train", "val", "test"]


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_")


def parse_csv(text: str | None) -> set[str]:
    if not text:
        return set()
    return {part.strip() for part in str(text).split(",") if part.strip()}


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_colab_module():
    if not COLAB_SCRIPT.exists():
        raise FileNotFoundError(COLAB_SCRIPT)
    spec = importlib.util.spec_from_file_location("full_finetune_colab_local", COLAB_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {COLAB_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict(model, state: dict[str, Any]) -> None:
    sd = state.get("model", state)
    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        stripped = {}
        for key, value in sd.items():
            stripped[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(stripped, strict=True)


def profile_dir(output: Path, dataset_id: str, run_id: str, profile_name: str) -> Path:
    return output / "embeddings" / safe_name(dataset_id) / safe_name(run_id) / "profiles" / safe_name(profile_name)


def completed_profile(profile: Path) -> bool:
    if not (profile / "split_info.json").exists():
        return False
    for split in SPLITS:
        split_dir = profile / "embeddings" / "multicapa_norm" / split
        if not (split_dir / "labels.npy").exists():
            return False
        if not list(split_dir.glob("z_dim_*.npy")):
            return False
    return True


def write_split_embeddings(profile: Path, split: str, arrays: dict[str, np.ndarray], labels: np.ndarray, paths: list[str]) -> None:
    out = profile / "embeddings" / "multicapa_norm" / split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "labels.npy", labels.astype(np.int64))
    for key, arr in arrays.items():
        np.save(out / f"z_dim_{int(key)}.npy", arr.astype(np.float32))
    pd.DataFrame({"path": paths, "label": labels.astype(int)}).to_csv(out / "paths.csv", index=False)


def make_loaders(mod, dataset_id: str, image_size: int, batch_size: int, num_workers: int):
    _, eval_tfms = mod.make_transforms(image_size, augment=False)
    split_dir = PACK_DIR / "splits" / dataset_id
    datasets_root = PACK_DIR / "datasets"
    loaders = {}
    for split in SPLITS:
        csv_path = split_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        ds = mod.CsvRelImageDataset(csv_path, datasets_root, eval_tfms)
        loaders[split] = mod.DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
        )
    return loaders


def selected_hook_modules(model, arch: str) -> dict[str, Any]:
    arch = arch.lower()
    if arch.startswith("efficientnet_"):
        return {f"features_{i}": model.features[i] for i in range(1, len(model.features))}
    if arch.startswith("resnet"):
        return {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }
    if arch == "densenet121":
        return {
            "transition1": model.features.transition1,
            "transition2": model.features.transition2,
            "transition3": model.features.transition3,
            "norm5": model.features.norm5,
        }
    if arch == "mobilenet_v3_large":
        candidates = [1, 3, 6, 10, 12, 15, 16]
        return {f"features_{i}": model.features[i] for i in candidates if i < len(model.features)}
    if arch == "convnext_tiny":
        candidates = [1, 3, 5, 7]
        return {f"features_{i}": model.features[i] for i in candidates if i < len(model.features)}
    if arch == "vit_b_16":
        return {"encoder": model.encoder}
    raise ValueError(f"Unsupported architecture for feature hooks: {arch}")


def reduce_feature(torch, nn, tensor):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    if tensor.ndim == 4:
        return nn.functional.adaptive_avg_pool2d(tensor, (1, 1)).flatten(1)
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    if tensor.ndim == 2:
        return tensor
    return tensor.flatten(1)


def extract_embeddings(mod, model, hooks: dict[str, Any], loader, device) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], list[dict[str, Any]]]:
    torch = mod.torch
    nn = mod.nn
    outputs: dict[str, Any] = {}
    handles = []
    for name, module in hooks.items():
        def _hook(_m, _inp, out, hook_name=name):
            outputs[hook_name] = out.detach()
        handles.append(module.register_forward_hook(_hook))

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in hooks}
    labels: list[int] = []
    paths: list[str] = []
    model.eval()
    with torch.no_grad():
        for x, y, rel in loader:
            outputs.clear()
            _ = model(x.to(device, non_blocking=True))
            for name in hooks:
                feat = reduce_feature(torch, nn, outputs[name])
                chunks[name].append(feat.detach().cpu().numpy().astype(np.float32))
            labels.extend([int(v) for v in y.numpy().tolist()])
            paths.extend(list(rel))
    for handle in handles:
        handle.remove()

    arrays: dict[str, np.ndarray] = {}
    hook_meta: list[dict[str, Any]] = []
    for name, parts in chunks.items():
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        dim = int(arr.shape[1])
        meta = {"hook_name": name, "dim": dim, "n_samples": int(arr.shape[0]), "used": False}
        key = str(dim)
        if key not in arrays:
            arrays[key] = l2_rows(arr)
            meta["used"] = True
        else:
            meta["skip_reason"] = "duplicate_embedding_dim"
        hook_meta.append(meta)
    return arrays, np.asarray(labels, dtype=np.int64), paths, hook_meta


def iter_jobs(results_root: Path, dataset_filter: set[str], model_filter: set[str], include_incomplete: bool) -> list[dict[str, Any]]:
    jobs = []
    for config_path in sorted(results_root.rglob("job_config.json")):
        job_dir = config_path.parent
        cfg = read_json(config_path, {})
        dataset_id = str(cfg.get("dataset_id", ""))
        arch = str(cfg.get("arch", ""))
        if dataset_filter and dataset_id not in dataset_filter:
            continue
        if model_filter and arch not in model_filter:
            continue
        ckpt_path = job_dir / "best.pt"
        if not ckpt_path.exists():
            jobs.append({"status": "missing_best_checkpoint", "job_dir": str(job_dir), "dataset_id": dataset_id, "arch": arch})
            continue
        metrics_path = job_dir / "metrics.json"
        done_path = job_dir / "done.json"
        if not include_incomplete and not metrics_path.exists():
            jobs.append({"status": "missing_metrics_json", "job_dir": str(job_dir), "dataset_id": dataset_id, "arch": arch})
            continue
        if not include_incomplete and not done_path.exists():
            jobs.append({"status": "missing_done_json", "job_dir": str(job_dir), "dataset_id": dataset_id, "arch": arch})
            continue
        jobs.append(
            {
                "status": "ready",
                "job_dir": str(job_dir),
                "checkpoint_path": str(ckpt_path),
                "metrics_path": str(metrics_path),
                "config": cfg,
                "dataset_id": dataset_id,
                "arch": arch,
            }
        )
    return jobs


def run(args) -> None:
    mod = load_colab_module()
    torch = mod.torch
    mod.set_seed(int(args.seed))

    results_root = Path(args.results_root)
    output = Path(args.output)
    datasets = parse_csv(args.datasets)
    models = parse_csv(args.models)
    jobs = iter_jobs(results_root, datasets, models, bool(args.include_incomplete))
    ready_jobs = [job for job in jobs if job["status"] == "ready"]
    skipped_jobs = [job for job in jobs if job["status"] != "ready"]
    report_dir = output / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(jobs).to_csv(report_dir / "colab_full_finetune_embedding_jobs.csv", index=False)

    print(f"Found jobs: {len(jobs)} | ready: {len(ready_jobs)} | skipped: {len(skipped_jobs)}")
    if args.dry_run:
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    extracted_rows = []
    error_rows = []
    for idx, job in enumerate(ready_jobs, start=1):
        cfg = job["config"]
        dataset_id = str(cfg["dataset_id"])
        arch = str(cfg["arch"])
        weights = str(cfg.get("weights", "imagenet"))
        freeze_mode = str(cfg.get("freeze_mode", "full"))
        seed = int(cfg.get("seed", args.seed))
        image_size = int(cfg.get("image_size", mod.MODEL_DEFAULTS.get(arch, {}).get("image_size", 224)))
        batch_size = int(args.batch_size or cfg.get("batch_size") or mod.MODEL_DEFAULTS.get(arch, {}).get("batch_size", 16))
        classes = [str(c) for c in cfg.get("classes", [])]
        if not classes:
            info = read_json(PACK_DIR / "splits" / dataset_id / "dataset_info.json", {})
            classes = [str(c) for c in info.get("classes", [])]
        if not classes:
            error_rows.append({"dataset_id": dataset_id, "arch": arch, "error": "missing_classes", "job_dir": job["job_dir"]})
            continue

        run_id = f"colab_seed{seed}"
        profile_name = f"{arch}_{weights}_{freeze_mode}_img{image_size}_colab_best_gap"
        prof = profile_dir(output, dataset_id, run_id, profile_name)
        if completed_profile(prof) and not args.force:
            print(f"[{idx}/{len(ready_jobs)}] skip existing {dataset_id} {arch}: {prof}")
            extracted_rows.append(
                {
                    "dataset_id": dataset_id,
                    "arch": arch,
                    "profile_name": profile_name,
                    "profile_path": str(prof),
                    "status": "skipped_existing",
                }
            )
            continue

        print(f"[{idx}/{len(ready_jobs)}] extract {dataset_id} | {arch} | img={image_size} | batch={batch_size}", flush=True)
        t0 = time.time()
        try:
            bundle = mod.build_model(arch, len(classes), weights="none")
            model = bundle.model.to(device)
            state = load_checkpoint(torch, Path(job["checkpoint_path"]), device)
            load_state_dict(model, state)
            hooks = selected_hook_modules(model, arch)
            loaders = make_loaders(mod, dataset_id, image_size, batch_size, int(args.num_workers))

            split_hook_meta = {}
            embedding_dims: list[int] = []
            for split in SPLITS:
                arrays, labels, paths, hook_meta = extract_embeddings(mod, model, hooks, loaders[split], device)
                write_split_embeddings(prof, split, arrays, labels, paths)
                split_hook_meta[split] = hook_meta
                embedding_dims = sorted(int(dim) for dim in arrays)
                print(f"  {split}: n={len(labels)} dims={embedding_dims}", flush=True)

            metrics = read_json(Path(job["metrics_path"]), {})
            save_json(
                {
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "profile_name": profile_name,
                    "classes": classes,
                    "arch": arch,
                    "weights": weights,
                    "fine_tuned": True,
                    "freeze_mode": freeze_mode,
                    "source": "imported_colab_full_finetune_best_checkpoint",
                    "job_dir": job["job_dir"],
                    "checkpoint_path": job["checkpoint_path"],
                    "image_size": image_size,
                    "seed": seed,
                    "embedding_dims": embedding_dims,
                    "hook_modules": list(hooks.keys()),
                    "hook_meta": split_hook_meta,
                    "colab_metrics": metrics,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                prof / "split_info.json",
            )
            save_json({"classes": classes, "base_split_info": {"classes": classes}}, prof / "splits" / "split_info.json")
            elapsed = time.time() - t0
            extracted_rows.append(
                {
                    "dataset_id": dataset_id,
                    "arch": arch,
                    "profile_name": profile_name,
                    "profile_path": str(prof),
                    "embedding_dims": ",".join(map(str, embedding_dims)),
                    "status": "extracted",
                    "elapsed_sec": elapsed,
                }
            )
            print(f"  profile: {prof} ({elapsed / 60:.2f} min)", flush=True)
        except Exception as exc:
            error_rows.append(
                {
                    "dataset_id": dataset_id,
                    "arch": arch,
                    "profile_name": profile_name,
                    "job_dir": job["job_dir"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"ERROR {dataset_id} {arch}: {exc!r}", flush=True)

    pd.DataFrame(extracted_rows).to_csv(report_dir / "colab_full_finetune_embedding_extracted.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(report_dir / "colab_full_finetune_embedding_errors.csv", index=False)
    print("Extraction report:", report_dir / "colab_full_finetune_embedding_extracted.csv")
    if error_rows:
        print("Errors:", report_dir / "colab_full_finetune_embedding_errors.csv")


def build_parser():
    p = argparse.ArgumentParser(description="Extract post-hoc embedding profiles from imported Colab full fine-tuned checkpoints.")
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--datasets", default="", help="Comma-separated dataset ids; empty means all.")
    p.add_argument("--models", default="", help="Comma-separated architectures; empty means all.")
    p.add_argument("--batch-size", type=int, default=0, help="0 means use each job/default batch size.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="", help="Empty means cuda if available, else cpu.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-incomplete", action="store_true", help="Also extract jobs that have best.pt but no final metrics/done marker.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
