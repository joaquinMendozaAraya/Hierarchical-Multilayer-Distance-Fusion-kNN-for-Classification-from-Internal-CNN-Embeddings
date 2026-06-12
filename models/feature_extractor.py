"""Internal CNN view extraction helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def selected_hook_modules(model: nn.Module, architecture: str) -> dict[str, Any]:
    architecture = architecture.lower()
    if architecture.startswith("efficientnet_"):
        return {
            f"features_{index}": model.features[index]
            for index in range(1, len(model.features))
        }
    if architecture.startswith("resnet"):
        return {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }
    if architecture == "densenet121":
        return {
            "transition1": model.features.transition1,
            "transition2": model.features.transition2,
            "transition3": model.features.transition3,
            "norm5": model.features.norm5,
        }
    if architecture == "mobilenet_v3_large":
        indices = [1, 3, 6, 10, 12, 15, 16]
        return {
            f"features_{index}": model.features[index]
            for index in indices
            if index < len(model.features)
        }
    if architecture == "convnext_tiny":
        return {
            f"features_{index}": model.features[index]
            for index in [1, 3, 5, 7]
            if index < len(model.features)
        }
    raise ValueError(f"Unsupported architecture: {architecture}")


def reduce_feature(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    if tensor.ndim == 4:
        return nn.functional.adaptive_avg_pool2d(tensor, (1, 1)).flatten(1)
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    if tensor.ndim == 2:
        return tensor
    return tensor.flatten(1)
