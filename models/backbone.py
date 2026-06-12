"""Torchvision backbone construction used by the full fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torchvision import models


@dataclass
class ModelBundle:
    model: nn.Module
    classifier: nn.Module
    feature_dim: int


def build_backbone(
    architecture: str, number_of_classes: int, *, imagenet_weights: bool = True
) -> ModelBundle:
    """Build one of the nine CNN backbones evaluated in the paper."""
    architecture = architecture.lower()
    builders = {
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
        ),
        "efficientnet_b2": (
            models.efficientnet_b2,
            models.EfficientNet_B2_Weights.DEFAULT,
        ),
        "efficientnet_b3": (
            models.efficientnet_b3,
            models.EfficientNet_B3_Weights.DEFAULT,
        ),
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "convnext_tiny": (
            models.convnext_tiny,
            models.ConvNeXt_Tiny_Weights.DEFAULT,
        ),
    }
    if architecture not in builders:
        raise ValueError(f"Unsupported architecture: {architecture}")

    builder, default_weights = builders[architecture]
    model = builder(weights=default_weights if imagenet_weights else None)

    if architecture.startswith("efficientnet_"):
        feature_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(feature_dim, number_of_classes)
        classifier = model.classifier
    elif architecture.startswith("resnet"):
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, number_of_classes)
        classifier = model.fc
    elif architecture == "densenet121":
        feature_dim = model.classifier.in_features
        model.classifier = nn.Linear(feature_dim, number_of_classes)
        classifier = model.classifier
    elif architecture == "mobilenet_v3_large":
        feature_dim = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(feature_dim, number_of_classes)
        classifier = model.classifier
    else:
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(feature_dim, number_of_classes)
        classifier = model.classifier
    return ModelBundle(model=model, classifier=classifier, feature_dim=feature_dim)
