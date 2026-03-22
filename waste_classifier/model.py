from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_model(
    num_classes: int,
    freeze_backbone: bool = True,
    use_pretrained: bool = True,
):
    """Build a ResNet18 classifier head for the waste dataset."""

    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

