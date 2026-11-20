"""ResNet18 model construction and loading helpers."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from config import MODEL_PATH


def build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Build a pretrained ResNet18 with a custom product classification head."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_model(model: nn.Module, path: Path = MODEL_PATH) -> None:
    """Save model weights and ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(num_classes: int, path: Path = MODEL_PATH, device: torch.device | None = None) -> nn.Module:
    """Load saved ResNet18 weights for evaluation."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
