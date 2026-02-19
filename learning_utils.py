from __future__ import annotations

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def save_labeled_image(image: Image.Image, label: str, data_dir: str = "data") -> str:
    """Save a user-labeled sample under ImageFolder-compatible structure."""
    safe_label = label.strip().replace(" ", "_")
    class_dir = Path(data_dir) / safe_label
    class_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"sample_{int(time.time() * 1000)}.jpg"
    file_path = class_dir / file_name
    image.convert("RGB").save(file_path, format="JPEG", quality=95)
    return str(file_path)


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _expand_classifier_if_needed(model: nn.Module, old_classes: List[str], new_classes: List[str]) -> nn.Module:
    if len(old_classes) == len(new_classes):
        return model

    old_fc: nn.Linear = model.fc
    new_fc = nn.Linear(old_fc.in_features, len(new_classes))

    with torch.no_grad():
        new_fc.weight[: len(old_classes)] = old_fc.weight
        new_fc.bias[: len(old_classes)] = old_fc.bias

    model.fc = new_fc
    return model


def fine_tune_from_dataset(
    model_path: str = "best_model.pth",
    data_dir: str = "data",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-4,
) -> Dict[str, object]:
    """Quick incremental fine-tuning after new labeled samples are added."""
    dataset = datasets.ImageFolder(data_dir, transform=DEFAULT_TRANSFORM)
    class_names = dataset.classes

    checkpoint = torch.load(model_path, map_location="cpu")
    old_classes = checkpoint["class_names"]
    model = _build_model(len(old_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = _expand_classifier_if_needed(model, old_classes, class_names)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
        },
        model_path,
    )

    return {
        "classes": class_names,
        "samples": len(dataset),
        "last_loss": last_loss,
    }


def import_from_web_manifest(manifest_path: str, data_dir: str = "data", timeout: int = 10) -> Dict[str, int]:
    """
    Download images from internet based on a JSON manifest:
    {
      "cat": ["https://...jpg", ...],
      "dog": ["https://...jpg", ...]
    }
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        sources = json.load(f)

    downloaded = {}
    for label, urls in sources.items():
        count = 0
        for idx, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                class_dir = Path(data_dir) / label
                class_dir.mkdir(parents=True, exist_ok=True)
                image.save(class_dir / f"web_{idx}_{int(time.time())}.jpg", format="JPEG")
                count += 1
            except Exception:
                continue
        downloaded[label] = count

    return downloaded
