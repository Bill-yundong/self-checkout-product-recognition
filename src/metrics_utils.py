"""Manual confusion-matrix and classification metric calculations."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_confusion_matrix(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> np.ndarray:
    """Compute a confusion matrix without relying on scikit-learn."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            for true_label, predicted_label in zip(labels, predictions):
                matrix[true_label.item(), predicted_label.item()] += 1
    return matrix


def metrics_from_confusion_matrix(matrix: np.ndarray) -> tuple[list[dict[str, float]], float]:
    """Calculate per-class precision, recall, F1, and overall accuracy."""
    rows = []
    total = int(matrix.sum())
    correct = int(np.trace(matrix))

    for class_index in range(matrix.shape[0]):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "class_index": class_index,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(matrix[class_index, :].sum()),
        })

    accuracy = correct / total if total else 0.0
    return rows, accuracy


def write_metrics_csv(rows: list[dict[str, float]], class_names: list[str], accuracy: float, path: Path) -> None:
    """Write per-class metrics and overall accuracy to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support", "overall_accuracy"])
        for row in rows:
            name = class_names[int(row["class_index"])]
            writer.writerow([
                name,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1_score']:.4f}",
                row["support"],
                f"{accuracy:.4f}",
            ])
