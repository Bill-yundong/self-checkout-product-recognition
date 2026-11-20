"""Evaluate the trained ResNet18 model and save metrics plus confusion matrix."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CONFUSION_MATRIX_PATH, METRICS_PATH, MODEL_PATH
from dataset_utils import build_dataloaders
from metrics_utils import compute_confusion_matrix, metrics_from_confusion_matrix, write_metrics_csv
from model_utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 on the test set.")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("ResNet18 Retail Product Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _, _, test_loader, class_names = build_dataloaders(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(class_names), MODEL_PATH, device)

    matrix = compute_confusion_matrix(model, test_loader, device, len(class_names))
    rows, accuracy = metrics_from_confusion_matrix(matrix)
    write_metrics_csv(rows, class_names, accuracy, METRICS_PATH)
    plot_confusion_matrix(matrix, class_names, CONFUSION_MATRIX_PATH)

    print(f"Test accuracy: {accuracy:.4f}")
    print("Metrics saved to", METRICS_PATH)
    print("Confusion matrix saved to", CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
