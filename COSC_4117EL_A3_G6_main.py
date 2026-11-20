# Generative AI note: implementation structure and boilerplate were prepared
# with Doubao assistance and reviewed for this assignment.
"""
Single-file implementation for COSC-4117EL Assignment 3.

Retail self-checkout product recognition using ResNet18 transfer learning,
with optional YOLOv8 comparison.

Examples:
    python COSC_4117EL_A3_G6_main.py collect --class-name bottled_water --split train --num 50
    python COSC_4117EL_A3_G6_main.py train --epochs 15
    python COSC_4117EL_A3_G6_main.py evaluate
    python COSC_4117EL_A3_G6_main.py realtime
    python COSC_4117EL_A3_G6_main.py train-yolo --epochs 30
    python COSC_4117EL_A3_G6_main.py yolo-realtime
    python COSC_4117EL_A3_G6_main.py compare
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "bottled_water",
    "cola_can",
    "juice_box",
    "chips_bag",
    "milk_bottle",
    "instant_noodle_cup",
]
IMAGE_SIZE = 224
RANDOM_SEED = 42
MODEL_PATH = OUTPUT_DIR / "resnet18_retail_best.pth"
METRICS_PATH = OUTPUT_DIR / "resnet_metrics.csv"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "resnet_confusion_matrix.png"
COMPARISON_PATH = OUTPUT_DIR / "resnet_yolo_comparison.csv"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------
def build_transforms(train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def require_split(split: str) -> Path:
    path = DATA_DIR / split
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the setup script and collect images first.")
    return path


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset = datasets.ImageFolder(require_split("train"), build_transforms(True))
    val_dataset = datasets.ImageFolder(require_split("val"), build_transforms(False))
    test_dataset = datasets.ImageFolder(require_split("test"), build_transforms(False))

    if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
        raise ValueError("Train, validation, and test class folders must match.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset.classes


# ---------------------------------------------------------------------------
# ResNet18 model
# ---------------------------------------------------------------------------
def build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_model(model: nn.Module, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(num_classes: int, path: Path = MODEL_PATH, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Webcam data collection
# ---------------------------------------------------------------------------
def collect_images(args: argparse.Namespace) -> None:
    output_dir = DATA_DIR / args.split / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    saved = 0
    last_capture = 0.0
    auto_mode = False
    print("Press c to capture, a to toggle auto-capture, q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.rectangle(display, (80, 60), (display.shape[1] - 80, display.shape[0] - 60), (0, 200, 0), 2)
        cv2.putText(display, f"{args.class_name} | saved {saved}/{args.num}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Webcam Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto_mode = not auto_mode
        should_capture = key == ord("c")
        if auto_mode and time.time() - last_capture >= 1.2:
            should_capture = True

        if should_capture:
            index = len(list(output_dir.glob("*.jpg")))
            file_path = output_dir / f"{args.class_name}_{index:04d}.jpg"
            cv2.imwrite(str(file_path), frame)
            saved += 1
            last_capture = time.time()
            if saved >= args.num:
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def run_train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def run_validation(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_resnet(args: argparse.Namespace) -> None:
    train_loader, val_loader, _, class_names = build_dataloaders(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    history = []
    best_val_accuracy = 0.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_last_block and epoch == max(3, args.epochs // 2):
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.2
            )
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_validation(model, val_loader, criterion, device)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc, time.time() - started])
        print(f"Epoch {epoch:02d} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")
        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "elapsed_seconds"])
        writer.writerows(history)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


def compute_confusion_matrix(model, loader, device, num_classes) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = torch.argmax(model(images), dim=1)
            for true_label, predicted_label in zip(labels, predictions):
                matrix[true_label.item(), predicted_label.item()] += 1
    return matrix


def metrics_from_confusion_matrix(matrix: np.ndarray) -> tuple[list[dict[str, float]], float]:
    rows = []
    for class_index in range(matrix.shape[0]):
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": class_index, "precision": precision, "recall": recall,
                     "f1_score": f1, "support": int(matrix[class_index, :].sum())})
    return rows, float(np.trace(matrix) / matrix.sum())


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
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
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > threshold else "black")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(figure)


def evaluate_resnet(args: argparse.Namespace) -> None:
    _, _, test_loader, class_names = build_dataloaders(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(class_names), MODEL_PATH, device)
    matrix = compute_confusion_matrix(model, test_loader, device, len(class_names))
    rows, accuracy = metrics_from_confusion_matrix(matrix)

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support", "overall_accuracy"])
        for row in rows:
            writer.writerow([
                class_names[int(row["class"])],
                f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                f"{row['f1_score']:.4f}", row["support"], f"{accuracy:.4f}",
            ])
    plot_confusion_matrix(matrix, class_names)
    print(f"Test accuracy: {accuracy:.4f}")


def realtime_resnet(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(CLASS_NAMES), MODEL_PATH, device)
    transform = build_transforms(False)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    frame_count = 0
    started = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = width // 6, height // 6, width * 5 // 6, height * 5 // 6
        roi = frame[y1:y2, x1:x2]

        with torch.no_grad():
            tensor = transform(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)
            confidence, index = torch.max(probabilities, dim=1)
            label = CLASS_NAMES[index.item()] if confidence.item() >= args.threshold else "Low confidence - reposition"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{label} ({confidence.item():.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        frame_count += 1
        fps = frame_count / max(time.time() - started, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("ResNet18 Retail Self-Checkout", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# YOLO bonus workflow
# ---------------------------------------------------------------------------
def train_yolo(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics or rerun setup without --skip-yolo.") from exc
    model = YOLO(args.weights)
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch_size, project=str(OUTPUT_DIR), name="yolov8_retail", exist_ok=True)


def yolo_realtime(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics or rerun setup without --skip-yolo.") from exc
    model = YOLO(args.weights)
    results = model.predict(source=args.camera, stream=True, conf=args.confidence, show=True, verbose=False)
    for _ in results:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def compare_models(_args: argparse.Namespace) -> None:
    rows = [
        ["dimension", "ResNet18", "YOLOv8n"],
        ["task", "image classification", "object detection"],
        ["primary output", "class probabilities", "class probabilities and bounding boxes"],
        ["product localization", "not directly supported", "supported through bounding boxes"],
        ["multiple products", "limited; first version assumes one product", "designed for multiple objects"],
        ["annotation cost", "class folder labels only", "bounding-box labels required"],
        ["model size", "relatively small", "lightweight detector but larger than classification-only setup"],
        ["retail self-checkout fit", "single-product classification in a fixed ROI", "localization and multi-product detection"],
        ["main limitation", "cannot distinguish product positions", "requires more annotation and detector-specific tuning"],
    ]
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print("Comparison table saved to", COMPARISON_PATH)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assignment 3 retail product recognition.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect webcam images.")
    collect_parser.add_argument("--class-name", required=True, choices=CLASS_NAMES)
    collect_parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    collect_parser.add_argument("--num", type=int, default=40)
    collect_parser.add_argument("--camera", type=int, default=0)
    collect_parser.set_defaults(func=collect_images)

    train_parser = subparsers.add_parser("train", help="Train ResNet18.")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--unfreeze-last-block", action="store_true")
    train_parser.set_defaults(func=train_resnet)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate ResNet18.")
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.set_defaults(func=evaluate_resnet)

    realtime_parser = subparsers.add_parser("realtime", help="Run real-time ResNet18.")
    realtime_parser.add_argument("--camera", type=int, default=0)
    realtime_parser.add_argument("--threshold", type=float, default=0.70)
    realtime_parser.set_defaults(func=realtime_resnet)

    yolo_train_parser = subparsers.add_parser("train-yolo", help="Train YOLOv8.")
    yolo_train_parser.add_argument("--weights", default="yolov8n.pt")
    yolo_train_parser.add_argument("--epochs", type=int, default=30)
    yolo_train_parser.add_argument("--imgsz", type=int, default=640)
    yolo_train_parser.add_argument("--batch-size", type=int, default=8)
    yolo_train_parser.set_defaults(func=train_yolo)

    yolo_realtime_parser = subparsers.add_parser("yolo-realtime", help="Run real-time YOLO.")
    yolo_realtime_parser.add_argument("--weights", default="outputs/yolov8_retail/weights/best.pt")
    yolo_realtime_parser.add_argument("--camera", type=int, default=0)
    yolo_realtime_parser.add_argument("--confidence", type=float, default=0.5)
    yolo_realtime_parser.set_defaults(func=yolo_realtime)

    compare_parser = subparsers.add_parser("compare", help="Create model comparison table.")
    compare_parser.set_defaults(func=compare_models)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
