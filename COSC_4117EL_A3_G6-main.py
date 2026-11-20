# -*- coding: utf-8 -*-
"""
COSC-4117EL-01 Artificial Intelligence, Fall 2026 - Assignment 3, Group 6
Real-Time Retail Self-Checkout Product Recognition with ResNet18 Transfer Learning

Group members: Qingdong Wu (0460101), Yuhang Zhao (0452606)

This program extends the course sample code (CNN_ResNet_Sample.py). The sample
collects webcam images, trains ResNet18 with a frozen backbone and a new final
layer, and shows live predictions. This version keeps that flow and adapts it
to a six-class retail self-checkout problem.

Provenance tags used in the comments below:
    [SAMPLE]    kept from CNN_ResNet_Sample.py (same logic, possibly renamed)
    [MODIFIED]  logic from the sample that was changed, with the reason
    [NEW]       added by Group 6 for this project

Generative AI acknowledgement: the initial code structure was drafted with
Doubao. This version was revised with Claude (Anthropic) to align it with the
course sample, fix the class-order and region-of-interest issues, and add the
evaluation, smoothing and YOLO comparison code. All generated code was reviewed
and tested by the group, and this use is also acknowledged in the report.

Workflow (run COSC_4117EL_A3_G6-setup.py first):
    1. Collect images:   python COSC_4117EL_A3_G6-main.py collect --class-name cola_can --split train --num 50
    2. Train ResNet18:   python COSC_4117EL_A3_G6-main.py train
    3. Evaluate:         python COSC_4117EL_A3_G6-main.py evaluate
    4. Live demo:        python COSC_4117EL_A3_G6-main.py realtime
    Bonus (YOLO):
    5. Train YOLOv8-cls: python COSC_4117EL_A3_G6-main.py train-yolo
    6. Evaluate YOLO:    python COSC_4117EL_A3_G6-main.py evaluate --model yolo
    7. Live demo YOLO:   python COSC_4117EL_A3_G6-main.py realtime --model yolo
    8. Compare models:   python COSC_4117EL_A3_G6-main.py compare

Keys in webcam windows: 'c' capture, 'a' toggle auto-capture, 'q' quit.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import matplotlib

matplotlib.use("Agg")  # [NEW] save figures without needing a display window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"        # dataset/<split>/<class>/*.jpg
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# [MODIFIED] The sample used three generic classes (class1..class3). We use six
# retail products. They are listed in alphabetical order on purpose, because
# torchvision ImageFolder and Ultralytics both number classes alphabetically.
# The trained class order is also stored in the checkpoint (see save_checkpoint).
CLASS_NAMES = [
    "bottled_water",
    "chips_bag",
    "cola_can",
    "instant_noodle_cup",
    "juice_box",
    "milk_bottle",
]
SPLITS = ["train", "val", "test"]
IMAGE_SIZE = 224                            # [SAMPLE] ResNet18 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]       # [SAMPLE]
IMAGENET_STD = [0.229, 0.224, 0.225]        # [SAMPLE]
RANDOM_SEED = 42

RESNET_CHECKPOINT = OUTPUT_DIR / "resnet18_retail_best.pth"
YOLO_RUN_NAME = "yolov8n_cls_retail"
YOLO_WEIGHTS = OUTPUT_DIR / YOLO_RUN_NAME / "weights" / "best.pt"
HISTORY_PATH = OUTPUT_DIR / "resnet_training_history.csv"
CURVES_PATH = OUTPUT_DIR / "resnet_training_curves.png"
COMPARISON_CSV = OUTPUT_DIR / "resnet_vs_yolo_comparison.csv"
COMPARISON_PNG = OUTPUT_DIR / "resnet_vs_yolo_comparison.png"

# A predictor takes one BGR image (the ROI) and returns class probabilities
# ordered like CLASS_NAMES.
Predictor = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_SEED) -> None:
    """[NEW] Make training and data shuffling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """[SAMPLE] Use CUDA if available. [MODIFIED] Also use Apple MPS on Macs."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_roi(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """[NEW] Return the product placement region and its box (x1, y1, x2, y2).

    The same region is used when collecting images and when predicting live,
    so the model sees the same kind of picture during training and use.
    """
    height, width = frame.shape[:2]
    x1, y1 = width // 6, height // 6
    x2, y2 = width * 5 // 6, height * 5 // 6
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """[SAMPLE] Open the webcam. [NEW] Fail with a clear message if it cannot."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")
    return cap


# ---------------------------------------------------------------------------
# Step 1: Collect data using the webcam
# ---------------------------------------------------------------------------
def collect_data(class_name: str, split: str, num_samples: int, camera_index: int) -> None:
    """Collect webcam images of one product for one data split.

    [SAMPLE] Press 'c' to save the current frame, stop after num_samples.
    [MODIFIED] Save only the green placement region (ROI) instead of the full
    frame, and store images in dataset/<split>/<class>/ so that train,
    validation and test images are kept apart.
    [NEW] Press 'a' to toggle automatic capture every 1.2 s, 'q' to quit early.
    """
    output_dir = DATA_DIR / split / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # [NEW] Continue numbering after existing files so nothing is overwritten.
    existing = [int(p.stem.rsplit("_", 1)[-1]) for p in output_dir.glob("*.jpg")
                if p.stem.rsplit("_", 1)[-1].isdigit()]
    next_index = max(existing, default=-1) + 1

    cap = open_camera(camera_index)
    count = 0
    auto_mode = False
    last_capture = 0.0
    print(f"Collecting {class_name} ({split}). Keys: c=capture, a=auto, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi, (x1, y1, x2, y2) = get_roi(frame)

        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
        mode = "AUTO" if auto_mode else "manual"
        cv2.putText(display, f"{class_name} [{split}] {count}/{num_samples} ({mode})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Collect Data", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto_mode = not auto_mode
        capture = key == ord("c") or (auto_mode and time.time() - last_capture >= 1.2)

        if capture:
            file_path = output_dir / f"{class_name}_{next_index:04d}.jpg"
            cv2.imwrite(str(file_path), roi)
            print(f"sample {count + 1}: {file_path.name}")
            next_index += 1
            count += 1
            last_capture = time.time()
            if count >= num_samples:
                break

    # [SAMPLE] Release the webcam and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Step 2: Preprocessing, model and training
# ---------------------------------------------------------------------------
def build_transforms(train: bool) -> transforms.Compose:
    """Build the image preprocessing pipeline.

    [SAMPLE] Resize to 224x224, random rotation, ImageNet normalization.
    [MODIFIED] The sample applied the same random augmentation to the live
    frames as well. Here augmentation is used for training only; validation,
    test and live frames get the plain resize + normalize pipeline.
    [MODIFIED] Horizontal flip from the sample is removed: it mirrors the text
    and logos on the packaging, which never appear mirrored at a checkout.
    [NEW] Random resized crop and colour jitter simulate different placements,
    distances and lighting.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create training and validation loaders from dataset/train and dataset/val.

    [MODIFIED] The sample loaded every image into memory once, so each image got
    one fixed random augmentation for all epochs. ImageFolder applies a new
    random augmentation every epoch and assigns labels from the folder names.
    """
    for split in ("train", "val"):
        if not (DATA_DIR / split).exists():
            raise FileNotFoundError(f"Missing {DATA_DIR / split}. Run setup and collect images first.")
    train_set = datasets.ImageFolder(DATA_DIR / "train", build_transforms(train=True))
    val_set = datasets.ImageFolder(DATA_DIR / "val", build_transforms(train=False))
    if train_set.classes != val_set.classes:
        raise ValueError("Class folders in dataset/train and dataset/val must match.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_set.classes


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Load ResNet18, freeze it, and replace the final layer.

    [SAMPLE] ImageNet-pretrained ResNet18 with all layers frozen and a new
    fully connected layer for our classes.
    [MODIFIED] resnet18(pretrained=True) is deprecated in torchvision, so the
    weights enum is used instead. num_classes is 6 instead of 3.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new layer is trainable
    return model


def keep_frozen_batchnorm_fixed(model: nn.Module) -> None:
    """[NEW] Put frozen BatchNorm layers in eval mode.

    model.train() would otherwise keep updating the running mean/variance of
    the frozen ImageNet layers with our small batches, changing the features
    even though their weights are frozen.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: optim.Optimizer | None = None) -> tuple[float, float]:
    """Run one pass over a loader; train if an optimizer is given.

    [SAMPLE] zero_grad -> forward -> loss -> backward -> step.
    [NEW] Also used for validation, and returns average loss and accuracy.
    """
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_batchnorm_fixed(model)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def save_checkpoint(model: nn.Module, class_names: list[str], path: Path = RESNET_CHECKPOINT) -> None:
    """[SAMPLE] Save the model state dict.
    [MODIFIED] Also save the class order, so predictions map to the right names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": class_names}, path)


def load_checkpoint(device: torch.device, path: Path = RESNET_CHECKPOINT) -> tuple[nn.Module, list[str]]:
    """[SAMPLE] Load the saved weights and switch to evaluation mode."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Train the model first.")
    checkpoint = torch.load(path, map_location=device)
    class_names = list(checkpoint["class_names"])
    model = build_resnet18(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, class_names


def plot_training_curves(history: list[list[float]]) -> None:
    """[NEW] Plot loss and accuracy per epoch for the report."""
    epochs = [row[0] for row in history]
    figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, [row[1] for row in history], marker="o", label="train")
    loss_axis.plot(epochs, [row[3] for row in history], marker="o", label="validation")
    loss_axis.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    acc_axis.plot(epochs, [row[2] for row in history], marker="o", label="train")
    acc_axis.plot(epochs, [row[4] for row in history], marker="o", label="validation")
    acc_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    for axis in (loss_axis, acc_axis):
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(CURVES_PATH, dpi=200)
    plt.close(figure)


def train_resnet(args: argparse.Namespace) -> None:
    """Train ResNet18 with transfer learning and keep the best validation model.

    [SAMPLE] CrossEntropyLoss + Adam (lr 0.001) on the final layer only.
    [NEW] Validation after each epoch, best-model saving, history CSV and plot,
    and optional fine-tuning of the last residual block (layer4).
    """
    set_seed()
    device = get_device()
    train_loader, val_loader, class_names = build_dataloaders(args.batch_size)
    print(f"Device: {device} | classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | validation images: {len(val_loader.dataset)}")

    model = build_resnet18(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    unfreeze_epoch = max(2, args.epochs // 2 + 1)

    history: list[list[float]] = []
    best_val_acc, best_val_loss = -1.0, float("inf")
    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_last_block and epoch == unfreeze_epoch:
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.1)
            print(f"Epoch {epoch}: layer4 unfrozen for fine-tuning")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])
        print(f"Epoch {epoch}/{args.epochs}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, "
              f"val loss {val_loss:.4f}, val acc {val_acc:.4f}")

        # Keep the model with the best validation accuracy (lower loss breaks ties).
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            save_checkpoint(model, class_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writerows(history)
    plot_training_curves(history)
    print(f"Best validation accuracy: {best_val_acc:.4f}. Model saved to {RESNET_CHECKPOINT}")


# ---------------------------------------------------------------------------
# Predictors shared by evaluation and the live demo
# ---------------------------------------------------------------------------
def make_resnet_predictor(device: torch.device) -> tuple[Predictor, list[str], int]:
    """Return a ResNet18 predictor, its class names and its parameter count.

    [SAMPLE] Transform the frame, add a batch dimension, run without gradients.
    [MODIFIED] Convert OpenCV's BGR colour order to RGB first (the sample fed
    BGR frames to a model pretrained on RGB images) and use softmax so we get
    a confidence value, not only the predicted index.
    """
    model, class_names = load_checkpoint(device)
    transform = build_transforms(train=False)

    def predict(image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    num_params = sum(p.numel() for p in model.parameters())
    return predict, class_names, num_params


def load_yolo():
    """[NEW] Import Ultralytics YOLO only when the bonus part is used."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed. Run the setup script without --skip-yolo.") from exc
    return YOLO


def make_yolo_predictor() -> tuple[Predictor, list[str], int]:
    """[NEW] Return a YOLOv8-cls predictor with the same interface as ResNet18."""
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing {YOLO_WEIGHTS}. Run the train-yolo command first.")
    model = load_yolo()(str(YOLO_WEIGHTS))
    device = str(get_device())  # same device as ResNet18 for a fair speed comparison
    names = model.names  # {index: class_name}
    class_names = [names[i] for i in sorted(names)]

    def predict(image_bgr: np.ndarray) -> np.ndarray:
        # Ultralytics accepts OpenCV BGR arrays directly.
        result = model.predict(image_bgr, imgsz=IMAGE_SIZE, device=device, verbose=False)[0]
        return result.probs.data.cpu().numpy()

    num_params = sum(p.numel() for p in model.model.parameters())
    return predict, class_names, num_params


def make_predictor(model_name: str) -> tuple[Predictor, list[str], int]:
    """[NEW] Build the predictor for 'resnet' or 'yolo'."""
    if model_name == "yolo":
        return make_yolo_predictor()
    return make_resnet_predictor(get_device())


# ---------------------------------------------------------------------------
# Step 3: Evaluation on the held-out test set
# ---------------------------------------------------------------------------
def list_test_images(class_names: list[str]) -> list[tuple[Path, int]]:
    """[NEW] Return (image path, label index) pairs from dataset/test."""
    samples = []
    for label, class_name in enumerate(class_names):
        folder = DATA_DIR / "test" / class_name
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, label))
    if not samples:
        raise FileNotFoundError(f"No test images found in {DATA_DIR / 'test'}.")
    return samples


def per_class_metrics(matrix: np.ndarray) -> list[dict[str, float]]:
    """[NEW] Precision, recall, F1 and support for each class from a confusion matrix."""
    rows = []
    for k in range(matrix.shape[0]):
        tp = matrix[k, k]
        fp = matrix[:, k].sum() - tp
        fn = matrix[k, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"precision": float(precision), "recall": float(recall),
                     "f1": float(f1), "support": int(matrix[k, :].sum())})
    return rows


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    """[NEW] Save a labelled confusion matrix figure."""
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set(xlabel="Predicted label", ylabel="True label", title=title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model on dataset/test: accuracy, P/R/F1, confusion matrix, speed.

    [NEW] The sample had no evaluation. Test images go through exactly the same
    predictor as the live demo, and the time per image is measured so the
    report can state inference latency and the maximum achievable FPS.
    """
    predict, class_names, num_params = make_predictor(args.model)
    samples = list_test_images(class_names)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    # Warm-up runs so one-time start-up cost is not counted in the timing.
    warmup_image = cv2.imread(str(samples[0][0]))
    for _ in range(3):
        predict(warmup_image)

    latencies_ms = []
    for path, label in samples:
        image = cv2.imread(str(path))
        start = time.perf_counter()
        probabilities = predict(image)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        matrix[label, int(np.argmax(probabilities))] += 1

    rows = per_class_metrics(matrix)
    accuracy = float(np.trace(matrix) / matrix.sum())
    macro = {key: float(np.mean([row[key] for row in rows])) for key in ("precision", "recall", "f1")}
    mean_latency = float(np.mean(latencies_ms))
    device = str(get_device())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"{args.model}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name, row in zip(class_names, rows):
            writer.writerow([name, f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                             f"{row['f1']:.4f}", row["support"]])
        writer.writerow(["macro_average", f"{macro['precision']:.4f}", f"{macro['recall']:.4f}",
                         f"{macro['f1']:.4f}", int(matrix.sum())])

    summary = {
        "model": "ResNet18" if args.model == "resnet" else "YOLOv8n-cls",
        "test_accuracy": f"{accuracy:.4f}",
        "macro_precision": f"{macro['precision']:.4f}",
        "macro_recall": f"{macro['recall']:.4f}",
        "macro_f1": f"{macro['f1']:.4f}",
        "mean_latency_ms": f"{mean_latency:.2f}",
        "max_model_fps": f"{1000 / mean_latency:.1f}",
        "parameters": num_params,
        "test_images": int(matrix.sum()),
        "device": device,
    }
    with (OUTPUT_DIR / f"{args.model}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    plot_confusion_matrix(matrix, class_names, f"{summary['model']} confusion matrix (test set)",
                          OUTPUT_DIR / f"{args.model}_confusion_matrix.png")
    print(f"{summary['model']}: accuracy {accuracy:.4f}, macro F1 {macro['f1']:.4f}, "
          f"{mean_latency:.1f} ms/image ({1000 / mean_latency:.1f} FPS max)")
    print(f"Results saved in {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Step 4: Real-time classification with the webcam
# ---------------------------------------------------------------------------
def realtime(args: argparse.Namespace) -> None:
    """Classify the product in the placement region of each webcam frame.

    [SAMPLE] Read frame -> preprocess -> predict -> draw label -> 'q' to quit.
    [MODIFIED] Only the green placement region is classified (same region as
    in data collection) instead of the whole frame.
    [NEW] Real-time adaptations:
      * temporal smoothing: probabilities are averaged over the last N frames,
        which stops the label from flickering between classes;
      * confidence threshold: below it the system asks the customer to
        reposition the product instead of guessing;
      * on-screen model latency (ms) and smoothed FPS for the report and video.
    """
    predict, class_names, _ = make_predictor(args.model)
    title = "ResNet18" if args.model == "resnet" else "YOLOv8n-cls"
    recent = deque(maxlen=max(1, args.smooth))
    cap = open_camera(args.camera)
    fps, latency_ms = 0.0, 0.0
    previous = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi, (x1, y1, x2, y2) = get_roi(frame)

        start = time.perf_counter()
        recent.append(predict(roi))
        frame_latency = (time.perf_counter() - start) * 1000
        # Exponential moving averages keep the on-screen numbers readable.
        latency_ms = 0.9 * latency_ms + 0.1 * frame_latency if latency_ms else frame_latency
        probabilities = np.mean(recent, axis=0)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        if confidence >= args.threshold:
            label, colour = f"{class_names[index]} ({confidence:.2f})", (0, 200, 0)
        else:
            label, colour = f"Low confidence ({confidence:.2f}) - reposition", (0, 0, 255)

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - previous, 1e-6)
        fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
        previous = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(frame, f"{title} | {latency_ms:.1f} ms | {fps:.1f} FPS",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(f"Predictions - {title}", frame)

        # [SAMPLE] Exit loop on pressing 'q' key.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Average model latency {latency_ms:.1f} ms, display FPS {fps:.1f}")
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Bonus: YOLO comparison
# ---------------------------------------------------------------------------
def train_yolo(args: argparse.Namespace) -> None:
    """[NEW] Fine-tune YOLOv8n-cls on the same dataset folders as ResNet18.

    The classification variant of YOLOv8 reads dataset/train, dataset/val and
    dataset/test directly, so no bounding-box annotation is needed and both
    models are trained and tested on exactly the same images.
    """
    YOLO = load_yolo()
    model = YOLO(args.weights)
    model.train(data=str(DATA_DIR), epochs=args.epochs, imgsz=IMAGE_SIZE, batch=args.batch_size,
                project=str(OUTPUT_DIR), name=YOLO_RUN_NAME, exist_ok=True, seed=RANDOM_SEED,
                fliplr=0.0)  # no mirroring, same reason as for ResNet18
    print(f"YOLO weights saved to {YOLO_WEIGHTS}")


def read_summary(model_name: str) -> dict[str, str] | None:
    """[NEW] Read outputs/<model>_summary.csv written by the evaluate command."""
    path = OUTPUT_DIR / f"{model_name}_summary.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def compare(_args: argparse.Namespace) -> None:
    """[NEW] Build a measured ResNet18 vs YOLOv8n-cls comparison table and chart."""
    summaries = [s for s in (read_summary("resnet"), read_summary("yolo")) if s]
    if len(summaries) < 2:
        raise FileNotFoundError("Run 'evaluate' and 'evaluate --model yolo' before 'compare'.")

    fields = ["test_accuracy", "macro_precision", "macro_recall", "macro_f1",
              "mean_latency_ms", "max_model_fps", "parameters", "test_images", "device"]
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + [s["model"] for s in summaries])
        for field in fields:
            writer.writerow([field] + [s[field] for s in summaries])

    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    keys = ["test_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(len(labels))
    figure, (score_axis, speed_axis) = plt.subplots(1, 2, figsize=(12, 4.5),
                                                     gridspec_kw={"width_ratios": [3, 1]})
    for offset, summary in zip((-0.2, 0.2), summaries):
        bars = score_axis.bar(x + offset, [float(summary[k]) for k in keys], width=0.4, label=summary["model"])
        score_axis.bar_label(bars, fmt="%.2f", fontsize=8)
    score_axis.set_xticks(x, labels)
    score_axis.set(ylim=(0, 1.2), title="Test-set classification quality")
    score_axis.legend(loc="upper center", ncol=2)
    bars = speed_axis.bar([s["model"] for s in summaries], [float(s["mean_latency_ms"]) for s in summaries],
                          color=["tab:blue", "tab:orange"])
    speed_axis.bar_label(bars, fmt="%.1f ms", fontsize=8)
    speed_axis.set(title="Mean latency per image", ylabel="ms")
    figure.tight_layout()
    figure.savefig(COMPARISON_PNG, dpi=200)
    plt.close(figure)
    print(f"Comparison saved to {COMPARISON_CSV} and {COMPARISON_PNG}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """[NEW] Sub-commands replace the sample's single top-to-bottom script, so
    each stage can be rerun on its own (e.g. retrain without recollecting)."""
    parser = argparse.ArgumentParser(description="Assignment 3 Group 6: retail product recognition.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Collect webcam images for one class and split.")
    p.add_argument("--class-name", required=True, choices=CLASS_NAMES)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--camera", type=int, default=0)
    p.set_defaults(func=lambda a: collect_data(a.class_name, a.split, a.num, a.camera))

    p = sub.add_parser("train", help="Train ResNet18 with transfer learning.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--unfreeze-last-block", action="store_true",
                   help="Fine-tune layer4 during the second half of training.")
    p.set_defaults(func=train_resnet)

    p = sub.add_parser("evaluate", help="Evaluate a trained model on dataset/test.")
    p.add_argument("--model", choices=["resnet", "yolo"], default="resnet")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("realtime", help="Run live webcam classification.")
    p.add_argument("--model", choices=["resnet", "yolo"], default="resnet")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--smooth", type=int, default=5, help="Number of frames to average.")
    p.set_defaults(func=realtime)

    p = sub.add_parser("train-yolo", help="Bonus: train YOLOv8n-cls on the same dataset.")
    p.add_argument("--weights", default="yolov8n-cls.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.set_defaults(func=train_yolo)

    p = sub.add_parser("compare", help="Bonus: build the ResNet18 vs YOLO comparison.")
    p.set_defaults(func=compare)
    return parser


def main() -> None:
    """Parse the command line and run the selected stage."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
