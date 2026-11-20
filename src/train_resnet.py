"""Train ResNet18 with transfer learning for retail product classification."""
import argparse
import csv
import time

import torch
import torch.nn as nn

from config import LEARNING_RATE, NUM_EPOCHS, OUTPUT_DIR, TRAINING_HISTORY_PATH
from dataset_utils import build_dataloaders
from model_utils import build_resnet18, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 product classifier.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--unfreeze-last-block", action="store_true",
                        help="Fine-tune the final residual block after the classifier head.")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
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


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    train_loader, val_loader, _, class_names = build_dataloaders(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18(num_classes=len(class_names), freeze_backbone=True).to(device)
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
            print("Unfrozen layer4 for limited fine-tuning.")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - started
        history.append([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            save_model(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "elapsed_seconds"])
        writer.writerows(history)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Training history saved to", TRAINING_HISTORY_PATH)


if __name__ == "__main__":
    main()
