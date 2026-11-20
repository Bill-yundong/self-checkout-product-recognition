"""Train a lightweight YOLOv8 model for retail product detection."""
import argparse

from config import OUTPUT_DIR, YOLO_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 product detector.")
    parser.add_argument("--data", default=str(YOLO_DATA_DIR / "data.yaml"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(OUTPUT_DIR),
        name="yolov8_retail",
        exist_ok=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main()
