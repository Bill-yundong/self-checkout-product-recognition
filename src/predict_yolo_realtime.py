"""Real-time YOLO retail product detection with bounding boxes."""
import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO webcam product detection.")
    parser.add_argument("--weights", default="outputs/yolov8_retail/weights/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.weights)
    results = model.predict(
        source=args.camera,
        stream=True,
        conf=args.confidence,
        show=True,
        verbose=False,
    )

    for result in results:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main()
