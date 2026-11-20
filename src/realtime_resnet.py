"""Real-time webcam product classification using the trained ResNet18."""
import argparse
import time

import cv2
import torch
from PIL import Image

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, MODEL_PATH
from dataset_utils import build_transforms
from model_utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time ResNet18 product recognition.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(CLASS_NAMES), MODEL_PATH, device)
    transform = build_transforms(train=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    frame_count = 0
    started = time.time()
    prediction_text = "No prediction"
    confidence = 0.0

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
            confidence_value, index = torch.max(probabilities, dim=1)
            confidence = confidence_value.item()
            if confidence >= args.threshold:
                prediction_text = CLASS_NAMES[index.item()]
            else:
                prediction_text = "Low confidence - reposition"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{prediction_text} ({confidence:.2f})", (20, 40),
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


if __name__ == "__main__":
    main()
