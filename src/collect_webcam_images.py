"""Collect webcam images for one product class and dataset split."""
import argparse
import time
from pathlib import Path

import cv2

from config import CLASS_NAMES, DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect webcam product images.")
    parser.add_argument("--class-name", required=True, choices=CLASS_NAMES)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = DATA_DIR / args.split / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    saved = 0
    last_capture = 0.0
    print(f"Collecting images in {output_dir}")
    print("Press c to capture, a to auto-capture every 1.2 seconds, q to quit.")

    auto_mode = False
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
            print(f"Auto capture: {auto_mode}")

        should_capture = key == ord("c")
        if auto_mode and time.time() - last_capture >= 1.2:
            should_capture = True

        if should_capture:
            file_path = output_dir / f"{args.class_name}_{len(list(output_dir.glob('*.jpg'))):04d}.jpg"
            cv2.imwrite(str(file_path), frame)
            saved += 1
            last_capture = time.time()
            print(f"Saved {file_path}")
            if saved >= args.num:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. New images captured in this session: {saved}")


if __name__ == "__main__":
    main()
