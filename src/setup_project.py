"""Create all required class folders before webcam collection."""
from config import CLASS_NAMES, DATA_DIR, OUTPUT_DIR, YOLO_DATA_DIR


def main() -> None:
    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Project folders are ready.")


if __name__ == "__main__":
    main()
