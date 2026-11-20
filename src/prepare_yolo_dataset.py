"""Create a YOLO data.yaml file for the annotated retail product dataset."""
import yaml

from config import CLASS_NAMES, YOLO_DATA_DIR


def main() -> None:
    for split in ["train", "val", "test"]:
        image_dir = YOLO_DATA_DIR / "images" / split
        label_dir = YOLO_DATA_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    output_path = YOLO_DATA_DIR / "data.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print("YOLO configuration written to", output_path)


if __name__ == "__main__":
    main()
