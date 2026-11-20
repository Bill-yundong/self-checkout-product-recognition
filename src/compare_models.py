"""Create a qualitative comparison table for ResNet18 and YOLOv8."""
import csv

from config import OUTPUT_DIR


def main() -> None:
    output_path = OUTPUT_DIR / "resnet_yolo_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print("Comparison table saved to", output_path)


if __name__ == "__main__":
    main()
