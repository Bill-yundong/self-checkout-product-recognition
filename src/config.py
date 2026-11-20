# Generative AI note: project boilerplate and implementation structure were
# prepared with Doubao assistance and reviewed for this assignment.
"""Central configuration for the retail self-checkout product classifier."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
YOLO_DATA_DIR = PROJECT_ROOT / "yolo_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"

CLASS_NAMES = [
    "bottled_water",
    "cola_can",
    "juice_box",
    "chips_bag",
    "milk_bottle",
    "instant_noodle_cup",
]

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
CONFIDENCE_THRESHOLD = 0.70
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_PATH = OUTPUT_DIR / "resnet18_retail_best.pth"
METRICS_PATH = OUTPUT_DIR / "resnet_metrics.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "resnet_confusion_matrix.png"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
