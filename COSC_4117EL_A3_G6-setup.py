# -*- coding: utf-8 -*-
"""
COSC-4117EL-01 Artificial Intelligence, Fall 2026 - Assignment 3, Group 6
Environment and folder setup. Run this file once before COSC_4117EL_A3_G6-main.py.

    python COSC_4117EL_A3_G6-setup.py               # install packages + create folders
    python COSC_4117EL_A3_G6-setup.py --skip-install
    python COSC_4117EL_A3_G6-setup.py --skip-yolo   # skip the bonus YOLO package

[NEW] The course sample created data/class1..class3 at start-up. This script
creates dataset/<split>/<class>/ for the six products and the three data
splits (train / val / test) so evaluation uses images never seen in training.

Generative AI acknowledgement: drafted with Doubao and revised with Claude
(Anthropic); reviewed and tested by the group, and acknowledged in the report.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Must match CLASS_NAMES in COSC_4117EL_A3_G6-main.py (alphabetical order).
CLASS_NAMES = [
    "bottled_water",
    "chips_bag",
    "cola_can",
    "instant_noodle_cup",
    "juice_box",
    "milk_bottle",
]
SPLITS = ["train", "val", "test"]

CORE_PACKAGES = ["torch", "torchvision", "opencv-python", "numpy", "pillow", "matplotlib"]
YOLO_PACKAGES = ["ultralytics"]


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Set up the Assignment 3 environment.")
    parser.add_argument("--skip-install", action="store_true", help="Only create folders.")
    parser.add_argument("--skip-yolo", action="store_true", help="Do not install the bonus YOLO package.")
    return parser.parse_args()


def install_packages(packages: list[str]) -> None:
    """Install packages into the Python interpreter running this script."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def create_folders() -> None:
    """Create dataset/<split>/<class>/ and outputs/."""
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Install dependencies (optional) and build the folder structure."""
    args = parse_args()
    if not args.skip_install:
        install_packages(CORE_PACKAGES)
        if not args.skip_yolo:
            install_packages(YOLO_PACKAGES)
    create_folders()
    print("Setup complete. Project root:", PROJECT_ROOT)
    print("Next: collect images, e.g.")
    print("  python COSC_4117EL_A3_G6-main.py collect --class-name cola_can --split train --num 50")


if __name__ == "__main__":
    main()
