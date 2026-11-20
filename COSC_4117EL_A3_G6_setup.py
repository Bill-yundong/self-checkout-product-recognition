# Generative AI note: environment-setup boilerplate was prepared with Doubao
# assistance and reviewed for this assignment.
"""
Environment and project-folder builder for COSC-4117EL Assignment 3.

Run this file first:
    python COSC_4117EL_A3_G6_setup.py

Optional arguments:
    --skip-install      Do not install Python packages.
    --skip-yolo         Do not install or configure the optional YOLO workflow.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "bottled_water",
    "cola_can",
    "juice_box",
    "chips_bag",
    "milk_bottle",
    "instant_noodle_cup",
]

CORE_PACKAGES = [
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "pillow",
    "matplotlib",
    "PyYAML",
]
YOLO_PACKAGES = ["ultralytics"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Assignment 3 environment.")
    parser.add_argument("--skip-install", action="store_true", help="Only create folders and configuration.")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO dependency and YOLO configuration.")
    parser.add_argument("--skip-pip-upgrade", action="store_true", help="Do not upgrade pip.")
    return parser.parse_args()


def run_pip(packages: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def install_dependencies(include_yolo: bool, upgrade_pip: bool) -> None:
    if upgrade_pip:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_pip(CORE_PACKAGES)
    if include_yolo:
        run_pip(YOLO_PACKAGES)


def create_folders() -> None:
    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_yolo_config() -> None:
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {YOLO_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (YOLO_DIR / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.skip_install:
        install_dependencies(
            include_yolo=not args.skip_yolo,
            upgrade_pip=not args.skip_pip_upgrade,
        )
    create_folders()
    if not args.skip_yolo:
        write_yolo_config()
    print("Environment setup completed.")
    print("Project root:", PROJECT_ROOT)


if __name__ == "__main__":
    main()
