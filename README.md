# Real-Time Retail Product Recognition for Self-Checkout

COSC-4117EL Assignment 3: real-time webcam classification of six retail products using ResNet18 transfer learning, with an optional YOLOv8 comparison for the bonus requirement.

## Product classes

1. `bottled_water`
2. `cola_can`
3. `juice_box`
4. `chips_bag`
5. `milk_bottle`
6. `instant_noodle_cup`

## Merged code files

The submission workflow uses two root-level Python files:

| File | Purpose |
|---|---|
| `COSC_4117EL_A3_G6_setup.py` | Builds the environment, installs dependencies, creates dataset folders, and generates the YOLO configuration file. |
| `COSC_4117EL_A3_G6_main.py` | Contains all assignment functionality: webcam collection, ResNet18 training, evaluation, real-time inference, YOLO training/detection, and model comparison. |

The environment-verification script and PDF-report generator are auxiliary files and are not merged into the two main code files.

## 1. Optional virtual environment

Open a terminal in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the virtual environment activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Build the environment and project folders

Run the setup script:

```powershell
python COSC_4117EL_A3_G6_setup.py
```

This script installs the required packages, creates the ResNet and YOLO folder structures, and writes `yolo_dataset/data.yaml`.

Useful options:

```powershell
# Create folders and configuration without installing packages
python COSC_4117EL_A3_G6_setup.py --skip-install

# Skip the optional YOLO dependency and configuration
python COSC_4117EL_A3_G6_setup.py --skip-yolo
```

## 3. Collect webcam images

Use the main script to collect images for one class and split:

```powershell
python COSC_4117EL_A3_G6_main.py collect --class-name bottled_water --split train --num 50
```

Repeat the command for every class and every split. Recommended minimum counts:

- train: 40 images per class
- validation: 10 images per class
- test: 10 images per class

Collect multiple angles, distances, lighting conditions, backgrounds, and slight rotations. Do not duplicate the same image across train, validation, and test.

During collection:

- press `c` to capture one image;
- press `a` to toggle automatic capture every 1.2 seconds;
- press `q` to quit.

## 4. Train ResNet18

```powershell
python COSC_4117EL_A3_G6_main.py train --epochs 15 --batch-size 8
```

Optional limited fine-tuning of the final residual block:

```powershell
python COSC_4117EL_A3_G6_main.py train --epochs 20 --unfreeze-last-block
```

The best model is saved as:

```text
outputs/resnet18_retail_best.pth
```

Training history is saved as:

```text
outputs/training_history.csv
```

## 5. Evaluate ResNet18

```powershell
python COSC_4117EL_A3_G6_main.py evaluate
```

Outputs:

- `outputs/resnet_metrics.csv`
- `outputs/resnet_confusion_matrix.png`

The evaluation script calculates accuracy, per-class precision, recall, F1-score, and support without relying on scikit-learn.

## 6. Run real-time ResNet18 inference

```powershell
python COSC_4117EL_A3_G6_main.py realtime
```

Press `q` to quit. The green rectangle is the fixed product recognition region. If confidence is below the threshold, the program asks the user to reposition the product instead of forcing a prediction.

Optional arguments:

```powershell
python COSC_4117EL_A3_G6_main.py realtime --camera 0 --threshold 0.70
```

## 7. Optional YOLOv8 bonus workflow

First ensure the setup script was not run with `--skip-yolo`. The YOLO configuration file should be:

```text
yolo_dataset/data.yaml
```

Place annotated images and labels under:

```text
yolo_dataset/images/train
yolo_dataset/images/val
yolo_dataset/images/test
yolo_dataset/labels/train
yolo_dataset/labels/val
yolo_dataset/labels/test
```

Train YOLOv8:

```powershell
python COSC_4117EL_A3_G6_main.py train-yolo --epochs 30 --imgsz 640 --batch-size 8
```

Run real-time YOLO detection:

```powershell
python COSC_4117EL_A3_G6_main.py yolo-realtime
```

Create the qualitative ResNet versus YOLO comparison table:

```powershell
python COSC_4117EL_A3_G6_main.py compare
```

The comparison table is saved as:

```text
outputs/resnet_yolo_comparison.csv
```

## 8. Report and video demonstration

The report template is provided as:

```text
report/COSC_4117EL_A3_G6.pdf
```

After running the experiments, replace the placeholder result fields and YouTube link with the actual values. The video should show:

- the webcam and self-checkout setup;
- all six product classes;
- real-time ResNet18 predictions and confidence values;
- at least one low-confidence or misclassification case;
- YOLO detection boxes if the bonus workflow is completed;
- the final model comparison and conclusions.

Upload the video to YouTube and insert the link in the PDF report.

## 9. Auxiliary files

The `src/` folder contains the earlier modular implementation and auxiliary scripts. The separate verification and PDF-generation scripts remain available there, but they are not required for the two-file main workflow.

Rename the files if your group number is not G6. Do not upload a ZIP file.

## Generative AI statement

Generative AI assistance was used for boilerplate code, implementation structure, and report drafting. The proposed code was reviewed, adapted, and validated for the assignment. Any generated text or code is acknowledged in both the code comments and the report, as required by the assignment instructions.
