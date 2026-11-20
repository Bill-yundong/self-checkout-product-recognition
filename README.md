# Real-Time Retail Product Recognition for Self-Checkout

COSC-4117EL-01 Artificial Intelligence, Fall 2026, Assignment 3, Group 6
(Qingdong Wu, Yuhang Zhao)

This project classifies six retail products in real time from a webcam feed. It uses ResNet18 transfer learning and is built on the course sample code (`CNN_ResNet_Sample.py`). For the bonus part, it also compares ResNet18 against YOLOv8n-cls.

## Product classes

Classes are listed in alphabetical order, which is the order the model uses:

| ID | Class |
|---|---|
| 0 | `bottled_water` |
| 1 | `chips_bag` |
| 2 | `cola_can` |
| 3 | `instant_noodle_cup` |
| 4 | `juice_box` |
| 5 | `milk_bottle` |

## Files

| File | Purpose |
|---|---|
| `COSC_4117EL_A3_G6-setup.py` | Installs dependencies and creates `dataset/<split>/<class>/` and `outputs/`. |
| `COSC_4117EL_A3_G6-main.py` | Handles data collection, ResNet18 training and evaluation, the real-time demo, and the YOLO bonus comparison. |

The comments in `-main.py` tag each part as `[SAMPLE]` (kept from the course sample), `[MODIFIED]` (changed from the sample, with the reason) or `[NEW]` (added by the group).

Only the two `.py` files and the PDF report are submitted. Do not upload a ZIP file.

## 1. Set up the environment

You can optionally create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then run the setup script:

```bash
python COSC_4117EL_A3_G6-setup.py                 # install packages and create folders
python COSC_4117EL_A3_G6-setup.py --skip-install  # create folders only
python COSC_4117EL_A3_G6-setup.py --skip-yolo     # skip the bonus YOLO package
```

## 2. Collect webcam images

```bash
python COSC_4117EL_A3_G6-main.py collect --class-name cola_can --split train --num 50
```

Repeat this command for all 6 classes and all 3 splits (`train`, `val`, `test`). Recommended number of images per class:

- train: 40 to 60
- val: 10 to 15
- test: 10 to 15

While collecting, place the product inside the green box. Only that region is saved, and the live demo classifies the same region.

Keys:

- `c`: capture one image
- `a`: toggle automatic capture (one image every 1.2 s)
- `q`: quit

Tips:

- Vary the angle, distance, lighting and background.
- Collect each split in a separate session, because neighbouring frames are nearly identical and would leak between train and test.

## 3. Train ResNet18

```bash
python COSC_4117EL_A3_G6-main.py train
python COSC_4117EL_A3_G6-main.py train --epochs 20 --unfreeze-last-block   # also fine-tune layer4
```

Options:

- `--epochs`: default 15
- `--batch-size`: default 8
- `--lr`: default 0.001

Outputs:

- `outputs/resnet18_retail_best.pth`: best validation model, with the class order stored in it
- `outputs/resnet_training_history.csv`
- `outputs/resnet_training_curves.png`

## 4. Evaluate on the test set

```bash
python COSC_4117EL_A3_G6-main.py evaluate
```

Outputs:

- `outputs/resnet_metrics.csv`: per-class precision, recall, F1 and support, plus the macro average
- `outputs/resnet_summary.csv`: accuracy, macro P/R/F1, mean latency (ms/image), maximum FPS, parameter count, device
- `outputs/resnet_confusion_matrix.png`

## 5. Real-time demo

```bash
python COSC_4117EL_A3_G6-main.py realtime
python COSC_4117EL_A3_G6-main.py realtime --camera 0 --threshold 0.70 --smooth 5
```

The demo shows the product name and confidence, the model latency and the FPS. Probabilities are averaged over the last `--smooth` frames. When the confidence is below `--threshold`, the box turns red and the customer is asked to reposition the product. Press `q` to quit.

## 6. Bonus: compare ResNet18 with YOLO

YOLOv8n-cls is trained on the same `dataset/` folders, so no bounding-box annotation is needed and both models are tested on the same images.

```bash
python COSC_4117EL_A3_G6-main.py train-yolo                 # --epochs 30 --batch-size 8
python COSC_4117EL_A3_G6-main.py evaluate --model yolo
python COSC_4117EL_A3_G6-main.py realtime --model yolo
python COSC_4117EL_A3_G6-main.py compare
```

Outputs:

- `outputs/yolov8n_cls_retail/weights/best.pt`
- `outputs/yolo_metrics.csv`, `outputs/yolo_summary.csv`, `outputs/yolo_confusion_matrix.png`
- `outputs/resnet_vs_yolo_comparison.csv` and `.png`: measured accuracy, P/R/F1, latency and parameter counts for both models

## 7. Report and video

1. Copy the results from `outputs/` into the report `COSC_4117EL_A3_G6.pdf`.
2. Record the demo, upload it to YouTube and put the link in the report.

The video should show:

- the setup
- all six products
- live predictions with confidence and FPS
- at least one low-confidence or misclassified case
- the YOLO comparison

## Generative AI statement

The initial code structure was drafted with Doubao. The code was then revised with Claude (Anthropic) to build it on the course sample and add the evaluation, smoothing and YOLO comparison. The group reviewed and tested all generated code. This use is acknowledged in the code comments and in the report, as the assignment requires.
