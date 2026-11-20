"""Generate the Assignment 3 PDF report with reportlab."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ListStyle, ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import CLASS_NAMES, REPORT_DIR

OUTPUT_PATH = REPORT_DIR / "COSC_4117EL_A3_G6.pdf"


def bullet(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=22,
    )


def numbered(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=12) for item in items],
        bulletType="1",
        leftIndent=22,
    )


def make_table(data: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", parent=styles["BodyText"], alignment=4, leading=14))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))
    body = styles["Justify"]
    heading1 = styles["Heading1"]
    heading2 = styles["Heading2"]

    story = []
    story.append(Paragraph("Real-Time Image Classification with CNNs Using Transfer Learning", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Application: Retail Self-Checkout Product Recognition", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Course: COSC-4117EL-01 Artificial Intelligence", body))
    story.append(Paragraph("Term: Fall 2026", body))
    story.append(Paragraph("Group: G6", body))
    story.append(Paragraph("Group members: [Insert full names]", body))
    story.append(Paragraph("YouTube demonstration: [Insert YouTube URL]", body))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Abstract", heading1))
    story.append(Paragraph(
        "This project investigates real-time recognition of packaged retail products for a simulated self-checkout "
        "station. A pretrained ResNet18 convolutional neural network is adapted through transfer learning to classify "
        "six common drink and snack products. OpenCV captures webcam frames, while a fixed region of interest defines "
        "the product placement area. The final system displays the predicted product label, confidence score, and "
        "real-time frame rate. A lightweight YOLOv8 detector is also discussed as the bonus comparison because it can "
        "localize products with bounding boxes and handle multiple products more naturally than a classification-only model.",
        body,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Part 1: Problem Proposal and Justification", heading1))
    story.append(Paragraph("1. Problem Identification", heading2))
    story.append(Paragraph(
        "Traditional self-checkout systems require customers to locate barcodes and scan each product individually. "
        "This process can be slow when the barcode is difficult to find, the package is rotated, or multiple small "
        "items are being purchased. The proposed system uses a webcam to recognize products placed in a checkout area "
        "and immediately predicts their category. The application is relevant to new retail environments because "
        "vision-based recognition can reduce manual scanning effort and support faster customer service.",
        body,
    ))
    story.append(Paragraph("The first version classifies six products:", body))
    class_table = [["Class ID", "Class name", "Product example"]]
    examples = [
        "Bottled water", "Canned cola", "Juice carton", "Chips bag",
        "Milk or yogurt bottle", "Instant noodle cup or snack cup",
    ]
    for index, (name, example) in enumerate(zip(CLASS_NAMES, examples)):
        class_table.append([str(index), name, example])
    story.append(make_table(class_table, [0.8 * inch, 2.3 * inch, 3.2 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. Significance and Real-World Impact", heading2))
    story.append(bullet([
        "<b>Checkout efficiency:</b> product recognition can reduce the time spent searching for barcodes.",
        "<b>Customer experience:</b> placing an item in a defined area is simpler than repeated manual scanning.",
        "<b>Operational support:</b> recognized product categories can later support checkout analytics and inventory workflows.",
        "<b>Technical comparison:</b> the project exposes the trade-off between CNN classification and YOLO object detection.",
    ], body))

    story.append(Paragraph("3. Dataset Description", heading2))
    story.append(Paragraph(
        "The dataset is collected using a webcam in a simulated self-checkout setting. Each image contains one product "
        "inside the recognition region. The recommended split is 40 to 60 training images, 10 to 15 validation images, "
        "and 10 to 15 test images per class. Images should cover front and side views, slight rotations, different "
        "distances, centered and offset positions, lighting changes, and simple background variations.",
        body,
    ))
    story.append(Paragraph(
        "The first version does not include payment, price lookup, inventory management, or large-scale deployment. "
        "It focuses on single-product recognition in a fixed region of interest.",
        body,
    ))

    story.append(Paragraph("4. Data Augmentation", heading2))
    story.append(bullet([
        "Random resized cropping and resizing to 224 by 224 pixels.",
        "Random horizontal flips and rotations up to 12 degrees.",
        "Brightness, contrast, and saturation jitter.",
        "ImageNet mean and standard deviation normalization.",
    ], body))
    story.append(Paragraph(
        "Augmentation is applied only to the training split. Validation and test images remain unaugmented so that "
        "evaluation reflects the final inference pipeline.",
        body,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Part 2: Implementation Guide", heading1))
    story.append(Paragraph("1. Transfer Learning Approach", heading2))
    story.append(Paragraph(
        "ResNet18 is selected because it is compact, well supported in PyTorch, and compatible with the course sample "
        "code. Its ImageNet-pretrained weights already encode general visual features such as edges, textures, colors, "
        "and object shapes. These features are useful for retail packaging even when the custom dataset is small.",
        body,
    ))
    story.append(numbered([
        "Load ResNet18 with ImageNet-pretrained weights.",
        "Freeze the convolutional backbone so that its general features are retained.",
        "Replace the final fully connected layer with a six-class linear layer.",
        "Train the classifier head using cross-entropy loss and Adam optimization.",
        "Optionally unfreeze layer4 for limited fine-tuning when more data are available.",
    ], body))

    story.append(Paragraph("2. Architecture Modification", heading2))
    architecture_table = [
        ["Component", "Configuration"],
        ["Input", "224 x 224 RGB webcam image"],
        ["Backbone", "Pretrained ResNet18 convolutional layers"],
        ["Training mode", "Backbone frozen; custom fully connected layer trained"],
        ["Output layer", "Linear layer with six product classes"],
        ["Loss", "CrossEntropyLoss"],
        ["Optimizer", "Adam, default learning rate 0.001"],
    ]
    story.append(make_table(architecture_table, [2.0 * inch, 4.3 * inch]))

    story.append(Paragraph("3. Real-Time Processing", heading2))
    story.append(numbered([
        "OpenCV reads a frame from the webcam.",
        "A fixed green rectangle marks the product recognition region.",
        "The region is converted from BGR to RGB and resized to 224 by 224.",
        "The image is normalized and converted to a PyTorch batch tensor.",
        "ResNet18 produces class probabilities.",
        "The softmax confidence and predicted class are displayed.",
        "If confidence is below the threshold, the system asks the user to reposition the product.",
    ], body))

    story.append(Paragraph("4. Libraries and Tools", heading2))
    story.append(bullet([
        "PyTorch and torchvision provide model construction, pretrained weights, datasets, and tensor operations.",
        "OpenCV handles webcam access, frame display, and region-of-interest visualization.",
        "Matplotlib generates the confusion matrix figure.",
        "Ultralytics YOLOv8 is used for the optional object-detection comparison.",
    ], body))

    story.append(PageBreak())
    story.append(Paragraph("Part 3: Performance Evaluation and Testing", heading1))
    story.append(Paragraph("1. Evaluation Metrics", heading2))
    story.append(bullet([
        "<b>Accuracy</b> = correct predictions / total predictions.",
        "<b>Precision</b> = TP / (TP + FP); important because false product predictions can cause wrong checkout.",
        "<b>Recall</b> = TP / (TP + FN); measures how often a product is missed.",
        "<b>F1-score</b> = 2 x Precision x Recall / (Precision + Recall); balances precision and recall.",
        "<b>Latency and FPS</b> measure real-time suitability on the selected hardware.",
    ], body))

    result_table = [
        ["Metric", "Result"],
        ["Test accuracy", "[Insert value from outputs/resnet_metrics.csv]"],
        ["Macro precision", "[Insert after test evaluation]"],
        ["Macro recall", "[Insert after test evaluation]"],
        ["Macro F1-score", "[Insert after test evaluation]"],
        ["Average inference time", "[Insert measured milliseconds per frame]"],
        ["Real-time FPS", "[Insert measured FPS]"],
        ["Device", "[CPU or GPU model]"],
    ]
    story.append(make_table(result_table, [2.5 * inch, 3.8 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    per_class_table = [["Class", "Precision", "Recall", "F1", "Support"]]
    for name in CLASS_NAMES:
        per_class_table.append([name, "[insert]", "[insert]", "[insert]", "[insert]"])
    story.append(make_table(per_class_table, [1.9 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch]))

    story.append(Paragraph("2. Anticipated Challenges", heading2))
    challenge_table = [
        ["Challenge", "Mitigation"],
        ["Lighting changes", "Collect lighting variations; use brightness and contrast augmentation."],
        ["Angled packaging", "Collect multiple viewpoints and use rotation augmentation."],
        ["Reflective packages", "Avoid direct glare and include reflective samples in the dataset."],
        ["Occlusion", "Require complete placement in the ROI; warn when confidence is low."],
        ["Similar products", "Choose visually distinct products and analyze the confusion matrix."],
        ["Low FPS", "Use ResNet18, lower camera resolution, and skip unnecessary processing."],
    ]
    story.append(make_table(challenge_table, [2.0 * inch, 4.3 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("Bonus: ResNet18 versus YOLO Model Comparison", heading1))
    story.append(Paragraph(
        "ResNet18 performs image classification and predicts the product category represented in the input image. It "
        "does not directly output a product location. YOLO performs object detection and returns both the category and "
        "a bounding box. This makes YOLO more appropriate for product localization, multi-product detection, and "
        "checkout scenes where the product may not fill the entire frame. However, YOLO requires bounding-box "
        "annotation, additional detector-specific tuning, and generally more complex implementation.",
        body,
    ))
    comparison_table = [
        ["Dimension", "ResNet18", "YOLOv8n"],
        ["Task", "Classification", "Detection"],
        ["Position output", "No", "Yes, bounding box"],
        ["Multiple products", "Limited", "Supported"],
        ["Annotation", "Class folders", "Bounding boxes"],
        ["Model complexity", "Lower", "Higher"],
        ["Best fit", "Single-product ROI classification", "Localization and multi-product detection"],
    ]
    story.append(make_table(comparison_table, [1.7 * inch, 2.3 * inch, 2.3 * inch]))

    story.append(Paragraph("Video Demonstration", heading1))
    story.append(Paragraph(
        "The demonstration video will show the webcam setup, six product classes, real-time ResNet18 predictions, "
        "confidence values, a low-confidence or misclassification case, and YOLO detection boxes if the bonus workflow "
        "is completed. The YouTube link must be inserted on the cover page and should be accessible to the instructor.",
        body,
    ))

    story.append(Paragraph("Group Contributions", heading1))
    story.append(Paragraph(
        "[Insert each member's contribution, such as data collection, model training, evaluation, YOLO comparison, "
        "report writing, and video preparation.]",
        body,
    ))

    story.append(Paragraph("Generative AI Acknowledgement", heading1))
    story.append(Paragraph(
        "Generative AI assistance was used for boilerplate code, implementation structure, and report drafting. The "
        "generated material was reviewed, adapted, and tested by the group. All AI-generated text and code is "
        "acknowledged here and in the code comments in accordance with the assignment instructions.",
        body,
    ))

    story.append(Paragraph("References", heading1))
    story.append(bullet([
        "He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.",
        "PyTorch Documentation. torchvision.models.resnet18.",
        "OpenCV Documentation. Video Capture and High-Level GUI.",
        "Ultralytics Documentation. YOLOv8 Documentation.",
        "Course Assignment 3 Handout, COSC-4117EL, Fall 2026.",
    ], styles["Small"]))

    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=letter, rightMargin=0.7 * inch, leftMargin=0.7 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(story)
    print("PDF report generated:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
