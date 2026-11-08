# Roboflow Dataset Version 3: Raspberry Pi Camera Enclosure Detection

**Roboflow Workspace:** `jaishanth-s-workspace`
**Roboflow Project:** `m1_m2`
**Dataset Version:** `3`
**Project URL:** [https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)
**Export Format:** YOLOv8 (Ultralytics YOLO compatible for YOLO26 Nano)
**Target Model:** YOLO26 Nano (`yolo26n`)
**Download Date:** November 8, 2026
**License:** Private (Academic & Project Use)

---

## 1. Dataset Overview

This dataset contains the official **Version 3** training, validation, and testing images exported from the Roboflow cloud platform for the **Raspberry Pi Camera Enclosure Detection** Edge AI project.

The target object is the custom 3D-printed Raspberry Pi camera enclosure designed by **Jaishanth** as an **Individual Design and Research Project (IDRP)**.

```
datasets/Roboflow_M1_M2_V3/
├── README.md                          # This specification document
├── data.yaml                          # YOLO dataset configuration
├── train/
│   ├── images/                        # 230 training images (640x640)
│   └── labels/                        # 230 YOLO label files
├── valid/
│   ├── images/                        # 30 validation images (640x640)
│   └── labels/                        # 30 YOLO label files
└── test/
    ├── images/                        # 14 test images (640x640)
    └── labels/                        # 14 YOLO label files
```

---

## 2. Preprocessing & Augmentation Pipeline

### Preprocessing Applied
- **Auto-Orientation:** Strips EXIF orientation tags and normalizes pixel orientation.
- **Resolution Normalization:** Resized to **640 × 640** pixels preserving aspect ratio using **Fit (black edges)** letterboxing.

### Augmentation Parameters (Applied to Training Set Only)
To enhance model robustness against real-world lighting, reflections, and orientation variations, the following augmentations were applied to generate 2 versions per source training image:
- **Horizontal Flip:** 50% probability
- **Random Rotation:** Between $-10^\circ$ and $+10^\circ$
- **Brightness Adjustment:** Random variation between $-15\%$ and $+15\%$
- **Saturation Adjustment:** Random variation between $-15\%$ and $+15\%$

---

## 3. Image Counts & Split Statistics

| Split | Images | % of Total | Labels | Object Annotations | True Negatives (Empty Labels) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **230** | 83.94% | 230 | **200** (176 bboxes + 24 polygons) | **30** |
| **Valid** | **30** | 10.95% | 30 | **27** (24 bboxes + 3 polygons) | **3** |
| **Test** | **14** | 5.11% | 14 | **12** (11 bboxes + 1 polygon) | **2** |
| **Total** | **274** | **100.0%** | **274** | **239** (211 bboxes + 28 polygons) | **35** |

### Explanation of Roboflow Display Count vs. Source Images
The Roboflow user interface displays **274 images**. It is critical to distinguish between unique source captures and augmented exports:
- **Original Source Dataset (159 Images):**
  - **139 Positive Enclosure Photographs:** Captured under real-world desktop and handheld conditions.
  - **20 True Negative Background Photographs:** Walls, curtains, desks, and empty hands with no enclosure present.
  - **Source Splits:** Train (115 images: 100 positive + 15 negative), Valid (30 unaugmented images: 27 positive + 3 negative), Test (14 unaugmented images: 12 positive + 2 negative).
- **Augmented Exported Dataset (274 Images):**
  - The 115 original training images underwent a **$2\times$ augmentation pipeline** (horizontal flip, $\pm 10^\circ$ rotation, $\pm 15\%$ brightness, $\pm 15\%$ saturation), expanding the training split from 115 to **230 training images**.
  - Validation (30 images) and Test (14 images) splits were preserved strictly unaugmented to guarantee uncompromised, unbiased evaluation.
  - **Export Sum:** $230\text{ (train)} + 30\text{ (valid)} + 14\text{ (test)} = \mathbf{274}\text{ images}$.
  - *Note:* Not all 274 exported images represent independently captured photographs; 115 training source images produced 230 training images via 2x augmentation.

---

## 4. Class & Annotation Specifications

- **Number of Classes (`nc`):** `1`
- **Class ID:** `0`
- **Class Label:** `'RASPBERY PI CAM ENCLOSURE'`
- **Annotation Breakdown:**
  - **Standard YOLO Bounding Boxes:** **211 annotations** formatted as `0 <x_center> <y_center> <width> <height>` with normalized coordinates in $[0.0, 1.0]$.
  - **YOLO Polygon Segmentations:** **28 annotations** formatted as `0 <x1> <y1> <x2> <y2> ... <xn> <yn>` representing fine-grained contour polygons.
  - **True Negative Backgrounds:** **35 empty label files** (`.txt` size 0 bytes: 30 train, 3 valid, 2 test) providing explicit negative rejection signals.
  - **Total Ground Truth Targets:** **239 annotations** across 274 image files.

### Annotation Compatibility Analysis for YOLO26 Nano
- **Format Verification:** The 28 polygon annotations are represented as native Roboflow segmentation vertex coordinates rather than pre-converted axis-aligned boxes.
- **Pipeline Compatibility:** Ultralytics YOLO26 Nano (`ultralytics` engine) natively supports polygon segmentation coordinates during object detection training (`yolo detect train`). The data loader automatically computes the minimum enclosing bounding rectangle ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$) on the fly for detection loss computation while retaining polygon boundaries for data augmentation transformations.
- **Data Preservation:** In accordance with strict data integrity standards, all original polygon coordinate streams are preserved completely intact in their exported format without manual or destructive down-conversion.

---

## 5. Dataset Integrity Verification

A complete programmatic audit was executed across all 274 images and labels:

| Verification Check | Result | Details |
| :--- | :---: | :--- |
| **Missing Image Files** | **0** | Every label has an exact corresponding image |
| **Missing Label Files** | **0** | Every image has an exact corresponding label |
| **Image Resolution** | **100% Pass** | All 274 images measure exactly $640 \times 640$ |
| **Coordinate Bounds** | **100% Pass** | All bounding box & polygon points reside strictly in $[0.0, 1.0]$ |
| **Cross-Split Duplicates** | **0** | Zero hash collisions between train, valid, and test splits |
| **Intra-Split Duplicates** | **0** | Zero duplicate images within any split |
| **Total Disk Footprint** | **8.65 MB** | 9,071,762 bytes (well below GitHub 100 MB file limits) |

---

## 6. Model Training & Evaluation Correlation

The trained model weights in this repository ([`models/best.pt`](../../models/best.pt) and [`models/best.onnx`](../../models/best.onnx)) were trained on this Version 3 dataset:
- **Validation mAP@50:** **99.28%**
- **Test Set mAP@50:** **99.31%**
- **Precision:** **100.0%**
- **Recall:** **91.7%**
- **Background False Positive Rate:** **0.0%** (100% true negative rejection)

Genuine visual annotation previews are preserved in [`evidence/M1/04_dataset_v1/roboflow_v3_previews/`](../../evidence/M1/04_dataset_v1/roboflow_v3_previews/).
