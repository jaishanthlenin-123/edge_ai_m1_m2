# Roboflow Project Information & Version 3 Overview

**Project Name:** `M1_M2`
**Roboflow Workspace:** `jaishanth-s-workspace`
**Latest Dataset Version:** `Version 3` (Generated November 5, 2026)
**Roboflow Direct URL:** [https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)
**Local Dataset Path:** [`datasets/Roboflow_M1_M2_V3/`](../../../datasets/Roboflow_M1_M2_V3/)

---

## 1. Important Context: Version 3 vs. Historical M1 Dataset

> [!IMPORTANT]
> **Dataset Version Distinction:**
> The original **Milestone 1 (M1) Dataset Audit** ([`dataset_audit_report.md`](../dataset_audit_report.md) and [`annotation_errors.csv`](../annotation_errors.csv)) was performed on the initial **124-image baseline dataset** (Version 1).
>
> **Version 3** is the latest, production-ready Roboflow dataset version, developed through subsequent iterations to resolve the root causes identified in the M1 audit. Version 3 must not be conflated with the historical 124-image baseline, nor does it replace the historical audit records.

---

## 2. Roboflow Version 3 Specifications

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Workspace** | `jaishanth-s-workspace` | Roboflow Cloud |
| **Project** | `m1_m2` | Single-class object detection |
| **Version ID** | `m1_m2-3` | Version 3 |
| **Target Class** | `RASPBERY PI CAM ENCLOSURE` | Single target class (ID `0`) |
| **Original Source Images** | **159 images** | 139 positive enclosure + 20 negative background |
| **Exported Total Images** | **274 images** | Includes 2x data augmentation on training set |
| **Split Breakdown** | **Train:** 230 imgs (83.9%) \| **Valid:** 30 imgs (11.0%) \| **Test:** 14 imgs (5.1%) |
| **Input Resolution** | **640 × 640** | Fit (black edges) letterbox |
| **Augmentation** | Flip (H), Rotate ($\pm 10^\circ$), Brightness ($\pm 15\%$), Saturation ($\pm 15\%$) | 2 versions per train image |
| **Local Download Path** | [`datasets/Roboflow_M1_M2_V3/`](../../../datasets/Roboflow_M1_M2_V3/) | Full YOLOv8 format export |

---

## 3. Key Evolution from M1 Baseline to Version 3

1. **Resolution of Annotation Errors:** The loose bounding boxes and misalignments documented in [`annotation_errors.csv`](../annotation_errors.csv) were tightened around the actual physical enclosure body.
2. **Integration of True Negatives:** Version 1 lacked negative background images, resulting in false positives on curtains and plain walls. Version 3 incorporates **20 true negative background samples** (augmented to 35 empty label files in the export), achieving **0.0% false positives** on background scenes.
3. **Model Convergence:** The fine-tuned weights ([`models/best.pt`](../../../models/best.pt) and [`models/best.onnx`](../../../models/best.onnx)) were trained on Version 3, achieving **99.31% mAP@50** and **100.0% precision** on the held-out test set.

---

## 4. Visual Previews & Documentation

- **Annotation Previews:** [`evidence/M1/04_dataset_v1/roboflow_v3_previews/`](./roboflow_v3_previews/) – Genuine side-by-side raw vs. annotated examples from the Version 3 dataset.
- **Dataset Version History:** [`evidence/M1/dataset_version_history.md`](../dataset_version_history.md) – Full timeline of dataset evolution.
- **Dataset Download Directory:** [`datasets/Roboflow_M1_M2_V3/`](../../../datasets/Roboflow_M1_M2_V3/) – Local copy of all images and YOLO label files.
