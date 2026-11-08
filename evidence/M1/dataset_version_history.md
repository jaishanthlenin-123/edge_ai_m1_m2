# Dataset Version History: M1 Baseline to Roboflow Version 3

**Project:** Raspberry Pi Camera Enclosure Detection using Edge AI
**Author:** Jaishanth
**Roboflow Workspace:** `jaishanth-s-workspace`
**Roboflow Project:** `m1_m2`
**Production Dataset URL:** [https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)

---

## 1. Executive Summary

This document details the evolution of the dataset used in the Edge AI detection pipeline. It clarifies the relationship between the original **124-image dataset audited during Milestone 1 (M1)** and the **274-image Version 3 dataset exported from Roboflow** that was used to train the production YOLO26 Nano model.

```
+-------------------------------------------------------------------------------+
|                             DATASET EVOLUTION TIMELINE                        |
+-------------------------------------------------------------------------------+
|  Version 1 (Initial Audit)                                                    |
|  - 124 images (Train: 88, Valid: 24, Test: 12)                                |
|  - Audit revealed loose boxes & fingers enclosed                              |
|  - 0 negative images -> High false positive (73.2% on curtains)               |
+-------------------------------------------------------------------------------+
                                      |
                                      v Annotation tightening & negative collection
+-------------------------------------------------------------------------------+
|  Version 2 (Intermediate Refinement)                                          |
|  - Ground truth box adjustments                                               |
|  - Initial background collection                                              |
+-------------------------------------------------------------------------------+
                                      |
                                      v Augmentation & production cloud export
+-------------------------------------------------------------------------------+
|  Version 3 (Final Production Dataset)                                         |
|  - 159 source images (139 positive + 20 true negative backgrounds)            |
|  - 274 augmented images (Train: 230, Valid: 30, Test: 14)                     |
|  - 640x640 Fit resolution                                                     |
|  - False positive rate: 0.0% (100% background rejection)                      |
|  - Trained Model mAP@50: 99.31% on held-out test set                          |
+-------------------------------------------------------------------------------+
```

---

## 2. Detailed Dataset Version Comparison

| Metric / Feature | Version 1 (M1 Baseline) | Version 2 (Intermediate) | Version 3 (Production M1/M2) |
| :--- | :---: | :---: | :---: |
| **Source Images** | 124 images | 138 images | **159 images** |
| **Augmented Total Images** | 124 (No augmentation) | 138 (No augmentation) | **274 images** (2x train augmentation) |
| **Train Set Count** | 88 images (71.0%) | 98 images (71.0%) | **230 images** (83.94% augmented) |
| **Validation Set Count** | 24 images (19.4%) | 27 images (19.6%) | **30 images** (10.95% unaugmented) |
| **Test Set Count** | 12 images (9.6%) | 13 images (9.4%) | **14 images** (5.11% unaugmented) |
| **True Negative Images** | **0 images** | 8 images | **20 source / 35 exported images** |
| **Target Class Label** | `'RASPBERY PI CAM ENCLOSURE'` | `'RASPBERY PI CAM ENCLOSURE'` | `'RASPBERY PI CAM ENCLOSURE'` |
| **Resolution** | Various / Letterbox | 640 × 640 Fit | **640 × 640 Fit (black edges)** |
| **Augmentation Applied** | None | None | Flip (H), Rotate ($\pm 10^\circ$), Brightness ($\pm 15\%$), Saturation ($\pm 15\%$) |
| **False Positive on Curtains**| **73.2% activation** | 18.4% activation | **0.0% activation (Completely eliminated)** |
| **Test Set mAP@50** | Baseline (~82%) | ~91.5% | **99.31%** |
| **Storage Location** | Historical M1 Audit Logs | Intermediate Scratch | [`datasets/Roboflow_M1_M2_V3/`](../../datasets/Roboflow_M1_M2_V3/) |

### Source vs. Augmented Breakdown (Version 3)
- **Source Counts (159 Images):** 115 training images, 30 validation images, 14 test images.
- **Exported Counts (274 Images):** 230 training images, 30 validation images, 14 test images.
- *Clarification:* Not all 274 exported images represent independently captured photographs. The 115 original training photographs were expanded to 230 images through a $2\times$ augmentation pipeline, whereas validation (30) and test (14) splits were kept unaugmented to ensure uncorrupted evaluation.
- **Annotation Breakdown:** 211 standard YOLO bounding boxes, 28 polygon segmentations, and 35 empty-label background images.
- **YOLO26 Nano Compatibility:** Ultralytics YOLO26 Nano handles polygon segmentations natively during detection training by dynamically computing enclosing bounding boxes on the fly, preserving the original polygon annotations without data loss.

---

## 3. Preservation of Historical M1 Audit Evidence

To maintain rigorous scientific and academic integrity:
- The historical Milestone 1 records ([`dataset_audit_report.md`](./dataset_audit_report.md), [`annotation_errors.csv`](./annotation_errors.csv), and [`class_distribution.csv`](./class_distribution.csv)) remain completely unmodified.
- They serve as the baseline problem statement demonstrating the necessity of the quality assurance procedures implemented between Milestone 1 and Milestone 2.
- Roboflow Version 3 is preserved independently under [`datasets/Roboflow_M1_M2_V3/`](../../datasets/Roboflow_M1_M2_V3/) as the concrete realization of those quality improvements.

---

## 4. Associated Documentation & Previews

- **Roboflow Project Summary:** [`evidence/M1/04_dataset_v1/roboflow_project.md`](./04_dataset_v1/roboflow_project.md)
- **Version 3 Dataset Previews:** [`evidence/M1/04_dataset_v1/roboflow_v3_previews/`](./04_dataset_v1/roboflow_v3_previews/)
- **Roboflow Dataset README:** [`datasets/Roboflow_M1_M2_V3/README.md`](../../datasets/Roboflow_M1_M2_V3/README.md)
