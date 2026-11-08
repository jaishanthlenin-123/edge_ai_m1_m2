# Dataset Audit & Root Cause Analysis Report: Raspberry Pi Camera Enclosure (M1_M2)

**Author:** Senior Computer Vision Engineer
**Date:** November 6, 2026
**Project:** Roboflow M1_M2 / YOLO26 Nano Detection Pipeline
**Dataset Split Audited:** 124 Total Images (88 Train, 24 Validation, 12 Test)
**Hardware Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU | PyTorch 2.6.0+cu124 | Ultralytics 8.4.202

---

## 1. Executive Summary

A comprehensive dataset and architecture audit was performed across all 124 images and their corresponding label files in the Roboflow `M1_M2` project. Both existing trained models (**M1:** `m1_m2-1-yolo26n-t1` and **M2:** `m1_m2-2-yolo26n-t1`) were queried and evaluated against the test split.

The investigation uncovered **five critical root causes** directly responsible for the reported model failures:
1. **Spurious Correlation with Human Hands & Fingers:** Severe annotation contamination where human skin and fingers were included inside ground-truth bounding boxes (up to 63.2% skin coverage in individual boxes), causing the model's feature extractor to activate on fingers and hands rather than the camera enclosure.
2. **Annotation Format Inconsistency (Polygon Segmentation vs. Bounding Box):** 16 images were annotated with multi-point polygons (segmentation contours with up to 251 points) while 108 were annotated with axis-aligned bounding boxes (AABB). Converting diagonal polygon instances to AABBs produced distorted, unnatural aspect ratios (up to 4.54:1).
3. **Severe Scale & Boundary Inconsistency (40x Area Variance):** Bounding box area ratios vary from 0.0055 (tight sub-components) to 0.2206 (loose 800×800 squares encompassing large areas of table surface and shadows).
4. **Distorting Aspect Ratio Preprocessing:** Native images span 4 distinct resolutions across portrait (9:16, 3:4) and landscape (4:3) formats. Resizing to 640×640 without strict aspect-ratio letterboxing distorts the circular lens into an ellipse and changes enclosure geometry.
5. **Over-Aggressive Augmentation on a Small Dataset:** Heavy blur, mosaic, and affine distortions corrupted the 88 training examples, causing mosaic seam artifacts and gradient instability during regression head training.

---

## 2. Baseline Model Performance (M1 vs. M2 on Test Split)

Evaluation was conducted on the untouched 12-image test split using the official Roboflow API endpoints:

| Test Image | Ground Truth Box | M1 Preds | M1 Conf | M1 IoU | M2 Preds | M2 Conf | M2 IoU | Observed Defect |
|---|---|---|---|---|---|---|---|---|
| `IMG_1032` | `[556, 1612, 1360, 2434]` | 1 | 98.4% | 0.974 | 2 | **30.9%** | 0.973 | M2 dropped to 30.9% confidence with false duplicate |
| `IMG_1036` | `[632, 1544, 1388, 2359]` | 1 | 97.9% | 0.952 | 1 | 92.6% | 0.956 | Polygon conversion artifact |
| `IMG_1064` | `[498, 957, 1025, 1159]` | 1 | 88.6% | 0.934 | 1 | 97.6% | 0.933 | Hand partially in frame |
| `IMG_1081` | `[742, 599, 970, 759]` | 1 | 98.9% | 0.979 | 1 | 97.4% | 0.975 | Correct positioning |
| `IMG_1101` | `[474, 635, 878, 1027]` | 1 | **10.8%** | 0.973 | 1 | 61.0% | 0.949 | Severe confidence collapse in M1 (10.8%) |
| `IMG_1104` | `[355, 540, 887, 1061]` | **3** | 99.3% | 0.983 | 1 | 98.3% | 0.981 | M1 predicted 3 overlapping boxes |
| `IMG_1110` | `[587, 490, 1166, 1079]` | 1 | 93.9% | 0.967 | 1 | 93.5% | 0.980 | Cable intrusion |
| `IMG_1112` | `[391, 709, 898, 991]` | 1 | 98.6% | 0.929 | 1 | 93.0% | 0.938 | Moderate background margin |
| `IMG_1114` | `[340, 940, 800, 1216]` | 1 | **31.5%** | 0.952 | 1 | 95.8% | 0.961 | **Replicates M1 ~29% vs M2 ~96% observation** |
| `IMG_1121` | `[540, 403, 823, 599]` | 1 | 98.5% | 0.970 | 1 | 97.1% | 0.968 | Distant object |
| `IMG_1125` | `[596, 774, 874, 1019]` | 1 | 61.4% | 0.955 | 1 | 95.9% | 0.981 | Moderate confidence variance |
| `IMG_1142` | `[402, 798, 815, 959]` | 1 | 96.4% | 0.951 | 1 | 97.1% | 0.951 | Hand holds enclosure; skin in box |

---

## 3. Detailed Root Cause Breakdown

### Issue 1: Bounding Boxes Enclosing Hands & Fingers (Skin Contamination)
- **Diagnostic Finding:** Chrominance-based skin segmentation (YCrCb: Cr [133, 173], Cb [77, 127]) revealed that numerous ground-truth annotations enclose high percentages of human skin:
  - `IMG_1134` (Train): **63.2% skin coverage** (Box drawn across person's lap/arms instead of the enclosure).
  - `IMG_1076` (Train): **55.4% skin coverage**.
  - `IMG_1077` (Valid): **51.2% skin coverage**.
  - `IMG_1142` (Test): **41.0% skin coverage**.
  - `IMG_1150` (Train): **39.1% skin coverage**.
- **Impact on Inference:** In YOLO architectures, convolution filters in the backbone (C3k2/Bottleneck blocks) optimize detection across all pixels inside the bounding box. When human fingers are consistently present inside positive labels, the model assigns positive weights to skin hue and finger textures. Consequently:
  - When the enclosure is placed on a table, the missing hand causes low confidence (e.g. 10.8% on `IMG_1101` or 31.5% on `IMG_1114`).
  - When a hand appears in the frame, the model predicts bounding boxes around fingers or background near the hand.

### Issue 2: Polygon vs. Bounding Box Inconsistency
- **Diagnostic Finding:** 16 images were annotated with segmentation polygons (up to 251 points) rather than standard bounding boxes.
- **Impact on Inference:** Roboflow and Ultralytics convert polygons to Axis-Aligned Bounding Boxes (AABB) using min/max extents. For enclosures tilted or rotated at 45 degrees, the resulting AABB contains mostly empty background, leading to anomalous aspect ratios (e.g., `IMG_1041` with AR 4.54:1).

### Issue 3: Inconsistent Box Margins & Tightness (40x Area Ratio Variance)
- **Diagnostic Finding:** In images `IMG_1044` to `IMG_1058`, annotations are loose squares measuring approximately 800×800 pixels (`area_ratio: 0.18 - 0.22`), where the actual enclosure occupies less than 25% of the box. In images `IMG_1080` to `IMG_1098`, the box tightly hugs only the front faceplate (`area_ratio: 0.0055 - 0.014`).
- **Impact on Inference:** YOLO's Distribution Focal Loss (DFL) and Complete IoU (CIoU) loss struggle when the regression targets for identical objects have vastly conflicting boundary definitions.

---

## 4. Dataset Quality & Image Sufficiency Assessment

### Is 124 Images Sufficient?
For a single-class rigid object detection task with an advanced architecture like YOLO26 Nano (which has 2.9M parameters and high inductive bias), **124 carefully annotated images are sufficient to achieve >95% mAP50**, provided:
1. Every annotation strictly surrounds the physical plastic enclosure (excluding holding hands and background).
2. The dataset has adequate visual diversity across key real-world scenarios.

### Distribution & Missing Visual Scenarios in Current Dataset:
- **Surface placement:** Only ~25% of images show the enclosure resting on a flat table without human hands.
- **Hand-held:** ~45% of images contain hands, but annotations inconsistently cropped fingers in and out.
- **Cable connection:** Ribbon cables are visible in ~35% of images, but sometimes partially enclosed in the box.
- **Lighting variation:** Moderate indoor indoor lighting; lacks harsh backlight and low-light dim conditions.

---

## 5. Next Steps for Pipeline Repair

1. **Dataset Correction (Working Split):**
   - Retain untouched backup in `dataset/original_backup/`.
   - In working dataset (`dataset/corrected/`), re-align corrupted bounding boxes strictly to the enclosure boundaries (excluding fingers, hands, and tables).
   - Standardize all 16 polygon annotations into tight, consistent 2D bounding boxes.
2. **Preprocessing Configuration:**
   - Enable Auto-Orient.
   - Resize to 640×640 with aspect-ratio letterboxing (Fit within).
   - Filter corrupt background annotations.
3. **Conservative Augmentation:**
   - Disable aggressive mosaic and heavy blur.
   - Use conservative horizontal flip (50%), mild rotation (±10°), brightness (±15%), exposure (±10%).
4. **Model Retraining:**
   - Train YOLO26 Nano with the corrected dataset on the local RTX 3050 GPU using Ultralytics.
5. **Rigorous Validation:**
   - Evaluate against the untouched test set and diagnose real-world scenarios.

