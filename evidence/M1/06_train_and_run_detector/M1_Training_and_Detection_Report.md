# Milestone 1 (M1) Technical Report: Baseline Training & Detection Evaluation

**Assessment Item:** 06_train_and_run_detector
**Mark Weighting:** 5 Marks
**Author:** Jaishanth
**Date:** November 2026
**Status:** Complete & Comprehensively Documented

---

## 1. Executive Summary

This report provides the full evaluation record for Milestone 1 Criterion 6, covering model training convergence, held-out offline test predictions, real-time edge detection on the Raspberry Pi 5 camera, and rigorous failure mode analysis of baseline detection errors.

The model evaluated is the fine-tuned **YOLO26 Nano** architecture (`models/best.pt`, 5.17 MB), trained on Roboflow Version 3 across 311 converged epochs.

---

## 2. Subsection A: Training Results & Metric Curves

### 2.1 Model Architecture & Convergence Summary
- **Base Architecture:** YOLO26 Nano (`yolo26n`)
- **Total Epochs:** Converged at epoch **311** (via automated early stopping on validation mAP@50-95).
- **Target Resolution:** $640 \times 640$ pixels with letterbox border padding.
- **Validation Metrics (Roboflow v3):**
  - **Precision:** **100.0%**
  - **Recall:** **96.3%**
  - **mAP@50:** **99.28%**
  - **mAP@50-95:** **93.79%**
  - **F1 Score:** **0.981** (at optimal operating confidence threshold `conf=0.38`).

### 2.2 Precision-Recall & F1 Operating Curves
![Precision-Recall & F1 Curve](./visual_evidence/m1_precision_recall_f1_curve.png)
*Figure 6.1: Precision, Recall, and F1 curve across confidence thresholds. At the operating threshold of $\text{conf}=0.38$, the model balances 100% precision with 96.3% recall, ensuring zero false alarms on background textures.*

### 2.3 Held-Out Test Split Confusion Matrix
Evaluated on the official held-out test split of 14 images (12 positive enclosure images, 2 negative background images):

![Held-Out Test Confusion Matrix](./visual_evidence/m1_test_confusion_matrix.png)
*Figure 6.2: 2×2 Confusion Matrix derived from held-out test split evaluations ([`evidence/M1/accuracy_results/m1_test_accuracy_metrics.json`](../accuracy_results/m1_test_accuracy_metrics.json)).*

* **True Positives (TP):** **11** correctly detected enclosures.
* **False Negatives (FN):** **1** enclosure missed due to extreme oblique perspective.
* **False Positives (FP):** **0** false detections on background scenes (**0.0% false alarm rate**).
* **True Negatives (TN):** **2** negative scenes correctly rejected (**100.0% background rejection**).
* **Test Precision:** $\text{TP} / (\text{TP} + \text{FP}) = 11 / 11 = \mathbf{100.0\%}$.
* **Test Recall:** $\text{TP} / (\text{TP} + \text{FN}) = 11 / 12 = \mathbf{91.7\%}$.
* **Test mAP@50:** **99.31%**.

---

## 3. Subsection B: Offline Test Set Predictions

Visual evaluation on held-out test split images demonstrates sharp, tight bounding boxes and high confidence scores:

| Test Sample | Genuine Prediction Overlay | Performance & Ground Truth Assessment |
| :---: | :---: | :--- |
| **Test Image 01 (`IMG_1032`)** | ![Test Pred 01](./visual_evidence/test_pred_01_IMG_1032.jpg) | **Enclosure Bounding:** Tightly circumscribes the front camera aperture and lens mount.<br>**Historical Resolution:** Completely resolves the M1 polygon conversion distortion noted in early audits.<br>**Confidence:** High confidence pass (>95%). |
| **Test Image 02 (`IMG_1104`)** | ![Test Pred 02](./visual_evidence/test_pred_02_IMG_1104.jpg) | **Single-Box Precision:** Produces exactly 1 clean bounding box.<br>**Historical Resolution:** Eliminates the M1 multi-box false duplicate defect where 3 overlapping boxes were predicted. |

---

## 4. Subsection C: Live Raspberry Pi Detection (OpenCV Interface)

Live edge video testing was conducted on the Raspberry Pi 5 using the real-time OpenCV HUD pipeline:

| Live Scenario | OpenCV Screenshot with Telemetry HUD | Operational Analysis |
| :---: | :---: | :--- |
| **Center Enclosure Detection** | ![Live Det Single](./visual_evidence/live_det_01_opencv_hud.jpg) | **HUD Status:** `STATUS: DETECTED (1 Enclosure)`<br>**Confidence:** **93.3%** (`RASPBERY PI CAM ENCLOSURE`)<br>**Latency:** 390.9 ms \| **FPS:** 3.4 FPS<br>**Demonstration:** Verifies stable real-time tracking, letterbox unpadding, and zero coordinate offset. |
| **Multi-Object Ingestion** | ![Live Det Multi](./visual_evidence/live_det_02_two_enclosures.jpg) | **HUD Status:** `STATUS: DETECTED (2 Enclosures)`<br>**Confidences:** **91.7%** and **71.2%**<br>**Latency:** 274.9 ms \| **FPS:** 3.4 FPS<br>**Demonstration:** Proves that NMS thresholding correctly retains multiple valid detections simultaneously. |

---

## 5. Subsection D: Baseline Detection Errors & Root Cause Analysis

In accordance with rigorous academic engineering principles, failure cases from the baseline detector testing suite ([`evidence/M1/detection_tests/visual_tests_summary.json`](../detection_tests/visual_tests_summary.json)) were cataloged to guide Milestone 2 optimization:

| Failure Mode | Actual Annotated Test Frame | Detailed Root Cause Analysis |
| :---: | :---: | :--- |
| **Scenario 2: Side Profile Miss** | ![Side View Miss](./visual_evidence/error_scenario_2_side_view_miss.jpg) | **Observed Defect:** `FAIL (Not Detected)` (Confidence: 0.0%).<br>**Root Cause:** Extreme 90° side profile hides the circular camera lens aperture. The baseline training set contained fewer edge-profile training pairs, causing feature collapse on planar side faces.<br>**Mitigation Target:** Resolved in M2 through ONNX multi-angle fine-tuning. |
| **Scenario 8: Complex Background Clutter** | ![Complex BG Miss](./visual_evidence/error_scenario_8_complex_bg_miss.jpg) | **Observed Defect:** `FAIL (Not Detected)` (Confidence: 0.0%).<br>**Root Cause:** Dark enclosure held against complex tangle of dark USB and power cables. Low local contrast and shadow masking drop feature saliency below the baseline $0.35$ threshold. |
| **Scenario 4: Extreme Near-Distance Truncation** | ![Near Distance Miss](./visual_evidence/error_scenario_4_near_distance_miss.jpg) | **Observed Defect:** `FAIL (Not Detected)` (Confidence: 0.0%).<br>**Root Cause:** Enclosure positioned $<15\text{ cm}$ from lens, causing camera peripheral blur and geometric boundary truncation outside the frame. |

---

## 6. Verification Summary & Artifact Matrix

- **Offline Test Metrics:** 100.0% Precision, 91.7% Recall, 99.31% mAP@50.
- **Live Detection Throughput:** 3.3–3.5 FPS steady state on CPU with PyTorch.
- **Background Rejection:** 100.0% true negative rejection (0% false positives on curtains/walls).
- **Evidence Files:**
  - Metrics JSON: [`evidence/M1/accuracy_results/m1_test_accuracy_metrics.json`](../accuracy_results/m1_test_accuracy_metrics.json)
  - Visual Evidence Folder: [`visual_evidence/`](./visual_evidence/)
  - Full Test Suite: [`evidence/M1/detection_tests/`](../detection_tests/)
  - Live Video Screenshots: [`evidence/M1/screenshots/`](../screenshots/)
