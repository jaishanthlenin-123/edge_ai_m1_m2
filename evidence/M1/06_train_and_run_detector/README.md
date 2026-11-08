# M1 Assessment Criterion 6: Train & Run Baseline Detector

**Assessment Item:** 06_train_and_run_detector  
**Status:** Complete & Verified  

---

## Overview

This module documents the initial baseline training of the YOLO26 Nano detector and real-time live video inference verification on edge hardware.

## Key Evidence Files

- **Accuracy Metrics (CSV):** [`evidence/M1/accuracy_results/m1_test_accuracy_metrics.csv`](../accuracy_results/m1_test_accuracy_metrics.csv)
- **Accuracy Metrics (JSON):** [`evidence/M1/accuracy_results/m1_test_accuracy_metrics.json`](../accuracy_results/m1_test_accuracy_metrics.json)
- **Visual Detection Test Suite (23 Images):** [`evidence/M1/detection_tests/`](../detection_tests/)
  - Includes raw vs. annotated comparisons across front view, side view, angled view, close distance, far distance, bright lighting, low lighting, and complex background.
- **Live Video Screenshots (19 Frames):** [`evidence/M1/screenshots/`](../screenshots/)
  - Documents live webcam detection frames with HUD telemetry showing detection confidence, bounding box coordinates, and latency.
- **Model Checkpoints:** [`models/best.pt`](../../../models/best.pt) (5.17 MB PyTorch weights).

## Verification Results

- Verified live detection of the Raspberry Pi Camera Enclosure at confidences ranging from 51.0% to 86.9% across diverse lighting and cluttered backgrounds.

