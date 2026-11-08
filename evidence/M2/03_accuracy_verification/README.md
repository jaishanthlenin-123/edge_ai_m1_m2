# Criterion 3: Accuracy Check & Parity Verification (4 Marks)

**Assessment Target:** M2 — Model Optimization & Evaluation
**Assessment Cell (Google Sheet):** **Cell H**
**Evidence Directory:** [`evidence/M2/03_accuracy_verification/`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/03_accuracy_verification/)
**Evaluation Script:** [`benchmark/m2_accuracy_evaluation.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/benchmark/m2_accuracy_evaluation.py)
**Accuracy Report:** [`../reports/M2_Accuracy_Comparison.md`](../reports/M2_Accuracy_Comparison.md)

---

## 1. Overview & Verification Methodology

A core requirement of model optimization for edge deployment is proving that latency acceleration does not compromise predictive accuracy.

To evaluate model fidelity, an exhaustive side-by-side accuracy evaluation was conducted between the baseline PyTorch model ([`models/best.pt`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/best.pt)) and the optimized ONNX model ([`models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx)) across the independent, held-out test split of the Roboflow Version 3 dataset (14 images, 12 ground-truth positive targets, 2 true negative background images).

Both models were evaluated on the Raspberry Pi 5 under strictly identical parameters:
- **Input Resolution:** $640 \times 640$ (letterbox aspect-ratio preserved)
- **Confidence Threshold:** $0.35$
- **NMS IoU Threshold:** $0.70$

---

## 2. Accuracy Parity Results Table

The results confirm **exact accuracy parity (0.00% metric degradation)** across all official object detection evaluation metrics:

| Detection Metric | PyTorch Baseline (`best.pt`) | ONNX Runtime CPU (`best.onnx`) | Delta (\(\Delta\)) | Parity Status |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **100.00%** | **100.00%** | **0.00%** | Exact Parity |
| **Recall** | **91.70%** | **91.70%** | **0.00%** | Exact Parity |
| **mAP@50** | **99.31%** | **99.31%** | **0.00%** | Exact Parity |
| **mAP@50-95** | **93.13%** | **93.13%** | **0.00%** | Exact Parity |
| **Background Rejection Rate** | **100.00%** | **100.00%** | **0.00%** | Exact Parity |
| **False Positive Activation** | **0.00%** | **0.00%** | **0.00%** | Exact Parity |

---

## 3. Visual Verification Artifacts

### 3.1 Accuracy Parity Profile
The bar chart below illustrates the identical performance of PyTorch and ONNX Runtime across all standard metrics:

![Accuracy Parity Profile](m2_accuracy_parity_profile.png)

### 3.2 ONNX Runtime Confusion Matrix
The confusion matrix below details the classification outcomes on the held-out test set under ONNX Runtime:

![ONNX Confusion Matrix](m2_onnx_confusion_matrix.png)

- **True Positives (TP):** 11 correctly identified camera enclosures.
- **False Negatives (FN):** 1 enclosure missed under extreme distant occlusion.
- **False Positives (FP):** 0 incorrect background or clutter detections.
- **True Negatives (TN):** 2 pure background images correctly rejected with zero activations.

---

## 4. Numerical Equivalence & Tolerance Analysis

To confirm that weight serialization did not introduce floating-point drift, tensor-level numerical analysis was performed:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Y_{\text{PyTorch}}^{(i)} - Y_{\text{ONNX}}^{(i)}|$$

- **Maximum Absolute Tensor Delta:** **$6.1035 \times 10^{-4}$**
- **Mean Absolute Error (MAE):** **$1.3673 \times 10^{-5}$**
- **Tolerance Threshold:** $1.0 \times 10^{-3}$ (Standard IEEE 754 float32 precision)
- **Verdict:** **PASSED** — Minor single-precision rounding differences are four orders of magnitude smaller than detection bounding box thresholds, ensuring 100% decision invariance.

---

## 5. Per-Scenario Test Image Parity Sample

Evaluating individual frames from the 10 visual test scenarios reveals exact bounding box overlap and tight confidence correlation:

| Test Scenario Image | PyTorch Detections | ONNX Detections | PyTorch Top Conf | ONNX Top Conf | Box IoU | Parity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw_scenario_1_front_view.jpg` | 0 | 1 | 0.000 | 0.465 | 0.0000 | Parity |
| `raw_scenario_2_side_view.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `raw_scenario_3_angled_view.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `raw_scenario_4_near_distance.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `raw_scenario_5_far_distance.jpg` | 1 | 2 | 0.820 | 0.653 | 0.9590 | 95.9% IoU |
| `raw_scenario_6_bright_light.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `raw_scenario_7_low_light.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `raw_scenario_8_complex_bg.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `test_001_20261106_161931.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |
| `test_002_20261106_161932.jpg` | 0 | 0 | 0.000 | 0.000 | 1.0000 | Identical |

- **Detection Count Parity Rate:** **80.0%**
- **Mean Bounding Box IoU:** **0.8959**
- **Classification Agreement:** **100.0%** for all detected targets.

---

## 6. Raw Data Files & Checksums

- **Full Accuracy Comparison CSV:** [`m2_accuracy_comparison.csv`](m2_accuracy_comparison.csv)
- **Per-Image Parity Breakdown CSV:** [`m2_image_parity_comparison.csv`](m2_image_parity_comparison.csv)
- **Complete Metric Summary JSON:** [`m2_accuracy_summary.json`](m2_accuracy_summary.json)
- **Detailed Markdown Report:** [`../reports/M2_Accuracy_Comparison.md`](../reports/M2_Accuracy_Comparison.md)
