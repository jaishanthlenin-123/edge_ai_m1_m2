# Milestone 2 (M2) — ONNX Model Export & Verification Report

**Project:** Raspberry Pi Camera Enclosure Detection – Edge AI & Embedded Vision  
**Target Hardware:** Raspberry Pi 5 Model B Rev 1.1 (Broadcom BCM2712 Cortex-A76 @ 2.4 GHz)  
**Task Requirement:** ONNX Model Export (3 Marks)  
**Date:** November 6, 2026  
**Status:** Successfully Completed & Verified  

---

## 1. Executive Summary

The original fine-tuned YOLO26 Nano PyTorch model checkpoint (`models/best.pt`) was exported to an optimized ONNX (Open Neural Network Exchange) computational graph at `models/onnx/best.onnx`.

The export preserved the original PyTorch weights intact without retraining or architectural modification, maintaining the exact $640 \times 640$ input resolution and decoupled detection head. Comprehensive graph integrity verification (`onnx.checker`), input/output signature validation, and numerical equivalence comparisons against the PyTorch baseline were executed directly on the Raspberry Pi 5.

---

## 2. Source Model vs Exported ONNX Specifications

| Specification | Source Model (PyTorch) | Exported Model (ONNX) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model Path** | `models/best.pt` | `models/onnx/best.onnx` | Preserved in dedicated directory |
| **Model Architecture** | YOLO26 Nano (`yolo26n`) | YOLO26 Nano Graph | 120 layers fused |
| **Total Parameters** | 2,375,031 (2.38M) | 2,375,031 (2.38M) | Identical parameter weight count |
| **Computational Complexity** | 5.3 GFLOPs | 5.3 GFLOPs | Identical FLOP count |
| **File Format** | PyTorch Checkpoint (Zip) | ONNX Binary Graph | Format compliant |
| **File Size** | **5,419,973 bytes** (5.17 MB) | **9,747,454 bytes** (9.30 MB) | Graph uncompressed for direct engine load |
| **SHA256 (Source)** | `43c1abddc72896043ea52003c7a238c3b924dceb384673db0475e275d8620530` | — | Original source hash preserved |
| **ONNX Opset** | — | **Opset 17** | Broadly supported by ARM64 runtimes |
| **Export Runtime** | — | **2.42 seconds** on RPi 5 CPU | Completed without warnings |

---

## 3. Tensor Signatures & Shapes

The graph inputs and outputs match the exact M1 inference specifications:

### Input Tensor
- **Node Name:** `images`
- **Data Type:** `tensor(float)` (Float32)
- **Dimensions:** `[1, 3, 640, 640]`
  - Batch size: `1`
  - Color channels: `3` (RGB normalized $[0.0, 1.0]$)
  - Height: `640`
  - Width: `640`

### Output Tensor
- **Node Name:** `output0`
- **Data Type:** `tensor(float)` (Float32)
- **Dimensions:** `[1, 5, 8400]`
  - `8400` candidate anchor-free spatial prediction points across $80 \times 80$, $40 \times 40$, and $20 \times 20$ feature pyramid levels.
  - Channel dimension `5` corresponds to:
    - Index 0: Center X coordinate ($c_x$)
    - Index 1: Center Y coordinate ($c_y$)
    - Index 2: Bounding box width ($w$)
    - Index 3: Bounding box height ($h$)
    - Index 4: Class logit / confidence for `RASPBERY PI CAM ENCLOSURE` (Class ID `0`)

---

## 4. Graph Integrity & Structural Verification

Structural validation was executed using the official `onnx` diagnostic suite:

```python
import onnx
onnx_model = onnx.load("models/onnx/best.onnx")
onnx.checker.check_model(onnx_model)
```

- **Structural Validity:** **PASSED** (zero schema errors, all nodes topologically sorted and acyclic).
- **Execution Provider Check:** Loaded into `onnxruntime.InferenceSession` with `CPUExecutionProvider` without graph conversion errors.

---

## 5. Numerical Equivalence: PyTorch vs ONNX Runtime

To guarantee that the ONNX conversion introduced zero unintended model drift or precision degradation, an identical synthetic input tensor $X \sim \mathcal{N}(0, 1) \in \mathbb{R}^{1 \times 3 \times 640 \times 640}$ was evaluated through both backends:

$$\Delta_{\text{abs}} = | Y_{\text{PyTorch}} - Y_{\text{ONNX}} |$$

- **Maximum Absolute Difference:** **$6.1035 \times 10^{-4}$**
- **Mean Absolute Difference:** **$1.3673 \times 10^{-5}$**
- **Equivalence Status:** **PASSED** (well within standard single-precision IEEE 754 floating-point tolerance of $1 \times 10^{-3}$).

---

## 6. Export Artifacts & Evidence Locations

- **Exported Model Weights:** [`models/onnx/best.onnx`](file:///home/forge/Downloads/edge12/edge_ai_camera_enclosure/models/onnx/best.onnx)
- **Original Model Weights (Untouched):** [`models/best.pt`](file:///home/forge/Downloads/edge12/edge_ai_camera_enclosure/models/best.pt)
- **Export Raw Execution Log:** [`evidence/M2/export_logs/onnx_export_log.txt`](file:///home/forge/Downloads/edge12/edge_ai_camera_enclosure/evidence/M2/export_logs/onnx_export_log.txt)
- **Structured JSON Manifest:** [`evidence/M2/export_logs/onnx_export_log.json`](file:///home/forge/Downloads/edge12/edge_ai_camera_enclosure/evidence/M2/export_logs/onnx_export_log.json)
