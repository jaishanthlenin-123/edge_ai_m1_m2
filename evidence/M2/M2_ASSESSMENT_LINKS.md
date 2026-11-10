# Milestone 2 (M2) — Assessment Submission Links & Sheet Mapping

This reference document provides the direct, verified GitHub URLs corresponding to each of the six Milestone 2 assessment criteria.

> [!IMPORTANT]
> **Instructions for Submission:**
> Copy the corresponding direct GitHub URL from the table below and paste it into the designated cell of your official Google Assessment Sheet.

---

## 1. Google Assessment Sheet Mapping Matrix

| Criterion Name | Marks | Target Sheet Cell | Exact GitHub URL to Copy & Paste | Key Verification Evidence |
| :--- | :---: | :---: | :--- | :--- |
| **Export to ONNX** | **3** | **Cell D** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/01_onnx_export` | ONNX Opset 17 export, tensor signatures (`[1,3,640,640]` $\rightarrow$ `[1,5,8400]`), `onnx.checker` verification, numerical parity ($6.10 \times 10^{-4}$ max diff), and execution logs. |
| **Convert and run live** | **4** | **Cell F** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/02_live_onnx_deployment` | Live ONNX Runtime execution on Raspberry Pi 5 CPU, `src/detect_onnx.py`, anchor-free decoding, NMS, HUD overlay, and 8 live camera detection frame captures. |
| **Accuracy check** | **4** | **Cell H** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/03_accuracy_verification` | Held-out test set accuracy parity (100% Precision, 91.7% Recall, 99.31% mAP@50, 0% degradation), confusion matrix, and parity chart. |
| **Fair benchmark** | **4** | **Cell J** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/04_fair_benchmark` | Controlled 45s sustained benchmark on RPi 5 CPU: +80.7% FPS (3.32 $\rightarrow$ 6.00), -44.7% mean latency, -69.5% tail latency, -14.1% CPU load, and latency plots. |
| **CPU vs accelerator stack** | **2** | **Cell L** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/05_cpu_accelerator_comparison` | RPi 5 hardware breakdown (ARM Cortex-A76 vs VideoCore VII GPU vs Hailo-8L NPU) and software runtimes (PyTorch, ONNX Runtime, TFLite, TensorRT, HailoRT). |
| **Own target and trade-off** | **3** | **Cell N** | `https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/06_optimization_tradeoff` | Formal trade-off decision: +80.7% FPS gain with zero accuracy degradation, FP32 vs INT8 trade-off, 640x640 resolution rationale, and production sign-off. |

---

## 2. Direct File Links for Detailed Review

If the assessor requests direct links to specific markdown report files or weights:

### Criterion 1: Export to ONNX (Cell D)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/01_onnx_export`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/01_onnx_export)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/01_onnx_export/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/01_onnx_export/README.md)
- **Model Checkpoint Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx)

### Criterion 2: Convert and Run Live (Cell F)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/02_live_onnx_deployment`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/02_live_onnx_deployment)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/02_live_onnx_deployment/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/02_live_onnx_deployment/README.md)
- **Live Script Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/src/detect_onnx.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/src/detect_onnx.py)

### Criterion 3: Accuracy Check (Cell H)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/03_accuracy_verification`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/03_accuracy_verification)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/03_accuracy_verification/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/03_accuracy_verification/README.md)
- **Accuracy CSV Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/03_accuracy_verification/m2_accuracy_comparison.csv`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/03_accuracy_verification/m2_accuracy_comparison.csv)

### Criterion 4: Fair Benchmark (Cell J)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/04_fair_benchmark`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/04_fair_benchmark)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/04_fair_benchmark/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/04_fair_benchmark/README.md)
- **Benchmark CSV Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/04_fair_benchmark/m2_performance_comparison.csv`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/04_fair_benchmark/m2_performance_comparison.csv)

### Criterion 5: CPU vs Accelerator Stack (Cell L)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/05_cpu_accelerator_comparison`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/05_cpu_accelerator_comparison)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/05_cpu_accelerator_comparison/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/05_cpu_accelerator_comparison/README.md)
- **Full Architecture Report:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/reports/M2_CPU_Accelerator_Comparison.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/reports/M2_CPU_Accelerator_Comparison.md)

### Criterion 6: Own Target and Trade-Off (Cell N)
- **Folder Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/06_optimization_tradeoff`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/06_optimization_tradeoff)
- **Report Link:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/06_optimization_tradeoff/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/06_optimization_tradeoff/README.md)
- **Full Optimization Report:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/reports/M2_Final_Optimization_Report.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/reports/M2_Final_Optimization_Report.md)

---

## 3. Master Index Link
- **Milestone 2 Master Evidence Index:** [`https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/README.md`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M2/README.md)
- **Photos & Videos Evidence Folder (Google Drive):** [Open M1/M2 Photos & Videos Evidence Folder](https://drive.google.com/drive/folders/1J-PwAAhBoObjruYKZ6qn_V1JmQRSrPKM?usp=sharing)
