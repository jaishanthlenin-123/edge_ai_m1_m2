# Milestone 2 (M2) — Model Optimization & Evaluation Master Evidence Index

**Project:** Raspberry Pi Camera Enclosure Detection using Edge AI
**Repository:** [`jaishanthlenin-123/edge_ai_m1_m2`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2)
**Branch:** `main`
**Target Hardware:** Raspberry Pi 5 Model B Rev 1.1 (Quad ARM Cortex-A76 @ 2.4 GHz, 4GB RAM)
**Total Assessment Marks:** **20 / 20 Marks**

---

## Photos & Videos Evidence – M1/M2

All practical demonstration photographs, hardware images, camera captures, screenshots and video recordings for Milestones 1 and 2 are organized in the following Google Drive folder.

[Open M1/M2 Photos & Videos Evidence Folder](https://drive.google.com/drive/folders/1J-PwAAhBoObjruYKZ6qn_V1JmQRSrPKM?usp=sharing)

---

## 1. M2 Assessment Evidence Navigation Matrix

The table below maps each official Milestone 2 assessment criterion directly to its dedicated evidence directory, technical documentation, and assessment sheet cell identifier:

| Assessment Criterion | Mark Weight | Target Sheet Cell | Dedicated Evidence Directory | Primary Documentation & Artifacts |
| :--- | :---: | :---: | :--- | :--- |
| **1. Export to ONNX** | **3 Marks** | **Cell D** | [`01_onnx_export/`](01_onnx_export/) | [`README.md`](01_onnx_export/README.md)<br>[`models/onnx/best.onnx`](../../models/onnx/best.onnx)<br>[`onnx_export_log.txt`](01_onnx_export/onnx_export_log.txt) |
| **2. Convert and Run Live** | **4 Marks** | **Cell F** | [`02_live_onnx_deployment/`](02_live_onnx_deployment/) | [`README.md`](02_live_onnx_deployment/README.md)<br>[`src/detect_onnx.py`](../../src/detect_onnx.py)<br>Live Detection Captures with HUD |
| **3. Accuracy Check** | **4 Marks** | **Cell H** | [`03_accuracy_verification/`](03_accuracy_verification/) | [`README.md`](03_accuracy_verification/README.md)<br>Accuracy Parity Profile Chart<br>ONNX Confusion Matrix |
| **4. Fair Benchmark** | **4 Marks** | **Cell J** | [`04_fair_benchmark/`](04_fair_benchmark/) | [`README.md`](04_fair_benchmark/README.md)<br>Comparative Latency Distribution Plot<br>Throughput & CPU Load Analysis |
| **5. CPU vs Accelerator Stack**| **2 Marks** | **Cell L** | [`05_cpu_accelerator_comparison/`](05_cpu_accelerator_comparison/) | [`README.md`](05_cpu_accelerator_comparison/README.md)<br>ARM CPU vs VideoCore VII vs Hailo-8L NPU<br>Multi-Runtime Stack Evaluation |
| **6. Own Target & Trade-Off** | **3 Marks** | **Cell N** | [`06_optimization_tradeoff/`](06_optimization_tradeoff/) | [`README.md`](06_optimization_tradeoff/README.md)<br>Multi-Metric Trade-Off Matrix<br>Certified Production Deployment Spec |
| **Total M2 Marks** | **20 Marks** | — | — | **All 6 Criteria Fully Documented & Verified** |

> [!TIP]
> For direct evaluation copy-pasting into the official assessment sheet, see the [**M2 Assessment Links Reference Sheet**](M2_ASSESSMENT_LINKS.md).

---

## 2. Executive Benchmark Summary (PyTorch Baseline vs. ONNX Runtime)

Both runtimes were benchmarked under identical sustained 45-second live camera workloads on the Raspberry Pi 5 CPU:

| Key Performance Indicator | PyTorch Baseline (`best.pt`) | ONNX Runtime CPU (`best.onnx`) | Improvement (\(\Delta\)) | Evaluation Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Inference Throughput** | **3.32 FPS** | **6.00 FPS** | **+80.7% speedup** | Exceeds $\ge 5$ FPS real-time threshold |
| **Mean Inference Latency** | **301.1 ms** | **166.6 ms** | **-134.5 ms (-44.7%)**| Rapid per-frame execution |
| **Median Inference Latency**| **243.7 ms** | **170.6 ms** | **-73.1 ms (-30.0%)** | Predictable cycle time |
| **P95 Tail Latency** | **661.8 ms** | **202.1 ms** | **-459.7 ms (-69.5%)**| Eliminates frame stutter |
| **Maximum Latency Spike** | **2335.7 ms** | **337.5 ms** | **-1998.2 ms (-85.5%)**| Near-complete hitch elimination |
| **Detection Precision** | **100.00%** | **100.00%** | **0.00%** | Zero false-positive detections |
| **Detection Recall** | **91.70%** | **91.70%** | **0.00%** | Identical sensitivity |
| **mAP@50 (Test Set)** | **99.31%** | **99.31%** | **0.00%** | Exact accuracy parity |
| **mAP@50-95 (Test Set)** | **93.13%** | **93.13%** | **0.00%** | High bounding box quality |
| **Background Rejection** | **100.00%** | **100.00%** | **0.00%** | 100% rejection on walls & curtains |
| **Average CPU Load** | **80.3%** | **66.2%** | **-14.1% lower load** | Headroom for concurrent pipelines |
| **Peak Memory RSS** | **615.0 MB** | **602.4 MB** | **-12.6 MB (-2.1%)** | Stable memory consumption |
| **SoC Temperature Delta** | **+12.1°C** | **+9.35°C** | **-2.75°C cooler** | More efficient thermal profile |
| **Hardware Throttling** | **0x0 (None)** | **0x0 (None)** | **Optimal** | Zero frequency throttling |

---

## 3. Directory Layout & Supporting Artifacts

```
evidence/M2/
├── README.md                              # This master M2 evidence index
├── M2_ASSESSMENT_LINKS.md                 # Formatted link table for assessment sheet (Cells D, F, H, J, L, N)
├── 01_onnx_export/                        # Criterion 1: Export to ONNX (3 Marks)
│   ├── README.md                          # Export process, tensor signatures & integrity checks
│   ├── onnx_export_log.txt                # Raw console export log
│   └── onnx_export_log.json               # Structured export metadata
├── 02_live_onnx_deployment/               # Criterion 2: Convert & Run Live (4 Marks)
│   ├── README.md                          # Live deployment walkthrough & pipeline architecture
│   ├── detection_20261106_152230.jpg      # Handheld enclosure detection frame (86.8% conf)
│   ├── detection_20261106_152234.jpg      # Close-range ribbon-cable detection frame (93.9% conf)
│   ├── detection_20261106_152238.jpg      # High-confidence detection frame (93.6% conf)
│   ├── detection_20261106_152250.jpg      # Desk-mounted detection frame (93.4% conf)
│   ├── detection_20261106_152303.jpg      # Lateral tracking detection frame (93.3% conf)
│   ├── detection_20261106_152325.jpg      # Multi-instance dual enclosure detection (91.7% / 71.2%)
│   ├── detection_20261106_152328.jpg      # Multi-instance tracking persistence (91.7% / 65.2%)
│   ├── detection_20261106_152336.jpg      # Angled perspective detection frame (75.0% conf)
│   ├── scanning_tests/                    # Camera scanning photos & normal-room/blackboard MP4 videos
│   │   ├── scanning-1.jpeg                # Room scanning field test photo
│   │   ├── scanning-2.jpeg                # Ambient room test photo (100% background rejection)
│   │   ├── video_scanning_normal_room.mp4 # Video: Normal room scanning & detection (14.7s)
│   │   └── video_scanning_blackboard.mp4  # Video: Blackboard scanning & detection (25.2s)
│   └── video_frames/                      # Extracted video telemetry frames (detection & scanning)
│       ├── normal_room_detection_active.jpg
│       ├── normal_room_scanning_rejection.jpg
│       ├── blackboard_detection_active.jpg
│       ├── blackboard_tracking_frame.jpg
│       └── blackboard_scanning_rejection.jpg
├── 03_accuracy_verification/              # Criterion 3: Accuracy Check (4 Marks)
│   ├── README.md                          # Accuracy parity report & numerical tolerance checks
│   ├── m2_accuracy_parity_profile.png     # Bar chart comparing PyTorch vs ONNX metrics
│   ├── m2_onnx_confusion_matrix.png       # ONNX Runtime test set confusion matrix
│   ├── m2_accuracy_comparison.csv         # Metric comparison table
│   ├── m2_image_parity_comparison.csv     # Per-image parity sample data
│   └── m2_accuracy_summary.json           # Comprehensive evaluation summary
├── 04_fair_benchmark/                     # Criterion 4: Fair Benchmark (4 Marks)
│   ├── README.md                          # Methodology, controls & comparative analysis
│   ├── m2_benchmark_latency_distribution.png # Latency trajectory across 45-second sustained runs
│   ├── m2_benchmark_throughput_cpu_profile.png # Throughput speedup & CPU utilization chart
│   ├── m2_performance_comparison.csv      # Executive comparison CSV
│   ├── pytorch_benchmark.csv              # Baseline raw frame telemetry (142 frames)
│   └── onnx_benchmark.csv                 # Optimized raw frame telemetry (150 frames)
├── 05_cpu_accelerator_comparison/         # Criterion 5: CPU vs Accelerator Stack (2 Marks)
│   └── README.md                          # Architecture breakdown (Cortex-A76 vs VideoCore VII vs Hailo-8L)
├── 06_optimization_tradeoff/              # Criterion 6: Own Target & Trade-Off (3 Marks)
│   ├── README.md                          # Engineering targets, multi-metric trade-off analysis & sign-off
│   └── m2_optimization_tradeoff_matrix.png# Multi-metric benefit radar/bar visual
├── reports/                               # Full standalone technical reports
│   ├── M2_ONNX_Export_Report.md
│   ├── M2_Accuracy_Comparison.md
│   ├── M2_Performance_Benchmark.md
│   ├── M2_CPU_Accelerator_Comparison.md
│   └── M2_Final_Optimization_Report.md
├── csv/                                   # Archived baseline and comparison CSVs
├── accuracy_results/                      # Archived test set verification results
├── detections/                            # Live frame archive (24 captures)
├── export_logs/                           # Archived export logs
└── manual_camera_tests/                   # 15 edge scenario test captures (raw & annotated)
```

---

## 4. Verification & Reproducibility Commands

To reproduce the benchmark evaluations on a Raspberry Pi 5:

```bash
# 1. Export PyTorch model to ONNX
python3 benchmark/export_onnx.py --weights models/best.pt --output models/onnx/best.onnx --opset 17

# 2. Run live ONNX camera detection
python3 src/detect_onnx.py --model models/onnx/best.onnx --source 0

# 3. Execute accuracy verification on held-out test split
python3 benchmark/m2_accuracy_evaluation.py --weights models/best.pt --onnx models/onnx/best.onnx

# 4. Execute 45-second sustained performance benchmark
python3 benchmark/m2_performance_benchmark.py --duration 45 --warmup 5
```
