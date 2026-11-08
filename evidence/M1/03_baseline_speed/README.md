# M1 Assessment Criterion 3: Baseline Speed Benchmark

**Assessment Item:** 03_baseline_speed
**Mark Weighting:** 3 Marks
**Status:** Complete & Verified

---

## 1. Overview

This module documents the unaccelerated baseline inference speed, pipeline latency, and system hardware utilization of the PyTorch YOLO26 Nano model (`models/best.pt`) executing directly on the Raspberry Pi 5 CPU prior to runtime graph optimization.

For the comprehensive engineering performance evaluation, refer to:
* **Comprehensive Performance Report:** [`evidence/M1/M1_Baseline_Performance_Report.md`](../M1_Baseline_Performance_Report.md)

---

## 2. Visual Performance Evidence & Telemetry Screenshots

### 2.1 Live Baseline Inference Screenshots (HUD Telemetry)
The unaccelerated PyTorch model exhibits significant frame-to-frame latency due to unoptimized FP32 graph traversal on CPU:

| Screenshot | Description & Observed Latency | HUD Telemetry Details |
| :---: | :--- | :--- |
| ![Baseline 8880ms](./benchmark_visuals/baseline_inference_latency_8880ms.jpg) | **Single-Thread Peak Latency:** Unaccelerated PyTorch execution ([Inspect](./benchmark_visuals/baseline_inference_latency_8880ms.jpg)) | **Latency:** 8,880.5 ms<br>**Frame Rate:** 0.1 FPS<br>**Detection:** 86.9% conf |
| ![Baseline 13908ms](./benchmark_visuals/baseline_inference_latency_13908ms.jpg) | **Cold-Start Worst-Case Latency:** Initial thread execution spike ([Inspect](./benchmark_visuals/baseline_inference_latency_13908ms.jpg)) | **Latency:** 13,908.3 ms<br>**Frame Rate:** 0.1 FPS<br>**Detection:** 53.5% conf |

---

### 2.2 Empirical Benchmark Plots (Derived Directly from 262-Frame CSV Logs)

The following charts were programmatically plotted from the genuine 262-frame timing and telemetry logs in [`evidence/M1/csv/baseline_m1_measurements.csv`](../csv/baseline_m1_measurements.csv):

#### Frame-by-Frame Pipeline Latency Profile
![Baseline Latency Profile](./benchmark_visuals/m1_baseline_latency_profile.png)
*Figure 3.1: Frame-by-frame latency breakdown showing total pipeline latency vs. raw model inference latency across 262 consecutive evaluation frames on Raspberry Pi 5.*

#### System Hardware Telemetry (CPU & SoC Temperature)
![System Telemetry Profile](./benchmark_visuals/m1_baseline_system_telemetry.png)
*Figure 3.2: Raspberry Pi 5 hardware telemetry recorded during the 5-minute benchmark run: Total CPU load averaged 55.7% with SoC temperatures rising +12.1°C without triggering thermal throttling (0x0).*

#### Baseline vs. Milestone 2 Optimization Comparison
![M1 vs M2 Speedup Comparison](./benchmark_visuals/m1_baseline_vs_m2_onnx_speedup.png)
*Figure 3.3: Direct benchmark comparison between Milestone 1 unaccelerated PyTorch CPU and Milestone 2 accelerated ONNX Runtime CPU, confirming an 80.7% throughput improvement.*

---

## 3. Key Baseline Performance Metrics

| Metric | PyTorch Baseline (M1 Unaccelerated) | Target / Milestone 2 Result | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Model Latency** | **301.1 ms** (Median: 243.7 ms) | **166.6 ms** (Median: 170.6 ms) | **-44.7% latency** |
| **Cold Start / Peak Latency**| **8,880.5 ms – 13,908.3 ms** | **337.5 ms** | **-97.6% peak latency** |
| **Inference FPS** | **3.32 FPS** | **6.00 FPS** | **+80.7% throughput** |
| **Average CPU Load** | **80.3%** | **66.2%** | **-14.2% CPU reduction** |
| **Thermal Delta** | **+12.1°C** (Peak 60.1°C) | **+9.35°C** | **-2.8°C thermal drop** |

---

## 4. Key Evidence Files

- **Visuals Directory:** [`benchmark_visuals/`](./benchmark_visuals/)
- **Comprehensive Baseline Report:** [`evidence/M1/M1_Baseline_Performance_Report.md`](../M1_Baseline_Performance_Report.md)
- **Raw Latency Measurements (262 frames CSV):** [`evidence/M1/csv/baseline_m1_measurements.csv`](../csv/baseline_m1_measurements.csv)
- **Summary TXT Log:** [`evidence/M1/benchmark_logs/m1_baseline_summary.txt`](../benchmark_logs/m1_baseline_summary.txt)
- **Summary JSON Log:** [`evidence/M1/benchmark_logs/m1_baseline_summary.json`](../benchmark_logs/m1_baseline_summary.json)
- **Benchmark Script:** [`benchmark/baseline_benchmark.py`](../../../benchmark/baseline_benchmark.py)
