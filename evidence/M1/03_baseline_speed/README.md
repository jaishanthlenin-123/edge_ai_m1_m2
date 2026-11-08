# M1 Assessment Criterion 3: Baseline Speed Benchmark

**Assessment Item:** 03_baseline_speed  
**Status:** Complete & Verified  

---

## Overview

This module contains the baseline throughput and latency measurements recorded during Milestone 1 prior to ONNX Runtime optimization and acceleration.

## Key Evidence Files

- **Comprehensive Performance Report:** [`evidence/M1/M1_Baseline_Performance_Report.md`](../M1_Baseline_Performance_Report.md)
- **Raw Latency Measurements (CSV):** [`evidence/M1/csv/baseline_m1_measurements.csv`](../csv/baseline_m1_measurements.csv)
- **Benchmark Summary (TXT):** [`evidence/M1/benchmark_logs/m1_baseline_summary.txt`](../benchmark_logs/m1_baseline_summary.txt)
- **Benchmark Summary (JSON):** [`evidence/M1/benchmark_logs/m1_baseline_summary.json`](../benchmark_logs/m1_baseline_summary.json)
- **Benchmark Execution Script:** [`benchmark/baseline_benchmark.py`](../../../benchmark/baseline_benchmark.py)

## Baseline Metrics Summary

| Metric | PyTorch Baseline (Unaccelerated CPU) | Target for Milestone 2 Optimization |
| :--- | :--- | :--- |
| **Inference Latency** | **8,880.5 ms** (~8.88 s) | $< 100\text{ ms}$ (ONNX / OpenVINO / CPU) |
| **Frame Rate (FPS)** | **0.11 FPS** | $> 10\text{ FPS}$ on Edge CPU |
| **Pre-process Latency** | $2.4\text{ ms}$ | $< 3.0\text{ ms}$ |
| **Post-process Latency**| $1.8\text{ ms}$ | $< 2.0\text{ ms}$ |
| **Bottleneck Identification**| PyTorch FP32 CPU graph traversal without vectorization | Resolved in M2 via ONNX graph quantization |

