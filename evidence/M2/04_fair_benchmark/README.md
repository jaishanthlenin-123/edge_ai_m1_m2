# Criterion 4: Fair Benchmark Methodology & Results (4 Marks)

**Assessment Target:** M2 — Model Optimization & Evaluation
**Assessment Cell (Google Sheet):** **Cell J**
**Evidence Directory:** [`evidence/M2/04_fair_benchmark/`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/04_fair_benchmark/)
**Benchmark Suite Script:** [`benchmark/m2_performance_benchmark.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/benchmark/m2_performance_benchmark.py)
**Detailed Benchmark Report:** [`../reports/M2_Performance_Benchmark.md`](../reports/M2_Performance_Benchmark.md)

---

## 1. Controlled Experimental Methodology

To satisfy the **Fair Benchmark** requirement, both runtimes were evaluated under strictly identical, reproducible operating conditions directly on the Raspberry Pi 5 hardware. No synthetic advantages, clock overclocking, or differing preprocessing pipelines were applied to either engine.

### Strict Experimental Controls:
- **Device Hardware:** Raspberry Pi 5 Model B Rev 1.1 (BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz, 4.0 GiB LPDDR4X SDRAM).
- **Operating Environment:** Debian GNU/Linux 13 (trixie) aarch64, Linux kernel 6.12.
- **Camera Device & Input Stream:** USB Camera (`/dev/video0`), fixed 640x480 V4L2 stream.
- **Inference Input Resolution:** $640 \times 640$ letterboxed with identical aspect-ratio padding (`(114, 114, 114)`).
- **Detection Parameters:** Confidence threshold $= 0.35$, NMS IoU threshold $= 0.70$.
- **Test Protocol:** Sustained **45-second continuous capture and inference** per backend following a mandatory 5-frame thermal and cache warm-up.
- **Telemetry Collection:** Synchronous hardware monitoring of per-frame latency, end-to-end pipeline time, process memory RSS, aggregate CPU utilization across all 4 cores, SoC temperature via `/sys/class/thermal/`, and hardware throttling via `vcgencmd get_throttled`.

---

## 2. Comparative Benchmark Results Table

| Performance Metric | Baseline: PyTorch CPU (`best.pt`) | Optimized: ONNX Runtime CPU (`best.onnx`) | Delta (\(\Delta\)) | Performance Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Model Inference FPS** | **3.32 FPS** | **6.00 FPS** | **+80.7%** | Significant throughput gain |
| **Pipeline FPS (End-to-End)**| **3.15 FPS** | **3.33 FPS** | **+5.7%** | Smoother video stream |
| **Median Inference Latency** | **243.7 ms** | **170.6 ms** | **-73.1 ms (-30.0%)** | Lower response time |
| **Mean Inference Latency** | **301.1 ms** | **166.6 ms** | **-134.5 ms (-44.7%)**| 44.7% faster average execution |
| **P95 Tail Latency** | **661.8 ms** | **202.1 ms** | **-459.7 ms (-69.5%)**| Near-complete tail jitter elimination |
| **Minimum Latency** | **163.2 ms** | **129.9 ms** | **-33.3 ms (-20.4%)** | Best-case execution floor |
| **Maximum Latency Spike** | **2335.7 ms** | **337.5 ms** | **-1998.2 ms (-85.5%)**| Eliminates multi-second pipeline stalls |
| **Process RAM Footprint (RSS)**| **615.0 MB** | **602.4 MB** | **-12.6 MB (-2.1%)** | Stable memory consumption |
| **Average CPU Load** | **80.3%** | **66.2%** | **-14.1% lower load** | Frees 34% headroom for system tasks |
| **SoC Temperature Delta** | **+12.1°C** | **+9.35°C** | **-2.75°C cooler** | More efficient thermal profile |
| **Thermal Throttling Register**| **0x0 (None)** | **0x0 (None)** | **Identical** | Zero frequency throttling |

---

## 3. Visual Performance Artifacts

### 3.1 Per-Frame Latency Distribution (45-Second Sustained Run)
The plot below illustrates the per-frame latency trajectory throughout the sustained benchmark run. While PyTorch suffers from frequent multi-second garbage collection and JIT hitches (spiking to 2335 ms), ONNX Runtime maintains a tightly bounded latency distribution centered at 166.6 ms:

![Latency Distribution Profile](m2_benchmark_latency_distribution.png)

### 3.2 Throughput & CPU Load Comparison
The bar charts below contrast the throughput speedup (+80.7%) alongside the aggregate CPU load reduction (-14.1%):

![Throughput and CPU Load Profile](m2_benchmark_throughput_cpu_profile.png)

---

## 4. Architectural Analysis: Why ONNX Runtime Outperforms PyTorch

1. **Static Graph Optimization:**
   PyTorch in eager mode dynamically re-creates execution graphs per forward pass, introducing significant Python interpreter overhead and dynamic memory allocations. ONNX Runtime loads a pre-fused, static computation graph, eliminating runtime graph tracing.
2. **ARM NEON SIMD & MLAS Acceleration:**
   ONNX Runtime’s `CPUExecutionProvider` utilizes Microsoft MLAS (Machine Learning Acceleration Subsystem) assembly kernels specifically tuned for ARMv8.2-A architectures. It vectorizes $3 \times 3$ and $1 \times 1$ convolutions across 128-bit NEON registers, achieving 1.8x higher computational efficiency than PyTorch’s generic OpenMP CPU backend.
3. **Operator Fusion:**
   Adjacent operations such as `Conv2d + BatchNorm2d + SiLU` are fused into monolithic execution kernels, drastically reducing DDR memory read/write cycles between layers.

---

## 5. Raw Benchmark Data Files

All raw telemetry samples and timing CSVs are preserved in this directory for independent verification:
- **Comprehensive Multi-Runtime Comparison CSV:** [`m2_performance_comparison.csv`](m2_performance_comparison.csv)
- **PyTorch Baseline Raw Per-Frame Telemetry (142 frames):** [`pytorch_benchmark.csv`](pytorch_benchmark.csv)
- **ONNX Runtime Raw Per-Frame Telemetry (150 frames):** [`onnx_benchmark.csv`](onnx_benchmark.csv)
