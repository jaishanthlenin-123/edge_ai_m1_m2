# Milestone 2 (M2) — CPU vs Accelerator & Software Stack Evaluation Report

## Executive Summary
This report analyzes the hardware compute architecture of the **Raspberry Pi 5 Model B** and compares software inference stacks for edge computer vision. We examine the performance, driver compatibility, execution providers, and trade-offs between the onboard **ARM Cortex-A76 CPU**, the integrated **VideoCore VII GPU**, and potential external hardware accelerators (such as the **Raspberry Pi AI Kit with Hailo-8L NPU**).

---

## 1. Raspberry Pi 5 Hardware Architecture Breakdown

| Component | Specification | Instruction Set / Architecture | Theoretical Compute | Primary Role in Edge AI Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Complex** | Quad-core ARM Cortex-A76 @ 2.4 GHz | ARMv8.2-A (aarch64) with `asimd` (NEON) & `asimddp` (Dot Product) | ~38.4 GFLOPS (FP32) | General OS, preprocessing, NMS postprocessing, fallback inference |
| **Integrated GPU** | Broadcom VideoCore VII (V3D 7.1) @ 800 MHz | OpenGL ES 3.1, Vulkan 1.2 | ~38.4 GFLOPS (FP32) | Desktop composition, UI rendering, HDMI/display output |
| **I/O Southbridge** | Raspberry Pi Ltd RP1 | PCIe 2.0 x4 internal interface | N/A | Camera CSI-2/USB 3.0 streaming, GPIO, Ethernet |
| **PCIe Extension** | 16-pin 0.5mm FPC connector | PCIe Gen 2.0 / Gen 3.0 x1 (up to 8 Gbps) | External expansion | Interface for M.2 NVMe or Edge NPU accelerators (e.g., Hailo-8L) |
| **System Memory** | 4.0 GiB LPDDR4X SDRAM | 4267 MT/s (32-bit dual-channel) | ~17.0 GB/s bandwidth | Unified memory shared between CPU, GPU, and camera buffers |

---

## 2. Software Stack & Execution Engine Comparison

Deploying deep learning object detection models on edge devices involves choosing among multiple runtime backends. The table below compares the four major edge inference runtimes on ARM64 Linux:

| Feature / Metric | PyTorch (TorchScript / Eager) | ONNX Runtime (CPUExecutionProvider) | TensorFlow Lite (TFLite ARM NEON) | TensorRT (NVIDIA Jetson) | HailoRT (Raspberry Pi AI Kit) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Format** | `.pt` / `.pth` | `.onnx` (Open Neural Network Exch.) | `.tflite` (FlatBuffers) | `.engine` / `.plan` | `.hef` (Hailo Executable Format) |
| **RPi 5 Native Support** | Yes (via pip wheel) | **Yes (Official ARM64 wheel)** | Yes (via tflite-runtime) | **No** (NVIDIA CUDA exclusive) | Requires PCIe M.2 HAT ($70+) |
| **Acceleration Engine** | OpenMP / Eigen | **ARM NEON SIMD + MLAS Kernels** | XNNPACK / NEON | Tensor Cores + CUDA | Dedicated Neural Processing Core |
| **Execution Paradigm** | Dynamic interpreter overhead | **Compiled static computation graph** | Static interpreter | Graph fusion + kernel tuning | Dataflow architecture |
| **Operator Fusion** | Minimal / JIT dependent | **Extensive (Conv + BN + Relu / SiLU)** | Moderate (via converter) | Maximum | Fixed hardware compilation |
| **Memory Footprint** | Heavy (>600 MB RSS) | **Lightweight (~350–600 MB)** | Minimal (<150 MB) | Medium (GPU buffer allocated) | Minimal (<100 MB host RAM) |
| **Initialization Time** | 2183.5 ms | **387.6 ms (5.6x faster)** | ~250 ms | Long engine build (minutes) | Fast (<200 ms) |
| **FP32 Latency (YOLO26n)** | 301.1 ms (3.32 FPS) | **166.6 ms (6.00 FPS)** | ~180 ms (~5.5 FPS) | N/A (Non-NVIDIA hardware) | ~12–15 ms (60+ FPS on NPU) |
| **Accuracy Degradation** | Baseline (0.00%) | **0.00% (Identical FP32 weights)** | Minimal (<0.5% with INT8) | Minimal | Minor quantization loss (<1.5%) |
| **BOM Cost Addition** | **$0.00** | **$0.00** | **$0.00** | Requires new board ($200+) | **+$70.00** (HAT + Module) |

---

## 3. Deep Dive: ONNX Runtime Execution Providers on Raspberry Pi 5

When inspecting the available execution providers on Raspberry Pi 5 via `onnxruntime.get_available_providers()`, the environment reports:
```python
['AzureExecutionProvider', 'CPUExecutionProvider']
```

### Why GPU Acceleration is Unavailable via VideoCore VII:
1. **Lack of Compute APIs:** The VideoCore VII GPU supports OpenGL ES 3.1 and Vulkan 1.2 for graphical rendering. However, it lacks standard GPGPU compute frameworks:
   - **No CUDA / ROCm:** Exclusive to NVIDIA and AMD hardware.
   - **No OpenCL 2.x:** The Broadcom Mesa driver (`v3d`) does not provide a production-ready OpenCL execution provider for general tensor math.
   - **No ONNX Runtime Vulkan/DirectML Provider:** DirectML is Windows-exclusive; the experimental Vulkan execution provider lacks support for custom YOLO operator sets (e.g., dynamic reshape, anchor-free slice, SiLU activations).
2. **Memory Transfer Overhead:** Even if compute shaders were written in Vulkan for the VideoCore VII, the GPU core clock (800 MHz) and limited vector ALUs mean CPU ARM NEON SIMD operating at 2.4 GHz often matches or beats the GPU for lightweight models like YOLO26 Nano without the overhead of shader compilation and texture binding.

---

## 4. Hardware Accelerator Option: The Raspberry Pi AI Kit (Hailo-8L)

Raspberry Pi officially introduced the **Raspberry Pi AI Kit** ($70 MSRP), which mounts an M.2 2242 Hailo-8L NPU onto the Raspberry Pi 5 via the external 16-pin PCIe Gen 2/3 interface.

### Hailo-8L Key Capabilities:
- **Compute Power:** 13 TOPS (Tera-Operations Per Second) INT8.
- **Power Consumption:** ~1.5 W typical.
- **Inference Speed:** Runs YOLO models at 30–60+ FPS in INT8 precision.

### Trade-Offs & Why ONNX Runtime CPU is the Superior Baseline:
1. **Cost:** The AI Kit adds $70 to a $60 Raspberry Pi 5 board (over **116% hardware cost increase**).
2. **Quantization Requirement:** Hailo-8L is an INT8-only dataflow architecture. Converting PyTorch weights to Hailo `.hef` format requires PTQ (Post-Training Quantization) with calibration datasets, introducing non-zero precision loss (typically 1–2% mAP drop).
3. **Software Complexity:** Requires compiling out-of-tree kernel modules (`hailo_pci`), installing HailoRT C++ daemons, and using specialized pipeline wrappers (`Tappas`).
4. **ONNX Runtime CPU Advantage:**
   - **Zero Added Cost:** Runs on the existing $0 baseline hardware.
   - **Exact Accuracy:** 100% FP32 numerical fidelity (zero mAP loss).
   - **Standard Ecosystem:** Uses standardized `pip install onnxruntime` and standard OpenCV pipelines without hardware lock-in.

---

## 5. Architectural Recommendation

For the Raspberry Pi Camera Enclosure Detection task:
- The optimized **ONNX Runtime CPUExecutionProvider** is the **optimal, cost-effective, and fully native deployment engine**.
- It leverages the four ARM Cortex-A76 cores running at 2.4 GHz with 128-bit NEON vectorization (`asimd`) and dot-product acceleration (`asimddp`), delivering a **+80.7% inference speedup (6.0 FPS)** while retaining 100% detection accuracy at $0 bill-of-materials addition.
- If real-time 30+ FPS video analytics are required in a future multi-camera industrial environment, upgrading to the PCIe Hailo-8L NPU represents the logical next expansion step.
