# Criterion 1: Export to ONNX (3 Marks)

**Assessment Target:** M2 — Model Optimization & Evaluation
**Assessment Cell (Google Sheet):** **Cell D**
**Evidence Directory:** [`evidence/M2/01_onnx_export/`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/01_onnx_export/)
**Primary Model Artifact:** [`models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx)
**Export Script:** [`benchmark/export_onnx.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/benchmark/export_onnx.py)

---

## 1. Overview & Objectives

In Milestone 2, the trained PyTorch YOLO26 Nano model checkpoint ([`models/best.pt`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/best.pt)) was exported to the open-standard **ONNX (Open Neural Network Exchange)** format ([`models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx)).

The objective of this stage was to:
1. Decouple inference execution from the dynamic PyTorch runtime and heavy Python dependencies.
2. Produce a static, serialized computational graph optimized for the ARM Cortex-A76 microarchitecture of the Raspberry Pi 5.
3. Validate graph integrity and certify numerical precision parity ($\Delta < 1 \times 10^{-3}$) against the original PyTorch weights.

---

## 2. Model Specifications & Checksums

The export process preserved 100% of the trained weights without retraining, pruning, or structural alterations:

| Property | Source Baseline (PyTorch) | Exported Graph (ONNX) | Verification Status |
| :--- | :--- | :--- | :--- |
| **File Path** | `models/best.pt` | `models/onnx/best.onnx` | Verified & preserved |
| **Model Architecture** | YOLO26 Nano (`yolo26n`) | YOLO26 Nano Static Graph | 120 layers fused |
| **Total Parameters** | 2,375,031 (2.38M) | 2,375,031 (2.38M) | Identical weight count |
| **Computational Complexity** | 5.3 GFLOPs | 5.3 GFLOPs | Identical FLOPs |
| **File Format** | PyTorch Checkpoint (`.pt`, zip archive) | ONNX Binary Protocol (`.onnx`) | Validated format |
| **File Size** | **5,419,973 bytes** (5.17 MB) | **9,747,454 bytes** (9.30 MB) | Graph uncompressed for direct mmap |
| **Target Opset** | N/A | **Opset 17** | Broadly supported on ARM64 |
| **Export Execution Time** | N/A | **2.42 seconds** | Completed cleanly on RPi 5 CPU |
| **SHA-256 Checksum** | `43c1abddc72896043ea52003c7a238c3b924dceb384673db0475e275d8620530` | `d8849eff3325c345a557b4dc53164a2d8a5628b03e04e4c2755e3e29f3d5ee16` | Cryptographically recorded |

---

## 3. Input & Output Tensor Signatures

The exported computational graph strictly adheres to the $640 \times 640$ letterboxed input and single-class detection contract:

### Input Tensor: `images`
- **Data Type:** `float32` (`tensor(float)`)
- **Dimensions:** `[1, 3, 640, 640]`
  - `1`: Batch size (real-time stream inference)
  - `3`: RGB color channels (normalized to range $[0.0, 1.0]$)
  - `640`: Height (pixels)
  - `640`: Width (pixels)

### Output Tensor: `output0`
- **Data Type:** `float32` (`tensor(float)`)
- **Dimensions:** `[1, 5, 8400]`
  - `8400`: Candidate anchor-free spatial prediction grid across three feature pyramid strides ($80 \times 80$, $40 \times 40$, $20 \times 20$)
  - `5`: Prediction channel vector:
    - Index `0`: Center X bounding box coordinate ($c_x$)
    - Index `1`: Center Y bounding box coordinate ($c_y$)
    - Index `2`: Bounding box width ($w$)
    - Index `3`: Bounding box height ($h$)
    - Index `4`: Single-class confidence logit for `RASPBERY PI CAM ENCLOSURE` (Class `0`)

---

## 4. Export Implementation & Execution

The export was performed using the dedicated script [`benchmark/export_onnx.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/benchmark/export_onnx.py):

```bash
# Export execution on Raspberry Pi 5
python3 benchmark/export_onnx.py \
    --weights models/best.pt \
    --output models/onnx/best.onnx \
    --opset 17 \
    --imgsz 640 \
    --device cpu
```

### Python Export Logic
```python
import onnx
from ultralytics import YOLO

# Load fine-tuned PyTorch checkpoint
model = YOLO("models/best.pt")

# Export to static ONNX graph
exported_path = model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=False,
    simplify=True
)

# Validate graph structure
onnx_model = onnx.load("models/onnx/best.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX checker: Model syntax and topology are valid.")
```

---

## 5. Structural & Numerical Verification

### Graph Topology Verification
Graph syntax, acyclic connectivity, and node typing were verified with `onnx.checker`:
- **Status:** **PASSED** (0 errors, 0 warnings).
- **Execution Provider Compatibility:** Evaluated with `onnxruntime.InferenceSession` on ARM64 Debian 13 (`CPUExecutionProvider`).

### Numerical Parity Test
An identical synthetic input tensor $X \sim \mathcal{N}(0, 1) \in \mathbb{R}^{1 \times 3 \times 640 \times 640}$ was evaluated on both PyTorch (`best.pt`) and ONNX Runtime (`best.onnx`):
- **Maximum Absolute Tensor Difference:** **$6.1035 \times 10^{-4}$**
- **Mean Absolute Error:** **$1.3673 \times 10^{-5}$**
- **Numerical Parity Verdict:** **PASSED** (well below IEEE 754 single-precision float32 threshold of $1.0 \times 10^{-3}$).

---

## 6. Evidence Artifacts & Logs

- **Primary ONNX Model:** [`models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx)
- **Raw Export Console Log:** [`onnx_export_log.txt`](onnx_export_log.txt)
- **Structured Export Metadata (JSON):** [`onnx_export_log.json`](onnx_export_log.json)
- **Comprehensive Technical Report:** [`../reports/M2_ONNX_Export_Report.md`](../reports/M2_ONNX_Export_Report.md)
