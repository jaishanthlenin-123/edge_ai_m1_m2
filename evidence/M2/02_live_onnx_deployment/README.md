# Criterion 2: Convert and Run Live Deployment (4 Marks)

**Assessment Target:** M2 — Model Optimization & Evaluation
**Assessment Cell (Google Sheet):** **Cell F**
**Evidence Directory:** [`evidence/M2/02_live_onnx_deployment/`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M2/02_live_onnx_deployment/)
**Live Deployment Script:** [`src/detect_onnx.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/src/detect_onnx.py)
**Manual Camera Test Suite:** [`src/manual_camera_test.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/src/manual_camera_test.py)

---

## 1. Overview & Live Deployment Architecture

This section documents the live deployment and real-time execution of the optimized **ONNX Runtime** engine directly on the **Raspberry Pi 5 Model B**. The detector runs live video inference, performing end-to-end frame capture, letterbox preprocessing, static graph inference, anchor-free box decoding, Non-Maximum Suppression (NMS), and real-time Heads-Up Display (HUD) rendering.

```
+----------------------------------------------------------------------------------------------------+
|                         END-TO-END RASPBERRY PI 5 ONNX PIPELINE                                   |
|                                                                                                    |
|  [ Camera Capture ]  --->  [ Letterbox Preprocess ]  --->  [ ONNX Runtime CPU ]                   |
|   V4L2 / Picamera2          Aspect-preserving 640x640        Cortex-A76 NEON Vectorization        |
|   640x480 Raw BGR           Float32 Normalization [0,1]      Latency: ~166.6 ms (6.00 FPS)        |
|                                                                      |                             |
|  [ Real-Time Display ] <--- [ HUD Annotation & Web ] <---  [ Anchor-Free Postprocess ]            |
|   Local GUI Window          MJPEG Server (:5000)             Coordinate Unpadding                 |
|   60 FPS Screen Blit        Confidence & Latency HUD         NMS (IoU=0.70, Conf=0.35)            |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Live Deployment Visual Evidence

The following genuine frame captures demonstrate live object detection of the **Raspberry Pi Camera Enclosure** using [`models/onnx/best.onnx`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/models/onnx/best.onnx) on Raspberry Pi 5:

### 2.1 Live Detection Captures with HUD Overlay

| Frame Description | Live Detection Visual Evidence |
| :--- | :--- |
| **Successful USB Camera Detection**<br>Live detection frame showing tight bounding box around the camera enclosure with 82.0% confidence score and HUD telemetry. | ![Successful USB Detection](successful_usb_detection.jpg) |
| **Real-Time Live Frame Stream**<br>Full live frame capture displaying real-time FPS overlay, inference time readout, and high-confidence enclosure identification. | ![Live Frame Capture](live_frame.jpg) |
| **Active Camera Viewport (640x480)**<br>Current camera viewport during sustained live detection session with stable tracking. | ![Current Camera View](current_cam_view.jpg) |

---

### 2.2 Multi-Angle & Distance Live Test Sequences

| Frame ID | Scene Description | Captured Frame |
| :--- | :--- | :--- |
| **Detection Frame 152227** | Handheld enclosure test at 0.5m distance under indoor LED lighting. | ![Detection 152227](detection_20261106_152227.jpg) |
| **Detection Frame 152238** | Angled enclosure view with connected camera ribbon cable visible. | ![Detection 152238](detection_20261106_152238.jpg) |
| **Detection Frame 152328** | Desk-mounted placement under variable ambient lighting. | ![Detection 152328](detection_20261106_152328.jpg) |
| **Manual Test 001** | High-definition annotated verification capture (1080p source). | ![Test 001](test_001_annotated.jpg) |
| **Manual Test 002** | High-definition annotated verification capture showing boundary precision. | ![Test 002](test_002_annotated.jpg) |

---

## 3. Deployment Script Walkthrough: `src/detect_onnx.py`

The production deployment script [`src/detect_onnx.py`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/src/detect_onnx.py) implements pure ONNX Runtime execution with zero PyTorch dependencies:

### Key Pipeline Stages:
1. **Engine Initialization:**
   ```python
   import onnxruntime as ort

   # Configure execution session on ARM64 Cortex-A76
   session_options = ort.SessionOptions()
   session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   session_options.intra_op_num_threads = 4  # Utilize all 4 Cortex-A76 cores

   session = ort.InferenceSession(
       "models/onnx/best.onnx",
       sess_options=session_options,
       providers=["CPUExecutionProvider"]
   )
   ```

2. **Aspect-Preserving Letterbox Preprocessing:**
   Input frames are padded with neutral gray (`(114, 114, 114)`) to $640 \times 640$, preserving the native camera aspect ratio without geometric distortion.

3. **Anchor-Free Output Decoding & Coordinate Unpadding:**
   The output tensor `[1, 5, 8400]` is sliced into center coordinates $(c_x, c_y)$, dimensions $(w, h)$, and objectness logits. Bounding box coordinates are dynamically unpadded back to the native camera frame resolution.

4. **Non-Maximum Suppression (NMS):**
   Applies fast multiclass NMS using OpenCV `cv2.dnn.NMSBoxes` with $IoU = 0.70$ and $Confidence = 0.35$.

---

## 4. Execution Instructions on Raspberry Pi 5

```bash
# Activate virtual environment
source venv/bin/activate

# Run live detection with USB Camera (/dev/video0)
python3 src/detect_onnx.py --model models/onnx/best.onnx --source 0

# Run live detection with Raspberry Pi Camera Module (CSI-2 via Picamera2)
python3 src/detect_onnx.py --model models/onnx/best.onnx --source picam

# Run with custom confidence threshold and MJPEG web stream
python3 src/detect_onnx.py --model models/onnx/best.onnx --source 0 --conf 0.40 --webstream
```

### Keyboard Controls During Live Operation:
- **`Q`** or **`ESC`**: Terminate live detection session cleanly.
- **`S`**: Save high-resolution detection frame with HUD annotations to disk.
- **`+` / `-`**: Dynamically increase or decrease detection confidence threshold in real time.

---

## 5. Live Runtime Performance Summary

- **Inference Latency:** **166.6 ms** (~6.00 FPS) on Raspberry Pi 5 CPU.
- **End-to-End Pipeline Throughput:** **3.33 FPS** (including camera capture, decoding, and HUD rendering).
- **CPU Utilization:** **66.16%** across 4 cores (stable, fan operating normally).
- **Core Temperature:** Stabilized at **61.7°C** (thermal throttling register: `0x0`).
- **Detection Stability:** Smooth bounding box tracking with zero flicker and zero false-positive activations on backgrounds.
