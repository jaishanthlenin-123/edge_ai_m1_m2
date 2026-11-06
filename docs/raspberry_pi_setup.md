# Raspberry Pi 5 Deployment Guide: Edge AI Camera Enclosure Detection

This guide walks you through setting up, deploying, and benchmarking real-time object detection on your **Raspberry Pi 5** using the fine-tuned **YOLO26 Nano** model (`models/best.pt` or `models/best.onnx`).

---

## 1. Hardware & Platform Overview (Verified on Device)

| Property | Actual Device Specification |
| :--- | :--- |
| **Model** | **Raspberry Pi 5 Model B Rev 1.1** |
| **SoC / Architecture** | Broadcom BCM2712 Quad-Core ARM Cortex-A76 @ 2.4 GHz (`aarch64`) |
| **Installed RAM** | **4.0 GiB** (LPDDR4X) |
| **Storage** | 57 GB MicroSD card (`/dev/mmcblk0p2`), 29 GB available |
| **Operating System** | **Debian GNU/Linux 13 (trixie)** / Raspberry Pi OS 64-bit |
| **Python Version** | **Python 3.13.5** |
| **Pip Version** | **pip 25.1.1** |
| **Video Acceleration** | Raspberry Pi 5 ISP Back End (`pispbe`: `/dev/video20`–`video35`, `/dev/media0`, `/dev/media1`) |

> [!NOTE]
> **Inference Runtime Note (No TensorRT):**
> NVIDIA TensorRT is exclusively designed for NVIDIA CUDA GPUs and is **NOT supported** on the Raspberry Pi 5.
> The supported high-performance edge inference runtimes on Raspberry Pi 5 are:
> 1. **PyTorch ARM64** (`models/best.pt` via `ultralytics`) — Baseline CPU runtime.
> 2. **ONNX Runtime ARM64** (`models/best.onnx` via `onnxruntime`) — High-performance CPU runtime.

---

## 2. Step 1: System Package Preparation

Ensure required Debian / Raspberry Pi OS build headers and graphics libraries are present:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git libgl1 libglib2.0-0 v4l-utils
```

### Camera Subsystem Check
Verify available camera hardware using the Raspberry Pi camera CLI:

```bash
rpicam-hello --list-cameras
```

- If a camera module is detected (e.g. `imx708`, `imx219`, `ov5647`), you will see its camera number, sensor model, and CSI interface.
- If it returns `No cameras available!`, refer to [Section 6: Camera Diagnostics & Troubleshooting](#6-camera-diagnostics--troubleshooting).

---

## 3. Step 2: Clone Repository & Create Virtual Environment

Clone the repository to your Raspberry Pi:

```bash
cd ~
git clone https://github.com/jaishanthlenin-123/edge_ai_camera_enclosure.git
cd edge_ai_camera_enclosure
```

### Virtual Environment Setup
> [!IMPORTANT]
> Raspberry Pi OS enforces PEP 668 externally-managed environments. Always use a virtual environment with `--system-site-packages` so that the native system-installed `picamera2` and `libcamera` bindings are inherited inside the venv:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

---

## 4. Step 3: Install ARM64-Compatible Dependencies

### A. Install CPU-Optimized PyTorch for ARM64
> [!WARNING]
> **Critical ARM64 Compatibility Rule:**
> Blindly running `pip install torch` on ARM64 platforms pulls over 2.5 GB of NVIDIA CUDA packages (`cuda-toolkit`, `nvidia-cufft`, `nvidia-nvjitlink`, etc.) that are unusable on Broadcom GPUs and cause out-of-memory errors on 4GB Raspberry Pi boards.
> Always install the lightweight, official CPU wheel (170 MB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### B. Install Project Requirements
Install the remaining packages (`ultralytics`, `onnxruntime`, `opencv-python`, etc.):

```bash
pip install -r requirements.txt
```

### C. Verify Imports
```bash
python3 -c "import torch, ultralytics, onnxruntime, cv2, picamera2; print('All edge AI dependencies loaded successfully!')"
```

---

## 5. Step 4: Model Verification & Specifications

The trained model checkpoint is located at:
- **File:** `models/best.pt`
- **File Size:** `5,419,973 bytes` (~5.17 MB)
- **SHA256 Checksum:** `43c1abddc72896043ea52003c7a238c3b924dceb384673db0475e275d8620530`
- **Architecture:** YOLO26 Nano (`yolo26n`)
- **Number of Classes:** 1 (`RASPBERY PI CAM ENCLOSURE`, Class ID `0`)
- **Input Resolution:** 640 × 640

Verify the model file directly in Python:
```bash
python3 -c "from ultralytics import YOLO; m = YOLO('models/best.pt'); print('Task:', m.task); print('Classes:', m.names)"
```

---

## 6. Step 5: Camera Diagnostics & Troubleshooting

Before running live inference, verify camera connectivity:

### 1. Raspberry Pi CSI Camera Module (Picamera2)
- Raspberry Pi 5 uses 22-pin 0.5mm pitch flex ribbon cables (unlike the 15-pin 1mm cables of Pi 4).
- Ensure the cable is plugged into `CAM/DISP 0` or `CAM/DISP 1` with the contacts facing the correct side and the connector latch locked firmly.
- If `rpicam-hello --list-cameras` reports `No cameras available!`:
  1. Power off the Raspberry Pi (`sudo poweroff`).
  2. Re-seat both ends of the ribbon cable (at the Pi 5 board and at the sensor).
  3. Ensure `/boot/firmware/config.txt` contains `camera_auto_detect=1`.

### 2. USB Webcam Fallback
If using a standard USB webcam:
- Connect the webcam to a USB 3.0 (blue) or USB 2.0 port.
- Check recognized video nodes:
  ```bash
  v4l2-ctl --list-devices
  ```
- Use the detected index (typically `--source 0`).

---

## 7. Step 6: Run Real-Time Object Detection

### Option A: Using Raspberry Pi Camera Module (CSI via Picamera2)
```bash
python3 src/detect.py --model models/best.pt --source picam
```

### Option B: Headless Mode (SSH / Terminal without Display)
If connected remotely via SSH without an active Wayland/X11 monitor:
```bash
python3 src/detect.py --model models/best.pt --source picam --no-view
```

### Option C: Using USB Webcam
```bash
python3 src/detect.py --model models/best.pt --source 0
```

### Option D: Using Pre-Recorded Test Video
```bash
python3 src/detect.py --model models/best.pt --source path/to/test_video.mp4 --no-view
```

---

## 8. Actual Test Results (Raspberry Pi 5 CPU Baseline)

The following benchmark metrics were measured directly on this Raspberry Pi 5 hardware running the original trained PyTorch checkpoint (`models/best.pt`) on CPU without quantization or conversion:

| Metric | Measured Baseline (PyTorch FP32 CPU) |
| :--- | :--- |
| **Model Load Latency** | **193.1 ms** |
| **Inference Resolution** | **640 × 480** |
| **Average Latency** | **395.82 ms** |
| **Median Latency** | **356.00 ms** |
| **Min Latency** | **314.85 ms** |
| **Max Latency** | **716.48 ms** |
| **Effective Frame Rate** | **~2.53 FPS** |
| **Process RAM Footprint** | **360.2 MB** (+125.8 MB delta over idle) |
| **CPU Utilization** | **71.1%** (Process), 19.0% (System) |
| **SoC Thermal Profile** | **62.8°C** start → **71.6°C** peak (+8.8°C delta under load) |

---

## 9. Real-Time Controls

When the detection GUI window is active:
- **`Q`** or **`ESC`**: Exit the application safely.
- **`S`**: Save current detection frame and annotations to `evidence/M2/detections/`.
- **`+`**: Increase confidence threshold (reduces false detections).
- **`-`**: Decrease confidence threshold (increases sensitivity).

---

## 10. Summary of Resolved Compatibility Issues

1. **Python 3.13 / Debian 13 PEP 668:**
   - *Issue:* Debian 13 trixie prohibits modifying system Python packages.
   - *Resolution:* Created isolated venv using `--system-site-packages` to inherit the Debian-packaged `picamera2` and `libcamera` C++ bindings.
2. **ARM64 PyTorch CUDA Dependency Bloat:**
   - *Issue:* Default PyPI `torch` wheels on `aarch64` pull multi-gigabyte CUDA toolkits (`cuda-toolkit 13.2`, `nvidia-cufft`).
   - *Resolution:* Configured the official CPU-only index (`https://download.pytorch.org/whl/cpu`), installing the clean 170 MB ARM64 CPU build.
3. **Graceful Camera Absence Handling:**
   - *Issue:* Running `detect.py` without a connected camera produced unhandled exception loops.
   - *Resolution:* Added `is_opened()` verification to `CameraCapture` in `src/camera.py` and diagnostic error reporting in `src/detect.py`.
