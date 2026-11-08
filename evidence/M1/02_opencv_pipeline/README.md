# M1 Assessment Criterion 2: OpenCV Pipeline Implementation

**Assessment Item:** 02_opencv_pipeline  
**Status:** Complete & Verified  

---

## Overview

This module documents the real-time computer vision pipeline responsible for frame ingestion, preprocessing, letterbox transformation, HUD overlay rendering, and post-inference bounding box visualization.

## Key Artifacts & Implementation

- **Inference & Pipeline Driver:** [`src/detect.py`](../../../src/detect.py) – Manages the end-to-end loop: Frame Capture $\rightarrow$ Letterbox Resizing $\rightarrow$ Tensor Conversion $\rightarrow$ Model Inference $\rightarrow$ NMS $\rightarrow$ Screen Rendering.
- **Visualizer & Drawing Utilities:** [`src/utils.py`](../../../src/utils.py) – Generates high-visibility bounding boxes, confidence score tags, and real-time Heads-Up Display (HUD) showing status, latency, and FPS.
- **Desktop Launcher:** [`run_webcam.bat`](../../../run_webcam.bat) – Standard webcam inference launcher.
- **Roboflow v3 Launcher:** [`run_webcam_roboflow_v3.bat`](../../../run_webcam_roboflow_v3.bat) – Dedicated launcher using fine-tuned weights.

## Verification Outcome

- Pipeline achieves zero frame drops during video acquisition.
- Interactive controls functional: `Q` (Quit), `S` (Save Frame), `+` / `-` (Threshold adjustment).

