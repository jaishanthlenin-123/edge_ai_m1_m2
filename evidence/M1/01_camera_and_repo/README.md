# M1 Assessment Criterion 1: Camera & Repository Setup

**Assessment Item:** 01_camera_and_repo  
**Status:** Complete & Verified  

---

## Overview

This module verifies the initial hardware environment configuration, Git repository initialization, and camera hardware interface setup for the Edge AI detection system.

## Key Artifacts & Implementation

- **Hardware Verification:** Configured for dual-mode execution on Raspberry Pi 5 (using `libcamera` / `Picamera2`) and desktop development workstations (using standard USB / V4L2 webcams via OpenCV).
- **Camera Connectivity Script:** [`test_connection.py`](../../../test_connection.py) – Verifies device indexing, hardware exposure, and frame buffer polling.
- **Camera Driver Module:** [`src/camera.py`](../../../src/camera.py) – Implements dual Picamera2 / OpenCV capture classes with automatic fallback.
- **Execution Script:** [`open_camera.bat`](../../../open_camera.bat) – One-click launcher for rapid camera diagnostics.

## Verification Outcome

- Camera sensor recognized at index `0`.
- Frame acquisition verified at 640×640 native letterbox resolution.
- Remote repository connected: [`jaishanthlenin-123/edge_ai_m1_m2`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2).

