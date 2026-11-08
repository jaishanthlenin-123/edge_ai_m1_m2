# Milestone 1 (M1) Central Evidence Index & Navigation

This directory serves as the primary evaluation portal for **Milestone 1 (M1)** of the Edge AI Raspberry Pi Camera Enclosure Detection project. It aggregates all experimental verification records, baseline benchmarks, dataset audit artifacts, and the hardware CAD engineering evidence.

---

## 1. M1 Assessment Navigation Matrix

All Milestone 1 assessment requirements are systematically indexed below with direct repository navigation:

| # | Assessment Category | Directory / Evidence Path | Description & Deliverables |
| :-: | :--- | :--- | :--- |
| **01** | **Camera & Repo Setup** | [`01_camera_and_repo/`](./01_camera_and_repo/) | Repository initialization, hardware config, dual-backend camera driver (`src/camera.py`), connection tests |
| **02** | **OpenCV Pipeline** | [`02_opencv_pipeline/`](./02_opencv_pipeline/) | End-to-end video acquisition loop, 640×640 letterbox preprocessing, HUD telemetry overlays |
| **03** | **Baseline Speed Benchmark**| [`03_baseline_speed/`](./03_baseline_speed/) | Baseline latency (8,880.5 ms / 0.11 FPS on unaccelerated CPU), raw timing CSVs, summary logs |
| **04** | **Dataset Version 1 Audit** | [`04_dataset_v1/`](./04_dataset_v1/) | Inspection of 124 original images, tabular annotation error catalog, false-positive diagnosis |
| **05** | **Correct Split Verification**| [`05_correct_split/`](./05_correct_split/) | Split distribution audit: Train (88 imgs / 71%), Valid (24 imgs / 19%), Test (12 imgs / 10%) |
| **06** | **Train & Run Detector** | [`06_train_and_run_detector/`](./06_train_and_run_detector/) | Initial YOLO26 Nano training, test set accuracy metrics, 23 visual detection test cases, 19 live video frames |
| **07** | **CAD Design Evidence (IDRP)**| [`07_cad_design/`](./07_cad_design/) | Preserved CAD archive (`ENCLOSURE_JAISHANTH.f3d`), engineering reports, multi-view drawings, 3D print specs |

---

## 2. Dedicated CAD Design Evidence (IDRP Hardware Component)

The Raspberry Pi Camera Enclosure is the primary hardware component developed as an **Individual Design and Research Project (IDRP)** by **Jaishanth**. This physical enclosure houses the camera module and represents the target ground-truth object class detected by the Edge AI vision model.

![Multi-View CAD Specification Drawing](./07_cad_design/cad_multi_view_drawing.png)

### 2.1 CAD Model Verification & Integrity

- **Original CAD File:** [`ENCLOSURE_JAISHANTH.f3d`](./07_cad_design/ENCLOSURE_JAISHANTH.f3d)
- **Direct GitHub Link (CAD File):** [Download / Inspect `ENCLOSURE_JAISHANTH.f3d`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/blob/main/evidence/M1/07_cad_design/ENCLOSURE_JAISHANTH.f3d)
- **CAD Evidence Folder:** [`evidence/M1/07_cad_design/`](https://github.com/jaishanthlenin-123/edge_ai_m1_m2/tree/main/evidence/M1/07_cad_design/)
- **CAD Format:** Autodesk Fusion 360 Document Archive (`.f3d`)
- **Geometry Kernel:** Autodesk Neutron / Autodesk Shape Manager (ASM 232.5.0.65535 NT)
- **File Size:** **874,983 bytes** (854.48 KB / ~875 KB)
- **SHA-256 Checksum:** `84444cee13753f74e86be3e41a7b0b865c8727c2576d06237835fc374914a4e1`
- **Integrity Status:** **Verified 100% accessible and non-corrupted.**

### 2.2 Genuine Previews & Engineering Views

- **3D Isometric View (Fusion 360 Native Render):** [`cad_isometric_view.png`](./07_cad_design/cad_isometric_view.png)
- **Front Elevation View (X-Y Projection):** [`cad_front_view.png`](./07_cad_design/cad_front_view.png)
- **Right Side Elevation View (Z-Y Projection):** [`cad_side_view.png`](./07_cad_design/cad_side_view.png)
- **Top Plan View (X-Z Projection):** [`cad_top_view.png`](./07_cad_design/cad_top_view.png)
- **Comprehensive Drawing Sheet:** [`cad_multi_view_drawing.png`](./07_cad_design/cad_multi_view_drawing.png)

### 2.3 Physical Specifications Summary

- **Envelope Dimensions:** Width (X) = **64.0 mm** | Length (Z) = **71.0 mm** | Exploded Height (Y) = **36.0 mm** (Assembled Chassis Height: ~18.0 mm).
- **Lens Aperture:** **Ø 16.0 mm** circular opening centered with optical axis.
- **Camera Standoffs:** 4 corner cylindrical standoffs matching standard Raspberry Pi Camera 21.0 mm × 12.5 mm mounting pattern.
- **Cable Routing:** Lateral clearance slot (**18.0 mm × 2.5 mm**) for 15-pin 16 mm flexible ribbon cable.
- **Material Specification:** 3D Printable Polymer (PLA / PETG recommended for FDM manufacturing).
- **Photographic Status:** Real-time camera detection frames verified; standalone macro photography tracked in [`PENDING_EVIDENCE_CHECKLIST.md`](./07_cad_design/PENDING_EVIDENCE_CHECKLIST.md).

---

## 3. Detailed CAD Documentation Index

For complete technical analysis, refer to the dedicated CAD engineering documents:

1. **Full Engineering Report:** [`M1_CAD_Design_Report.md`](./07_cad_design/M1_CAD_Design_Report.md) – In-depth analysis of purpose, design features, 3D print parameters, and relationship to the Edge AI detection pipeline.
2. **Technical Specification:** [`CAD_Design_Specification.md`](./07_cad_design/CAD_Design_Specification.md) – Exact geometric dimensions, assembly stack-up, manufacturing tolerances, and slicer configuration.
3. **File Metadata & Forensics:** [`CAD_File_Information.md`](./07_cad_design/CAD_File_Information.md) – Archive internal stream structure, ASM solid B-reps, checksums, and instructions for Autodesk Viewer.
4. **Physical Photo Checklist:** [`PENDING_EVIDENCE_CHECKLIST.md`](./07_cad_design/PENDING_EVIDENCE_CHECKLIST.md) – Capture protocols for physical 3D-printed enclosure photographs.

---

## 4. Summary of Dataset Findings (Initial 124 Images)

1. **Initial Model Failure Root Cause:**
   - The initial Roboflow model had significant bounding box misalignment (boxes enclosing fingers, empty background, or table corners instead of the camera enclosure).
   - Class label was initially recorded in Roboflow as `'RASPBERY PI CAM ENCLOSURE'`.
2. **Audit Artifacts:**
   - [`dataset_audit_report.md`](./dataset_audit_report.md): Detailed image-by-image inspection of all 124 original images.
   - [`annotation_errors.csv`](./annotation_errors.csv): Tabular log of identified box misalignments, loose bounds, and background errors.
   - [`class_distribution.csv`](./class_distribution.csv): Split breakdown across train (88), valid (24), and test (12).
   - [`M1_Baseline_Performance_Report.md`](./M1_Baseline_Performance_Report.md): Baseline performance benchmarking before ONNX acceleration.
3. **Correction:**
   - Ground truth annotations were cleaned and tightened locally and verified against real-world test scenes before progressing to Milestone 2.
