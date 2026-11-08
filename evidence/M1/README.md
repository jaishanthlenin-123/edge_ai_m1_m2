# Milestone 1 (M1) Central Evidence Index & Navigation

This directory serves as the primary evaluation portal for **Milestone 1 (M1)** of the Edge AI Raspberry Pi Camera Enclosure Detection project. It aggregates all experimental verification records, baseline benchmarks, dataset audit artifacts, the hardware CAD engineering evidence, and the **Roboflow Dataset Version 3** integration.

---

## 1. M1 Assessment Navigation Matrix

All Milestone 1 assessment requirements are systematically indexed below with direct repository navigation:

| # | Assessment Category | Directory / Evidence Path | Description & Deliverables |
| :-: | :--- | :--- | :--- |
| **01** | **Camera & Repo Setup** | [`01_camera_and_repo/`](./01_camera_and_repo/) | Repository initialization, hardware config, dual-backend camera driver (`src/camera.py`), connection tests |
| **02** | **OpenCV Pipeline** | [`02_opencv_pipeline/`](./02_opencv_pipeline/) | End-to-end video acquisition loop, 640×640 letterbox preprocessing, HUD telemetry overlays |
| **03** | **Baseline Speed Benchmark**| [`03_baseline_speed/`](./03_baseline_speed/) | Baseline latency (8,880.5 ms / 0.11 FPS on unaccelerated CPU), raw timing CSVs, summary logs |
| **04** | **Dataset Version 1 Audit & Roboflow v3** | [`04_dataset_v1/`](./04_dataset_v1/) | 124-image audit, error log, Roboflow project overview, Version 3 download & visual previews |
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

---

## 3. Roboflow Dataset Version 3 & Dataset Evolution

The repository incorporates the official **Roboflow Dataset Version 3** downloaded directly from the Roboflow platform:

- **Roboflow Project URL:** [https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)
- **Downloaded Dataset Location:** [`datasets/Roboflow_M1_M2_V3/`](../../datasets/Roboflow_M1_M2_V3/)
- **Dataset Version 3 Specification:** [`datasets/Roboflow_M1_M2_V3/README.md`](../../datasets/Roboflow_M1_M2_V3/README.md)
- **Dataset Version History Document:** [`dataset_version_history.md`](./dataset_version_history.md)
- **Roboflow Project & Version Context:** [`04_dataset_v1/roboflow_project.md`](./04_dataset_v1/roboflow_project.md)
- **Visual Dataset Previews (Raw & Annotated):** [`04_dataset_v1/roboflow_v3_previews/`](./04_dataset_v1/roboflow_v3_previews/)

### Dataset Statistics at a Glance
- **Total Images:** **274 images** (Train: 230 [83.9%], Valid: 30 [11.0%], Test: 14 [5.1%])
- **Total Annotations:** **239 annotated targets** (211 bboxes + 28 polygon segmentations)
- **True Negatives:** **35 empty labels** (30 train, 3 valid, 2 test) verifying 100% background rejection
- **Resolution:** $640 \times 640$ Fit (black edges)
- **Total Size:** **8.65 MB** (274 images, 274 labels, `data.yaml`)

---

## 4. Detailed Documentation Index

1. **Full CAD Engineering Report:** [`M1_CAD_Design_Report.md`](./07_cad_design/M1_CAD_Design_Report.md)
2. **CAD Technical Specification:** [`CAD_Design_Specification.md`](./07_cad_design/CAD_Design_Specification.md)
3. **CAD File Metadata & Forensics:** [`CAD_File_Information.md`](./07_cad_design/CAD_File_Information.md)
4. **Physical Photo Checklist:** [`PENDING_EVIDENCE_CHECKLIST.md`](./07_cad_design/PENDING_EVIDENCE_CHECKLIST.md)
5. **Dataset Version History:** [`dataset_version_history.md`](./dataset_version_history.md)
6. **Roboflow Project Documentation:** [`04_dataset_v1/roboflow_project.md`](./04_dataset_v1/roboflow_project.md)
7. **Baseline Performance Report:** [`M1_Baseline_Performance_Report.md`](./M1_Baseline_Performance_Report.md)
8. **Initial Dataset Audit Report (124 Images):** [`dataset_audit_report.md`](./dataset_audit_report.md)
