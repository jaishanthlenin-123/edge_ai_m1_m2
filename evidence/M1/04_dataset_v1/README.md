# M1 Assessment Criterion 4: Dataset Version 1 Audit & Roboflow Evolution

**Assessment Item:** 04_dataset_v1  
**Status:** Complete & Verified  

---

## 1. Overview

This module documents the comprehensive quality assurance audit performed on the initial 124-image training dataset in Roboflow, detailing error cataloging, root-cause analysis of early model failures, and the subsequent evolution to **Roboflow Dataset Version 3**.

---

## 2. Key Evidence & Documentation Files

- **Dataset Audit Report:** [`evidence/M1/dataset_audit_report.md`](../dataset_audit_report.md) – Image-by-image inspection of all 124 original baseline images.
- **Annotation Errors Log (CSV):** [`evidence/M1/annotation_errors.csv`](../annotation_errors.csv) – Tabular catalog of bounding box misalignments, loose bounds, and background errors.
- **Dataset Version History:** [`evidence/M1/dataset_version_history.md`](../dataset_version_history.md) – Evolution from 124 baseline images to 274 Version 3 images.
- **Roboflow Project & Version 3 Overview:** [`roboflow_project.md`](./roboflow_project.md) – Roboflow platform context and Version 3 specifications.
- **Visual Annotation Previews:** [`roboflow_v3_previews/`](./roboflow_v3_previews/) – Genuine side-by-side raw vs. annotated examples from the downloaded Version 3 dataset.
- **Downloaded Version 3 Dataset:** [`datasets/Roboflow_M1_M2_V3/`](../../../datasets/Roboflow_M1_M2_V3/) – Local copy of all 274 images and YOLO label files.
- **Version 3 Dataset README:** [`datasets/Roboflow_M1_M2_V3/README.md`](../../../datasets/Roboflow_M1_M2_V3/README.md) – Complete dataset specifications and audit breakdown.

---

## 3. Key Audit Findings & Corrective Evolution

1. **Bounding Box Misalignments:** Initial annotations in Version 1 enclosed extraneous user fingers, shadows, and cluttered background surfaces rather than tight boundaries on the enclosure.
2. **Class Typo Identified:** Original project label recorded in Roboflow as `'RASPBERY PI CAM ENCLOSURE'` (single 'r').
3. **Absence of Negative Samples:** Zero true negative images existed in Version 1, causing the initial model to produce 73.2% confidence false positives on blank curtains and plain walls.
4. **Corrective Action & Version 3:** Ground truth annotations were systematically tightened, and **20 verified true negative images** were integrated into **Dataset Version 3** (augmented to 35 empty label files in the export), completely eliminating background activations (**0.0% false positives**).
5. **Roboflow Project Link:** [https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)
