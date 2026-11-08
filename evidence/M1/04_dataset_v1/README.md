# M1 Assessment Criterion 4: Dataset Version 1 Audit

**Assessment Item:** 04_dataset_v1  
**Status:** Complete & Verified  

---

## Overview

This module documents the comprehensive quality assurance audit performed on the initial 124-image training dataset in Roboflow, detailing error cataloging and root-cause analysis of early model failures.

## Key Evidence Files

- **Dataset Audit Report:** [`evidence/M1/dataset_audit_report.md`](../dataset_audit_report.md) – Image-by-image inspection of all 124 original images.
- **Annotation Errors Log (CSV):** [`evidence/M1/annotation_errors.csv`](../annotation_errors.csv) – Tabular catalog of bounding box misalignments, loose bounds, and background errors.

## Key Audit Findings

1. **Bounding Box Misalignments:** Initial annotations enclosed extraneous user fingers, shadows, and cluttered background surfaces rather than tight boundaries on the enclosure.
2. **Class Typo Identified:** Original project label recorded in Roboflow as `'RASPBERY PI CAM ENCLOSURE'` (single 'r').
3. **Absence of Negative Samples:** Zero true negative images existed in v1, causing the initial model to produce 73.2% confidence false positives on blank curtains and plain walls.
4. **Corrective Action:** Annotations were systematically tightened, and 20 verified true negative images were integrated into Dataset Version 3, achieving 0.0% false positives.

