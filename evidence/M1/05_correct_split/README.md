# M1 Assessment Criterion 5: Dataset Split Verification

**Assessment Item:** 05_correct_split  
**Status:** Complete & Verified  

---

## Overview

This module verifies that the dataset splits across training, validation, and testing sets are correctly balanced, isolated, and representative of real-world operational domains without data leakage.

## Key Evidence Files

- **Class Distribution Table (CSV):** [`evidence/M1/class_distribution.csv`](../class_distribution.csv)
- **Dataset Configuration:** [`config/data.yaml`](../../../config/data.yaml)

## Split Distribution Breakdown

```
========================================================================================
                          DATASET SPLIT AUDIT SUMMARY
========================================================================================
Split Stage         Source Images   Percentage   Target Annotation Count
----------------------------------------------------------------------------------------
Train Set           88 images       71.0%        88 bounding box targets
Validation Set      24 images       19.3%        24 bounding box targets
Test Set            12 images        9.7%        12 bounding box targets
----------------------------------------------------------------------------------------
Total (v1 Baseline) 124 images     100.0%       124 targets (1 class)
========================================================================================
```

- **Stratification Check:** Zero overlap between training and testing environments.
- **Single-Class Model:** Focused exclusively on class `0`: `'RASPBERY PI CAM ENCLOSURE'`.

