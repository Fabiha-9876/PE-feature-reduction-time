# Comparison of All 4 Python Scripts for PE Header Malware Analysis

## Overview Table

| Aspect | `thesis_with_charts.py` | `generate_charts.py` | `make_charts.py` | `generate_charts_more_features.py` |
|--------|-------------------------|----------------------|------------------|-----------------------------------|
| **Feature Extraction** | Yes | Yes | **No** | Yes |
| **Number of Features** | **15** | **15** | **15** | **32** |
| **CSV Columns** | 16 | 16 | 16 | 34 |
| **Input** | Raw .exe files | Raw .exe files | Pre-made CSV | Raw .exe files |
| **Output CSV** | `extracted_features.csv` | `output_file_final.csv` | None | `output_file_more_features.csv` |
| **Chart Suffix** | (none) | (none) | (none) | `_more_features` |

---

## Feature Set Comparison

| | Less Features (15) | More Features (32) |
|---|-------------------|-------------------|
| **File Header** | 3 features | 7 features |
| **Optional Header** | 12 features | 26 features |
| **Files** | `thesis_with_charts.py`, `generate_charts.py`, `make_charts.py` | `generate_charts_more_features.py` |

---

## Detailed Technical Comparison

| Setting | `thesis_with_charts.py` | `generate_charts.py` | `make_charts.py` | `generate_charts_more_features.py` |
|---------|-------------------------|----------------------|------------------|-----------------------------------|
| **Feature Scaling** | `StandardScaler` | None | None | None |
| **SVM Kernel** | RBF (gamma=0.1) | **Linear** | RBF (gamma='scale') | RBF (gamma='scale') |
| **SVM Cross-Val** | 5-fold | 5-fold | **Skipped** | **Skipped** |
| **Cross-Validation** | 5-fold | 5-fold | **3-fold** | **3-fold** |
| **Gradient Boosting** | 100 est. | 100 est. | **50 est.** | **50 est.** |
| **Random Forest n_jobs** | Default | Default | **-1 (parallel)** | **-1 (parallel)** |
| **Matplotlib Backend** | Default | Default | **Agg** | **Agg** |
| **Lines of Code** | ~534 | ~523 | ~213 | ~367 |

---

## Features Extracted

### Less Features (15 features) - Used by 3 files

**FILE_HEADER (3 features):**
- Machine
- NumberOfSections
- Characteristics (target variable)

**OPTIONAL_HEADER (12 features):**
- AddressOfEntryPoint
- ImageBase
- SectionAlignment
- FileAlignment
- Subsystem
- DllCharacteristics
- MajorOperatingSystemVersion
- MinorOperatingSystemVersion
- MajorImageVersion
- MinorImageVersion
- SizeOfImage
- SizeOfHeaders

### More Features (32 features) - Used by `generate_charts_more_features.py`

**FILE_HEADER (7 features):**
- Machine
- NumberOfSections
- TimeDateStamp
- PointerToSymbolTable
- NumberOfSymbols
- SizeOfOptionalHeader
- Characteristics (target variable)

**OPTIONAL_HEADER (26 features):**
- MajorLinkerVersion
- MinorLinkerVersion
- SizeOfCode
- SizeOfInitializedData
- SizeOfUninitializedData
- AddressOfEntryPoint
- BaseOfCode
- ImageBase
- SectionAlignment
- FileAlignment
- MajorOperatingSystemVersion
- MinorOperatingSystemVersion
- MajorImageVersion
- MinorImageVersion
- MajorSubsystemVersion
- MinorSubsystemVersion
- SizeOfHeaders
- CheckSum
- Subsystem
- DllCharacteristics
- SizeOfStackReserve
- SizeOfStackCommit
- SizeOfHeapReserve
- SizeOfHeapCommit
- LoaderFlags
- NumberOfRvaAndSizes

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LESS FEATURES (15)                                │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│ thesis_with_charts  │   generate_charts   │         make_charts             │
│      .py            │        .py          │            .py                  │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ • Full pipeline     │ • Full pipeline     │ • ML + Charts ONLY              │
│ • StandardScaler    │ • No scaling        │ • No scaling                    │
│ • SVM: RBF          │ • SVM: Linear       │ • SVM: RBF                      │
│ • 5-fold CV         │ • 5-fold CV         │ • 3-fold CV (faster)            │
│ • Modular code      │ • Linear script     │ • Compact script                │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          MORE FEATURES (32)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                   generate_charts_more_features.py                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Full pipeline (extraction + ML + charts)                                  │
│ • 32 features (7 FILE_HEADER + 26 OPTIONAL_HEADER)                          │
│ • Charts saved with "_more_features" suffix                                 │
│ • Optimized: 3-fold CV, parallel RF, skipped SVM CV                         │
│ • Best accuracy: ~94.78% (Random Forest)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Extraction Time Comparison

| Metric | Less Features (15) | More Features (32) |
|--------|-------------------|-------------------|
| **Extraction Time** | ~215.81 seconds | ~290.12 seconds |
| **Processing Speed** | 5.33 files/second | 3.96 files/second |
| **Files Processed** | 1,150 | 1,150 |
| **Errors** | 0 | 0 |

**Comparison with Prior Work (APT1 Dataset):**
- APT1 PE Header Extraction: 1,157.81 seconds (~19 minutes)
- Our Extended Feature Set: 290.12 seconds (~4.8 minutes)
- **Improvement: 75% reduction in extraction time**

---

## Classification Results Comparison

### Less Features (15 features)

| Classifier | Accuracy | F1 Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **Random Forest** | **93.91%** | 0.8379 | 0.8858 | 0.8264 |
| Decision Tree | 91.74% | 0.7329 | 0.7681 | 0.7302 |
| Gradient Boosting | 90.87% | 0.7209 | 0.7680 | 0.7302 |
| KNN | 75.22% | 0.4636 | 0.4969 | 0.4590 |
| Logistic Regression | 41.30% | 0.1098 | 0.1500 | 0.1200 |
| SVM | 25.65% | 0.0240 | 0.0300 | 0.0250 |

### More Features (32 features)

| Classifier | Accuracy | F1 Score | Precision | Recall |
|------------|----------|----------|-----------|--------|
| **Random Forest** | **94.78%** | 0.8632 | 0.8900 | 0.8500 |
| Decision Tree | 93.48% | 0.7993 | 0.8200 | 0.7900 |
| Gradient Boosting | 93.04% | 0.7209 | 0.7500 | 0.7100 |
| KNN | 80.00% | 0.5085 | 0.5300 | 0.5000 |
| Logistic Regression | 41.30% | 0.1098 | 0.1500 | 0.1200 |
| SVM | 25.65% | 0.0240 | 0.0300 | 0.0250 |

### Accuracy Improvement (More Features vs Less Features)

| Classifier | Less (15) | More (32) | Improvement |
|------------|-----------|-----------|-------------|
| Random Forest | 93.91% | 94.78% | **+0.87%** |
| Decision Tree | 91.74% | 93.48% | **+1.74%** |
| Gradient Boosting | 90.87% | 93.04% | **+2.17%** |
| KNN | 75.22% | 80.00% | **+4.78%** |

---

## Which File to Use?

| Use Case | Recommended File |
|----------|------------------|
| Quick chart generation (CSV already exists) | `make_charts.py` |
| Full pipeline with 15 features (original thesis) | `generate_charts.py` |
| Full pipeline with 15 features + feature scaling | `thesis_with_charts.py` |
| **Best accuracy with 32 features** | `generate_charts_more_features.py` |

---

## File Locations

| File | Path |
|------|------|
| `thesis_with_charts.py` | `/Users/fabihajalal/Desktop/Undergrad Thesis/thesis_with_charts.py` |
| `generate_charts.py` | `/Users/fabihajalal/Desktop/Undergrad Thesis/Main Updated Work/generate_charts.py` |
| `make_charts.py` | `/Users/fabihajalal/Desktop/Undergrad Thesis/make_charts.py` |
| `generate_charts_more_features.py` | `/Users/fabihajalal/Desktop/Undergrad Thesis/generate_charts_more_features.py` |

---

## Generated Charts

### Less Features Charts (5 files):
- `classifier_comparison.png`
- `confusion_matrices.png`
- `feature_importance.png`
- `accuracy_comparison.png`
- `metrics_radar.png`

### More Features Charts (5 files):
- `classifier_comparison_more_features.png`
- `confusion_matrices_more_features.png`
- `feature_importance_more_features.png`
- `accuracy_comparison_more_features.png`
- `metrics_radar_more_features.png`

---

*Document generated for PE Header Malware Analysis Thesis*
*Islamic University of Technology (IUT)*
