# An Efficient Feature Extraction Method for Static Malware Analysis Using PE Header Files

A comparative study of feature set approaches for malware detection using Portable Executable (PE) header files and machine learning classifiers.

## Overview

This project develops an efficient feature extraction methodology for static malware analysis by extracting structural features from PE file headers. It compares two feature set approaches — a **reduced set (15 features)** and an **extended set (32 features)** — across six machine learning classifiers to evaluate the trade-off between feature dimensionality, extraction time, and classification accuracy.

### Key Results

| Classifier | Reduced (15) | Extended (32) | Improvement |
|---|---|---|---|
| **Random Forest** | **93.91%** | **94.78%** | +0.87% |
| Decision Tree | 91.74% | 93.48% | +1.74% |
| Gradient Boosting | 90.87% | 93.04% | +2.17% |
| KNN | 75.22% | 80.00% | +4.78% |
| Logistic Regression | 51.30% | 41.30% | -10.00% |
| SVM | 25.65% | 25.65% | 0.00% |

### Highlights

- **94.78% accuracy** with Random Forest on the extended feature set
- **75% reduction** in feature extraction time compared to prior work (APT1 dataset: ~19 min → ours: ~4.8 min)
- 1,150 malware samples from MalwareBazaar analyzed

## Project Structure

```
├── generate_charts.py                  # Full pipeline: reduced feature set (15 features)
├── generate_charts_more_features.py    # Full pipeline: extended feature set (32 features)
├── thesis_with_charts.py               # Modular implementation (15 features)
├── make_charts.py                      # Chart generation from existing CSV
├── rerun_all.py                        # Re-run both feature sets with consistent settings
├── malware_downloader.py               # Download samples from MalwareBazaar API
│
├── output_file_final.csv               # Extracted features: reduced set (1,150 × 16)
├── output_file_more_features.csv       # Extracted features: extended set (1,150 × 34)
│
├── classifier_comparison.png           # Performance comparison charts (reduced)
├── confusion_matrices.png              # Confusion matrices (reduced)
├── feature_importance.png              # Feature importance (reduced)
├── accuracy_comparison.png             # Accuracy ranking (reduced)
├── metrics_radar.png                   # Radar chart (reduced)
├── *_more_features.png                 # Same 5 charts for the extended set
│
├── Overleaf_Paper/                     # IEEE-formatted research paper (LaTeX + PDF)
├── Project_Documentation.tex           # Comprehensive project documentation (LaTeX)
├── Project_Documentation.pdf           # Compiled documentation (22 pages)
├── Paper_Updated.pdf                   # Updated research paper
└── File_Comparison_Analysis.md         # Script comparison analysis
```

## Feature Sets

### Reduced Feature Set (15 Features)

**FILE_HEADER (3):** Machine, NumberOfSections, Characteristics (target)

**OPTIONAL_HEADER (12):** AddressOfEntryPoint, ImageBase, SectionAlignment, FileAlignment, Subsystem, DllCharacteristics, MajorOperatingSystemVersion, MinorOperatingSystemVersion, MajorImageVersion, MinorImageVersion, SizeOfImage, SizeOfHeaders

### Extended Feature Set (32 Features)

All 15 from the reduced set, plus 17 additional fields:

**FILE_HEADER (+4):** TimeDateStamp, PointerToSymbolTable, NumberOfSymbols, SizeOfOptionalHeader

**OPTIONAL_HEADER (+14):** MajorLinkerVersion, MinorLinkerVersion, SizeOfCode, SizeOfInitializedData, SizeOfUninitializedData, BaseOfCode, MajorSubsystemVersion, MinorSubsystemVersion, CheckSum, SizeOfStackReserve, SizeOfStackCommit, SizeOfHeapReserve, SizeOfHeapCommit, LoaderFlags, NumberOfRvaAndSizes

## ML Classifiers

| Classifier | Configuration |
|---|---|
| Random Forest | 100 estimators, `n_jobs=-1` |
| K-Nearest Neighbors | k=5, Euclidean distance |
| Decision Tree | Information gain, `random_state=42` |
| SVM | RBF kernel, `gamma='scale'`, C=1.0 |
| Logistic Regression | `max_iter=1000` |
| Gradient Boosting | 50 estimators |

**Common settings:** 80/20 train-test split, 3-fold cross-validation, `random_state=42`

## Getting Started

### Prerequisites

```bash
pip install pefile pandas numpy scikit-learn matplotlib seaborn
```

### Usage

**Run full pipeline (reduced features):**
```bash
python3 generate_charts.py
```

**Run full pipeline (extended features):**
```bash
python3 generate_charts_more_features.py
```

**Regenerate charts from existing CSV:**
```bash
python3 make_charts.py
```

**Re-run both feature sets with consistent settings:**
```bash
python3 rerun_all.py
```

**Download malware samples:**
```bash
python3 malware_downloader.py
```

## Dataset

- **Source:** [MalwareBazaar](https://malwarebazaar.abuse.ch) by abuse.ch
- **Size:** 1,150 PE executable samples
- **Format:** Windows Portable Executable (.exe)
- **Target:** Characteristics field (multi-class classification)

## Feature Extraction Performance

| Metric | Reduced (15) | Extended (32) |
|---|---|---|
| Extraction Time | 215.81 sec | 290.12 sec |
| Processing Speed | 5.33 files/sec | 3.96 files/sec |
| Prior Work (APT1) | — | 1,157.81 sec |
| **Time Reduction** | **81%** | **75%** |

## Visualizations

Each feature set generates 5 charts:

1. **Classifier Comparison** — 4-panel figure with metrics, CV scores, F1 bars, and summary table
2. **Confusion Matrices** — 6 heatmaps (one per classifier)
3. **Feature Importance** — Random Forest feature ranking
4. **Accuracy Comparison** — Sorted bar chart
5. **Metrics Radar** — Spider chart across all classifiers and metrics

## Authors

- **Fabiha Jalal** — Islamic University of Technology (IUT)
- **Sadia Tasnim Dhruba** — Islamic University of Technology (IUT)
- **Onamika Hossain** — Islamic University of Technology (IUT)
- **Dr. Md Moniruzzaman** (Supervisor) — Assistant Professor, IUT

## License

This project is for academic and research purposes.
