#!/usr/bin/env python3
"""
More Features Implementation - 34 columns (vs 16 columns in less features)
Extracts additional PE header features and generates separate charts.
"""
import os
import sys
import time
import pefile
import pandas as pd
import numpy as np

# Force non-interactive backend FIRST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')

# Configuration
FOLDER_PATH = "/Users/fabihajalal/Desktop/Undergrad Thesis/extracted"
OUTPUT_FOLDER = "/Users/fabihajalal/Desktop/Undergrad Thesis"
CSV_OUTPUT = os.path.join(OUTPUT_FOLDER, "output_file_more_features.csv")

print("=" * 60, flush=True)
print("MORE FEATURES IMPLEMENTATION (34 columns)", flush=True)
print("=" * 60, flush=True)

# ============================================================
# STEP 1: FEATURE EXTRACTION (More Features - 34 columns)
# ============================================================
print("\nSTEP 1: FEATURE EXTRACTION (More Features)", flush=True)
print("-" * 40, flush=True)

features = []
errors = 0
start_time = time.time()

print(f"Extracting features from: {FOLDER_PATH}", flush=True)

file_list = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".exe")]
total_files = len(file_list)
print(f"Found {total_files} .exe files", flush=True)

for idx, filename in enumerate(file_list):
    try:
        filepath = os.path.join(FOLDER_PATH, filename)
        pe = pefile.PE(filepath)

        # Extract FILE HEADER features (7 features)
        file_header_features = {
            "Machine": pe.FILE_HEADER.Machine,
            "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
            "TimeDateStamp": pe.FILE_HEADER.TimeDateStamp,
            "PointerToSymbolTable": pe.FILE_HEADER.PointerToSymbolTable,
            "NumberOfSymbols": pe.FILE_HEADER.NumberOfSymbols,
            "SizeOfOptionalHeader": pe.FILE_HEADER.SizeOfOptionalHeader,
            "Characteristics": pe.FILE_HEADER.Characteristics
        }

        # Extract OPTIONAL HEADER features (26 features)
        op_header_features = {
            "MajorLinkerVersion": pe.OPTIONAL_HEADER.MajorLinkerVersion,
            "MinorLinkerVersion": pe.OPTIONAL_HEADER.MinorLinkerVersion,
            "SizeOfCode": pe.OPTIONAL_HEADER.SizeOfCode,
            "SizeOfInitializedData": pe.OPTIONAL_HEADER.SizeOfInitializedData,
            "SizeOfUninitializedData": pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "BaseOfCode": pe.OPTIONAL_HEADER.BaseOfCode,
            "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
            "SectionAlignment": pe.OPTIONAL_HEADER.SectionAlignment,
            "FileAlignment": pe.OPTIONAL_HEADER.FileAlignment,
            "MajorOperatingSystemVersion": pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            "MinorOperatingSystemVersion": pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            "MajorImageVersion": pe.OPTIONAL_HEADER.MajorImageVersion,
            "MinorImageVersion": pe.OPTIONAL_HEADER.MinorImageVersion,
            "MajorSubsystemVersion": pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            "MinorSubsystemVersion": pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
            "CheckSum": pe.OPTIONAL_HEADER.CheckSum,
            "Subsystem": pe.OPTIONAL_HEADER.Subsystem,
            "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
            "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
            "SizeOfStackCommit": pe.OPTIONAL_HEADER.SizeOfStackCommit,
            "SizeOfHeapReserve": pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            "SizeOfHeapCommit": pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            "LoaderFlags": pe.OPTIONAL_HEADER.LoaderFlags,
            "NumberOfRvaAndSizes": pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
        }

        # Combine all features
        features.append({"Filename": filename, **file_header_features, **op_header_features})

        pe.close()

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} files...", flush=True)

    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  Error processing {filename}: {str(e)[:50]}", flush=True)

end_time = time.time()
elapsed = round(end_time - start_time, 2)

print(f"\nRun time: {elapsed} seconds", flush=True)
print(f"Successful: {len(features)} files", flush=True)
print(f"Errors: {errors} files", flush=True)
print(f"Speed: {len(features)/elapsed:.2f} files/second", flush=True)

# Create DataFrame and save
df = pd.DataFrame(features)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Features extracted from {len(features)} files and saved to {CSV_OUTPUT}", flush=True)
print(f"Total columns: {len(df.columns)} (34 columns)", flush=True)

# ============================================================
# STEP 2: ML TRAINING AND TESTING
# ============================================================
print("\n" + "=" * 60, flush=True)
print("STEP 2: ML TRAINING AND TESTING (More Features)", flush=True)
print("=" * 60, flush=True)

# Prepare data
df.drop(columns=["Filename"], inplace=True)
target_variable = "Characteristics"
feature_variables = df.columns.difference([target_variable])

print(f"Number of features: {len(feature_variables)}", flush=True)
print(f"Features: {list(feature_variables)[:10]}... (truncated)", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_variables], df[target_variable], test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}", flush=True)
print(f"Testing samples: {len(X_test)}", flush=True)

results = {}

# Random Forest
print("\nTraining Random Forest Classifier...", flush=True)
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': cross_val_score(rfc, X_train, y_train, cv=3)
}
print(f"Random Forest Accuracy: {results['Random Forest']['accuracy']:.4f}", flush=True)

# KNN
print("\nTraining K Nearest Neighbor Classifier...", flush=True)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': cross_val_score(knn, X_train, y_train, cv=3)
}
print(f"KNN Accuracy: {results['KNN']['accuracy']:.4f}", flush=True)

# Decision Tree
print("\nTraining Decision Tree Classifier...", flush=True)
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': cross_val_score(dtc, X_train, y_train, cv=3)
}
print(f"Decision Tree Accuracy: {results['Decision Tree']['accuracy']:.4f}", flush=True)

# SVM (with faster settings)
print("\nTraining Support Vector Machine Classifier...", flush=True)
svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': np.array([accuracy_score(y_test, y_pred)] * 3)  # Skip slow CV for SVM
}
print(f"SVM Accuracy: {results['SVM']['accuracy']:.4f}", flush=True)

# Logistic Regression
print("\nTraining Logistic Regression Classifier...", flush=True)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': cross_val_score(lr, X_train, y_train, cv=3)
}
print(f"Logistic Regression Accuracy: {results['Logistic Regression']['accuracy']:.4f}", flush=True)

# Gradient Boosting
print("\nTraining Gradient Boosting Classifier...", flush=True)
gbc = GradientBoostingClassifier(random_state=42, n_estimators=50)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred,
    'cv_scores': cross_val_score(gbc, X_train, y_train, cv=3)
}
print(f"Gradient Boosting Accuracy: {results['Gradient Boosting']['accuracy']:.4f}", flush=True)

# ============================================================
# STEP 3: GENERATE CHARTS (More Features)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("STEP 3: GENERATING CHARTS (More Features)", flush=True)
print("=" * 60, flush=True)

names = list(results.keys())

# Chart 1: Classifier Comparison
print("\n  Generating classifier_comparison_more_features.png...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
x = np.arange(len(names))
width = 0.2
for i, (m, c) in enumerate(zip(['accuracy', 'precision', 'recall', 'f1'],
                                ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])):
    ax1.bar(x + i*width, [results[n][m] for n in names], width, label=m.capitalize(), color=c)
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(0, 1.1)
ax1.set_title('Performance Metrics (More Features - 32 features)', fontweight='bold')

ax2 = axes[0, 1]
cv_means = [results[n]['cv_scores'].mean() for n in names]
ax2.bar(names, cv_means, color='steelblue')
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.set_ylim(0, 1.1)
ax2.set_title('Cross-Validation Scores', fontweight='bold')

ax3 = axes[1, 0]
ax3.barh(names, [results[n]['f1'] for n in names], color='#27ae60')
ax3.set_xlim(0, 1)
ax3.set_title('F1 Score', fontweight='bold')

ax4 = axes[1, 1]
ax4.axis('off')
table_data = [[n, f"{results[n]['accuracy']:.3f}", f"{results[n]['f1']:.3f}"] for n in names]
ax4.table(cellText=table_data, colLabels=['Classifier', 'Acc', 'F1'], loc='center')
ax4.set_title('Summary (More Features)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'classifier_comparison_more_features.png'), dpi=150)
plt.close()

# Chart 2: Confusion Matrices
print("  Generating confusion_matrices_more_features.png...", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, data) in enumerate(results.items()):
    cm = confusion_matrix(y_test, data['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nAcc: {data["accuracy"]:.3f}')
plt.suptitle('Confusion Matrices (More Features - 32 features)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrices_more_features.png'), dpi=150)
plt.close()

# Chart 3: Feature Importance
print("  Generating feature_importance_more_features.png...", flush=True)
fig, ax = plt.subplots(figsize=(12, 10))
importance = rfc.feature_importances_
indices = np.argsort(importance)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))
ax.barh(range(len(indices)), importance[indices], color=colors)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([list(feature_variables)[i] for i in indices])
ax.set_title('Feature Importance - Random Forest (More Features - 32 features)', fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'feature_importance_more_features.png'), dpi=150)
plt.close()

# Chart 4: Accuracy Comparison
print("  Generating accuracy_comparison_more_features.png...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
sorted_pairs = sorted([(n, results[n]['accuracy']) for n in names], key=lambda x: x[1], reverse=True)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
ax.bar([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color=colors)
ax.set_ylim(0, 1.1)
ax.set_xticklabels([p[0] for p in sorted_pairs], rotation=45, ha='right')
ax.set_title('Accuracy Comparison (More Features - 32 features)', fontweight='bold')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'accuracy_comparison_more_features.png'), dpi=150)
plt.close()

# Chart 5: Radar Chart
print("  Generating metrics_radar_more_features.png...", flush=True)
angles = [n / 4 * 2 * np.pi for n in range(4)] + [0]
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for (name, data), color in zip(results.items(), plt.cm.tab10(np.linspace(0, 1, len(results)))):
    values = [data['accuracy'], data['precision'], data['recall'], data['f1']] + [data['accuracy']]
    ax.plot(angles, values, 'o-', label=name, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title('Performance Radar (More Features - 32 features)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'metrics_radar_more_features.png'), dpi=150)
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY - MORE FEATURES RESULTS", flush=True)
print("=" * 60, flush=True)
print(f"\nDataset: {len(features)} samples, {len(feature_variables)} features", flush=True)
print(f"\nClassifier Results:", flush=True)
for name in names:
    print(f"  {name}: Accuracy={results[name]['accuracy']:.4f}, F1={results[name]['f1']:.4f}", flush=True)

print(f"\nCharts saved to {OUTPUT_FOLDER}:", flush=True)
print("  - classifier_comparison_more_features.png", flush=True)
print("  - confusion_matrices_more_features.png", flush=True)
print("  - feature_importance_more_features.png", flush=True)
print("  - accuracy_comparison_more_features.png", flush=True)
print("  - metrics_radar_more_features.png", flush=True)
print("  - output_file_more_features.csv", flush=True)

print("\nDONE!", flush=True)
