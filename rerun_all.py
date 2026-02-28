#!/usr/bin/env python3
"""
Re-run ML training + chart generation for BOTH feature sets using existing CSVs.
Uses the fixed, consistent settings: RBF SVM, 3-fold CV, 50 GB estimators.
"""
import os
import sys
import pandas as pd
import numpy as np

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

BASE_DIR = "/Users/fabihajalal/Desktop/PE feature reduction time"


def train_all_classifiers(X_train, X_test, y_train, y_test):
    """Train all 6 classifiers with consistent settings matching the paper."""
    results = {}

    # 1. Random Forest (100 estimators)
    print("  Training Random Forest...", flush=True)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    results['Random Forest'] = {
        'model': rfc,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(rfc, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['Random Forest']['accuracy']:.4f}", flush=True)

    # 2. KNN (k=5)
    print("  Training KNN...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    results['KNN'] = {
        'model': knn,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(knn, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['KNN']['accuracy']:.4f}", flush=True)

    # 3. Decision Tree
    print("  Training Decision Tree...", flush=True)
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    results['Decision Tree'] = {
        'model': dtc,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(dtc, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['Decision Tree']['accuracy']:.4f}", flush=True)

    # 4. SVM (RBF kernel, gamma='scale')
    print("  Training SVM...", flush=True)
    svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results['SVM'] = {
        'model': svm,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(svm, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['SVM']['accuracy']:.4f}", flush=True)

    # 5. Logistic Regression
    print("  Training Logistic Regression...", flush=True)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(lr, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['Logistic Regression']['accuracy']:.4f}", flush=True)

    # 6. Gradient Boosting (50 estimators)
    print("  Training Gradient Boosting...", flush=True)
    gbc = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    results['Gradient Boosting'] = {
        'model': gbc,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
        'cv_scores': cross_val_score(gbc, X_train, y_train, cv=3)
    }
    print(f"    Accuracy: {results['Gradient Boosting']['accuracy']:.4f}", flush=True)

    return results


def generate_charts(results, y_test, feature_variables, output_folder, suffix="", label=""):
    """Generate all 5 charts for a feature set."""
    names = list(results.keys())
    title_label = f" ({label})" if label else ""

    # Chart 1: Classifier Comparison (4-panel)
    print(f"  Generating classifier_comparison{suffix}.png...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    x = np.arange(len(names))
    width = 0.2
    for i, (m, c) in enumerate(zip(['accuracy', 'precision', 'recall', 'f1'],
                                    ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])):
        ax1.bar(x + i*width, [results[n][m] for n in names], width, label=m.capitalize(), color=c)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.set_title(f'Classifier Performance Metrics{title_label}', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    cv_means = [results[n]['cv_scores'].mean() for n in names]
    cv_stds = [results[n]['cv_scores'].std() for n in names]
    bars = ax2.bar(names, cv_means, yerr=cv_stds, capsize=5,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f'3-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, cv_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    ax3 = axes[1, 0]
    f1_scores = [results[n]['f1'] for n in names]
    colors = ['#27ae60' if f >= 0.7 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_scores]
    bars = ax3.barh(names, f1_scores, color=colors)
    ax3.set_xlim(0, 1)
    ax3.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    for bar, score in zip(bars, f1_scores):
        ax3.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [[n, f"{results[n]['accuracy']:.3f}", f"{results[n]['precision']:.3f}",
                   f"{results[n]['recall']:.3f}", f"{results[n]['f1']:.3f}",
                   f"{results[n]['cv_scores'].mean():.3f}"] for n in names]
    table = ax4.table(cellText=table_data,
                      colLabels=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'CV Mean'],
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax4.set_title(f'Performance Summary{title_label}', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'classifier_comparison{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 2: Confusion Matrices
    print(f"  Generating confusion_matrices{suffix}.png...", flush=True)
    fig, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
    axes_cm = axes_cm.flatten()
    for idx, (name, data) in enumerate(results.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=True, square=True)
        axes_cm[idx].set_title(f'{name}\nAccuracy: {data["accuracy"]:.3f}', fontsize=11, fontweight='bold')
        axes_cm[idx].set_xlabel('Predicted', fontsize=10)
        axes_cm[idx].set_ylabel('Actual', fontsize=10)
    plt.suptitle(f'Confusion Matrices for All Classifiers{title_label}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'confusion_matrices{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 3: Feature Importance
    print(f"  Generating feature_importance{suffix}.png...", flush=True)
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_variables) * 0.3)))
    importance = results['Random Forest']['model'].feature_importances_
    indices = np.argsort(importance)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([list(feature_variables)[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Feature Importance - Random Forest{title_label}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'feature_importance{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 4: Accuracy Comparison
    print(f"  Generating accuracy_comparison{suffix}.png...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_pairs = sorted([(n, results[n]['accuracy']) for n in names], key=lambda x: x[1], reverse=True)
    names_sorted = [p[0] for p in sorted_pairs]
    acc_sorted = [p[1] for p in sorted_pairs]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names_sorted)))
    bars = ax.bar(names_sorted, acc_sorted, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Classifier Accuracy Comparison (Sorted){title_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, acc_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'accuracy_comparison{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 5: Radar Chart
    print(f"  Generating metrics_radar{suffix}.png...", flush=True)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for (name, data), color in zip(results.items(), plt.cm.tab10(np.linspace(0, 1, len(results)))):
        values = [data['accuracy'], data['precision'], data['recall'], data['f1']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Classifier Performance Radar Chart{title_label}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'metrics_radar{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_results_table(results, label):
    """Print formatted results table for paper reference."""
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE FOR PAPER - {label}")
    print(f"{'='*70}")
    print(f"\n{'Classifier':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 62)
    for name, data in results.items():
        print(f"{name:<22} {data['accuracy']*100:>9.2f}% {data['f1']:>10.4f} {data['precision']:>10.4f} {data['recall']:>10.4f}")
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 62)
    print(f"Best: {best[0]} (Accuracy: {best[1]['accuracy']*100:.2f}%)")


# ================================================================
# RUN 1: REDUCED FEATURE SET (15 features)
# ================================================================
print("=" * 70)
print("REDUCED FEATURE SET (15 features)")
print("=" * 70)

df_reduced = pd.read_csv(os.path.join(BASE_DIR, "output_file_final.csv"))
print(f"Loaded {len(df_reduced)} samples, {len(df_reduced.columns)} columns")

df_reduced.drop(columns=["Filename"], inplace=True)
target_variable = "Characteristics"
feature_variables_reduced = df_reduced.columns.difference([target_variable])
print(f"Features: {len(feature_variables_reduced)}, Target: {target_variable}")

X_train, X_test, y_train, y_test = train_test_split(
    df_reduced[feature_variables_reduced], df_reduced[target_variable],
    test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

results_reduced = train_all_classifiers(X_train, X_test, y_train, y_test)
print_results_table(results_reduced, "Reduced Feature Set (15 features)")
generate_charts(results_reduced, y_test, feature_variables_reduced, BASE_DIR,
                suffix="", label="15 Features")

# ================================================================
# RUN 2: EXTENDED FEATURE SET (32 features)
# ================================================================
print("\n" + "=" * 70)
print("EXTENDED FEATURE SET (32 features)")
print("=" * 70)

df_extended = pd.read_csv(os.path.join(BASE_DIR, "output_file_more_features.csv"))
print(f"Loaded {len(df_extended)} samples, {len(df_extended.columns)} columns")

df_extended.drop(columns=["Filename"], inplace=True)
target_variable = "Characteristics"
feature_variables_extended = df_extended.columns.difference([target_variable])
print(f"Features: {len(feature_variables_extended)}, Target: {target_variable}")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df_extended[feature_variables_extended], df_extended[target_variable],
    test_size=0.2, random_state=42)
print(f"Train: {len(X_train2)}, Test: {len(X_test2)}")

results_extended = train_all_classifiers(X_train2, X_test2, y_train2, y_test2)
print_results_table(results_extended, "Extended Feature Set (32 features)")
generate_charts(results_extended, y_test2, feature_variables_extended, BASE_DIR,
                suffix="_more_features", label="32 Features")

# ================================================================
# COMPARISON TABLE
# ================================================================
print("\n" + "=" * 70)
print("ACCURACY COMPARISON (for Paper Table IV)")
print("=" * 70)
print(f"\n{'Classifier':<22} {'Reduced':>10} {'Extended':>10} {'Improvement':>12}")
print("-" * 56)
for name in results_reduced:
    r = results_reduced[name]['accuracy'] * 100
    e = results_extended[name]['accuracy'] * 100
    diff = e - r
    print(f"{name:<22} {r:>9.2f}% {e:>9.2f}% {diff:>+11.2f}%")

# ================================================================
# FEATURE IMPORTANCE TABLE (for Paper Table V)
# ================================================================
print("\n" + "=" * 70)
print("TOP 10 FEATURE IMPORTANCE - Extended Set (for Paper Table V)")
print("=" * 70)
rf_model = results_extended['Random Forest']['model']
importance = rf_model.feature_importances_
feat_names = list(feature_variables_extended)
sorted_idx = np.argsort(importance)[::-1]
print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':>12}")
print("-" * 48)
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"{rank:<6} {feat_names[idx]:<30} {importance[idx]:>12.4f}")

print("\n" + "=" * 70)
print("ALL CHARTS REGENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nCharts saved to: {BASE_DIR}")
print("  Reduced set:  classifier_comparison.png, confusion_matrices.png,")
print("                feature_importance.png, accuracy_comparison.png, metrics_radar.png")
print("  Extended set: classifier_comparison_more_features.png, confusion_matrices_more_features.png,")
print("                feature_importance_more_features.png, accuracy_comparison_more_features.png,")
print("                metrics_radar_more_features.png")
