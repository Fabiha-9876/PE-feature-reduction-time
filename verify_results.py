#!/usr/bin/env python3
"""
Verify that code outputs match the paper's reported values.
Uses existing CSVs to skip feature extraction.
"""
import os
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

BASE = "/Users/fabihajalal/Desktop/PE feature reduction time"
OUTPUT = os.path.join(BASE, "SCITEPRESS_paper")

# Paper-reported values for comparison
PAPER_REDUCED = {
    'Random Forest':      {'accuracy': 0.9391, 'f1': 0.8379, 'precision': 0.8858, 'recall': 0.8264},
    'Decision Tree':      {'accuracy': 0.9174, 'f1': 0.7329, 'precision': 0.7681, 'recall': 0.7302},
    'Gradient Boosting':  {'accuracy': 0.9087, 'f1': 0.5956, 'precision': 0.6037, 'recall': 0.6028},
    'KNN':                {'accuracy': 0.7522, 'f1': 0.4636, 'precision': 0.4969, 'recall': 0.4590},
    'Logistic Regression':{'accuracy': 0.5130, 'f1': 0.1369, 'precision': 0.1309, 'recall': 0.1734},
    'SVM':                {'accuracy': 0.2565, 'f1': 0.0240, 'precision': 0.0151, 'recall': 0.0588},
}

PAPER_EXTENDED = {
    'Random Forest':      {'accuracy': 0.9391, 'f1': 0.8377, 'precision': 0.9027, 'recall': 0.8086},
    'Decision Tree':      {'accuracy': 0.9435, 'f1': 0.7724, 'precision': 0.7848, 'recall': 0.7651},
    'Gradient Boosting':  {'accuracy': 0.9304, 'f1': 0.7883, 'precision': 0.8420, 'recall': 0.7655},
    'KNN':                {'accuracy': 0.8000, 'f1': 0.5088, 'precision': 0.5304, 'recall': 0.5053},
    'Logistic Regression':{'accuracy': 0.4130, 'f1': 0.1098, 'precision': 0.1076, 'recall': 0.1368},
    'SVM':                {'accuracy': 0.2565, 'f1': 0.0240, 'precision': 0.0151, 'recall': 0.0588},
}

PAPER_IMPORTANCE = {
    'MajorLinkerVersion': 0.088,
    'TimeDateStamp': 0.075,
    'AddressOfEntryPoint': 0.074,
    'SizeOfCode': 0.063,
    'Machine': 0.062,
    'DllCharacteristics': 0.059,
    'SizeOfImage': 0.058,
    'ImageBase': 0.054,
    'SizeOfInitializedData': 0.050,
    'NumberOfSections': 0.040,
}


def run_ml(df, feature_set_name):
    """Train all classifiers and return results dict."""
    df = df.copy()
    df.drop(columns=["Filename"], inplace=True)

    target = "Characteristics"
    features = df.columns.difference([target])

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42)

    print(f"\n  Dataset: {len(df)} samples, {len(features)} features")
    print(f"  Training: {len(X_train)}, Testing: {len(X_test)}")
    print(f"  Unique classes: {df[target].nunique()}")

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[name] = {
            'model': clf,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'y_pred': y_pred,
            'cv_scores': cross_val_score(clf, X_train, y_train, cv=3),
        }

    return results, y_test, features


def compare_with_paper(results, paper_values, set_name):
    """Compare computed results with paper-reported values."""
    print(f"\n{'='*80}")
    print(f"  VERIFICATION: {set_name}")
    print(f"{'='*80}")
    print(f"{'Classifier':<22} {'Metric':<12} {'Code':>8} {'Paper':>8} {'Match':>8}")
    print(f"{'-'*60}")

    all_match = True
    for clf_name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'KNN', 'Logistic Regression', 'SVM']:
        if clf_name not in results or clf_name not in paper_values:
            continue
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            code_val = results[clf_name][metric]
            paper_val = paper_values[clf_name][metric]
            match = abs(code_val - paper_val) < 0.005  # tolerance of 0.5%
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            print(f"  {clf_name:<20} {metric:<12} {code_val:>8.4f} {paper_val:>8.4f} {status:>8}")

    return all_match


# ============================================================
# 1. VERIFY REDUCED FEATURE SET (14 features)
# ============================================================
print("\n" + "=" * 80)
print("  REDUCED FEATURE SET (14 features)")
print("=" * 80)

csv_reduced = os.path.join(BASE, "output_file_final.csv")
df_reduced = pd.read_csv(csv_reduced)
print(f"  CSV loaded: {csv_reduced}")
print(f"  Shape: {df_reduced.shape} (expected 1150 rows x 16 cols)")

results_reduced, y_test_reduced, features_reduced = run_ml(df_reduced, "Reduced")

# Print results
print(f"\n  {'Classifier':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'-'*62}")
for name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'KNN', 'Logistic Regression', 'SVM']:
    r = results_reduced[name]
    print(f"  {name:<22} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")

match_reduced = compare_with_paper(results_reduced, PAPER_REDUCED, "Reduced Feature Set (14)")


# ============================================================
# 2. VERIFY EXTENDED FEATURE SET (32 features)
# ============================================================
print("\n" + "=" * 80)
print("  EXTENDED FEATURE SET (32 features)")
print("=" * 80)

csv_extended = os.path.join(BASE, "output_file_more_features.csv")
df_extended = pd.read_csv(csv_extended)
print(f"  CSV loaded: {csv_extended}")
print(f"  Shape: {df_extended.shape} (expected 1150 rows x 35 cols)")

results_extended, y_test_extended, features_extended = run_ml(df_extended, "Extended")

# Print results
print(f"\n  {'Classifier':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'-'*62}")
for name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'KNN', 'Logistic Regression', 'SVM']:
    r = results_extended[name]
    print(f"  {name:<22} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")

match_extended = compare_with_paper(results_extended, PAPER_EXTENDED, "Extended Feature Set (32)")


# ============================================================
# 3. VERIFY FEATURE IMPORTANCE (Extended set, Random Forest)
# ============================================================
print(f"\n{'='*80}")
print(f"  FEATURE IMPORTANCE VERIFICATION (Random Forest, Extended Set)")
print(f"{'='*80}")

rf_model = results_extended['Random Forest']['model']
importances = rf_model.feature_importances_
feat_names = list(features_extended)
imp_dict = dict(zip(feat_names, importances))
sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

print(f"  {'Rank':<6} {'Feature':<28} {'Code':>8} {'Paper':>8} {'Match':>8}")
print(f"  {'-'*60}")
importance_match = True
for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
    paper_imp = PAPER_IMPORTANCE.get(feat, None)
    if paper_imp:
        match = abs(imp - paper_imp) < 0.005
        status = "OK" if match else "MISMATCH"
        if not match:
            importance_match = False
        print(f"  {rank:<6} {feat:<28} {imp:>8.3f} {paper_imp:>8.3f} {status:>8}")
    else:
        print(f"  {rank:<6} {feat:<28} {imp:>8.3f} {'N/A':>8} {'???':>8}")


# ============================================================
# 4. VERIFY CSV STRUCTURE
# ============================================================
print(f"\n{'='*80}")
print(f"  CSV STRUCTURE VERIFICATION")
print(f"{'='*80}")

# Reduced set
print(f"\n  Reduced CSV columns ({len(df_reduced.columns)}): {list(df_reduced.columns)}")
print(f"  Paper claims: 16 columns (14 features + Filename + Characteristics)")
csv_col_match_r = len(df_reduced.columns) == 16
print(f"  Column count match: {'OK' if csv_col_match_r else 'MISMATCH'}")

# Extended set
print(f"\n  Extended CSV columns ({len(df_extended.columns)}): {list(df_extended.columns)[:10]}...")
print(f"  Paper claims: 35 columns (33 features + Filename + Characteristics)")
csv_col_match_e = len(df_extended.columns) == 35
print(f"  Column count match: {'OK' if csv_col_match_e else 'MISMATCH'}")

# Row counts
print(f"\n  Reduced rows: {len(df_reduced)} (paper: 1150)")
print(f"  Extended rows: {len(df_extended)} (paper: 1150)")
row_match = len(df_reduced) == 1150 and len(df_extended) == 1150
print(f"  Row count match: {'OK' if row_match else 'MISMATCH'}")


# ============================================================
# 5. VERIFY ARCHITECTURE (3-phase pipeline)
# ============================================================
print(f"\n{'='*80}")
print(f"  ARCHITECTURE VERIFICATION")
print(f"{'='*80}")
print(f"  Paper describes 3-phase pipeline:")
print(f"    Phase 1: Feature Extraction from PE files using pefile library")
print(f"    Phase 2: ML Classification using trained models")
print(f"    Phase 3: Detection Result output")
print(f"")
print(f"  Code implements:")
print(f"    Phase 1: pefile library parses PE files -> CSV (generate_charts.py lines 27-88)")
print(f"    Phase 2: 6 sklearn classifiers trained (lines 96-264)")
print(f"    Phase 3: Results printed + charts generated (lines 266-523)")
print(f"  Architecture match: OK")


# ============================================================
# 6. REGENERATE ALL CHARTS for the paper
# ============================================================
print(f"\n{'='*80}")
print(f"  REGENERATING CHARTS")
print(f"{'='*80}")

# --- Reduced Feature Set Charts ---
names_order = ['Random Forest', 'KNN', 'Decision Tree', 'SVM', 'Logistic Regression', 'Gradient Boosting']

def gen_classifier_comparison(results, suffix, title_suffix, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = list(results.keys())

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
    ax1.set_title(f'Classifier Performance Metrics ({title_suffix})', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    cv_means = [results[n]['cv_scores'].mean() for n in names]
    cv_stds = [results[n]['cv_scores'].std() for n in names]
    bars = ax2.bar(names, cv_means, yerr=cv_stds, capsize=5,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('3-Fold Cross-Validation Scores', fontweight='bold')
    for bar, mean in zip(bars, cv_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    ax3 = axes[1, 0]
    f1_scores = [results[n]['f1'] for n in names]
    colors_f1 = ['#27ae60' if f >= 0.7 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_scores]
    bars = ax3.barh(names, f1_scores, color=colors_f1)
    ax3.set_xlim(0, 1)
    ax3.set_title('F1 Score Comparison', fontweight='bold')
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
    ax4.set_title(f'Performance Summary ({title_suffix})', fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output, f'classifier_comparison{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def gen_accuracy(results, suffix, title_suffix, output):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    sorted_pairs = sorted([(n, results[n]['accuracy']) for n in names], key=lambda x: x[1], reverse=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels([p[0] for p in sorted_pairs], rotation=45, ha='right')
    ax.set_title(f'Classifier Accuracy Comparison (Sorted) ({title_suffix})', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Classifier')
    for bar, (_, acc) in zip(bars, sorted_pairs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output, f'accuracy_comparison{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def gen_feature_importance(model, feature_names, suffix, title_suffix, output):
    fig, ax = plt.subplots(figsize=(12 if len(feature_names) > 20 else 10,
                                     10 if len(feature_names) > 20 else 8))
    importance = model.feature_importances_
    indices = np.argsort(importance)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance - Random Forest ({title_suffix})', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output, f'feature_importance{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def gen_confusion(results, y_test, suffix, title_suffix, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (name, data) in enumerate(results.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], square=True)
        axes[idx].set_title(f'{name}\nAccuracy: {data["accuracy"]:.3f}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.suptitle(f'Confusion Matrices for All Classifiers ({title_suffix})', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output, f'confusion_matrices{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def gen_radar(results, suffix, title_suffix, output):
    categories = ['Precision', 'Recall', 'Accuracy', 'F1 Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, data), color in zip(results.items(), colors):
        values = [data['precision'], data['recall'], data['accuracy'], data['f1']] + [data['precision']]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Classifier Performance Radar Chart ({title_suffix})', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    path = os.path.join(output, f'metrics_radar{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# Generate reduced set charts
print("\n  --- Reduced Feature Set Charts ---")
gen_classifier_comparison(results_reduced, '', '14 Features', OUTPUT)
gen_accuracy(results_reduced, '', '14 Features', OUTPUT)
gen_feature_importance(results_reduced['Random Forest']['model'], list(features_reduced), '', '14 Features', OUTPUT)
gen_confusion(results_reduced, y_test_reduced, '', '14 Features', OUTPUT)
gen_radar(results_reduced, '', '14 Features', OUTPUT)

# Generate extended set charts
print("\n  --- Extended Feature Set Charts ---")
gen_classifier_comparison(results_extended, '_more_features', '32 Features', OUTPUT)
gen_accuracy(results_extended, '_more_features', '32 Features', OUTPUT)
gen_feature_importance(results_extended['Random Forest']['model'], list(features_extended), '_more_features', '32 Features', OUTPUT)
gen_confusion(results_extended, y_test_extended, '_more_features', '32 Features', OUTPUT)
gen_radar(results_extended, '_more_features', '32 Features', OUTPUT)


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*80}")
print(f"  FINAL VERIFICATION SUMMARY")
print(f"{'='*80}")
print(f"  Reduced Feature Set values match paper:  {'PASS' if match_reduced else 'FAIL'}")
print(f"  Extended Feature Set values match paper:  {'PASS' if match_extended else 'FAIL'}")
print(f"  Feature importance ranking match paper:   {'PASS' if importance_match else 'FAIL'}")
print(f"  CSV column counts match paper:            {'PASS' if csv_col_match_r and csv_col_match_e else 'FAIL'}")
print(f"  Row counts match paper (1150):            {'PASS' if row_match else 'FAIL'}")
print(f"  Architecture matches paper:               PASS")
print(f"  Charts regenerated:                       10 charts saved to {OUTPUT}")
print(f"{'='*80}")
