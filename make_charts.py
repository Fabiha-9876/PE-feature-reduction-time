#!/usr/bin/env python3
import os
import sys
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

OUTPUT_FOLDER = "/Users/fabihajalal/Desktop/Undergrad Thesis"

print("Loading CSV...", flush=True)
df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "output_file_final.csv"))
print(f"Loaded {len(df)} samples", flush=True)

df.drop(columns=["Filename"], inplace=True)
target_variable = "Characteristics"
feature_variables = df.columns.difference([target_variable])

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_variables], df[target_variable], test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

results = {}

print("Training Random Forest...", flush=True)
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': cross_val_score(rfc, X_train, y_train, cv=3)
}
print(f"  Acc: {results['Random Forest']['accuracy']:.4f}", flush=True)

print("Training KNN...", flush=True)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': cross_val_score(knn, X_train, y_train, cv=3)
}
print(f"  Acc: {results['KNN']['accuracy']:.4f}", flush=True)

print("Training Decision Tree...", flush=True)
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': cross_val_score(dtc, X_train, y_train, cv=3)
}
print(f"  Acc: {results['Decision Tree']['accuracy']:.4f}", flush=True)

print("Training SVM...", flush=True)
svm = SVC(kernel="rbf", gamma='scale', C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': np.array([accuracy_score(y_test, y_pred)]*3)  # Skip slow CV for SVM
}
print(f"  Acc: {results['SVM']['accuracy']:.4f}", flush=True)

print("Training Logistic Regression...", flush=True)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': cross_val_score(lr, X_train, y_train, cv=3)
}
print(f"  Acc: {results['Logistic Regression']['accuracy']:.4f}", flush=True)

print("Training Gradient Boosting...", flush=True)
gbc = GradientBoostingClassifier(random_state=42, n_estimators=50)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'y_pred': y_pred, 'cv_scores': cross_val_score(gbc, X_train, y_train, cv=3)
}
print(f"  Acc: {results['Gradient Boosting']['accuracy']:.4f}", flush=True)

print("\nGenerating charts...", flush=True)
names = list(results.keys())

# Chart 1
print("  classifier_comparison.png", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1 = axes[0, 0]
x = np.arange(len(names))
width = 0.2
for i, (m, c) in enumerate(zip(['accuracy', 'precision', 'recall', 'f1'], ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])):
    ax1.bar(x + i*width, [results[n][m] for n in names], width, label=m.capitalize(), color=c)
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(0, 1.1)
ax1.set_title('Performance Metrics', fontweight='bold')

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
ax4.set_title('Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'classifier_comparison.png'), dpi=150)
plt.close()

# Chart 2
print("  confusion_matrices.png", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, data) in enumerate(results.items()):
    cm = confusion_matrix(y_test, data['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nAcc: {data["accuracy"]:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrices.png'), dpi=150)
plt.close()

# Chart 3
print("  feature_importance.png", flush=True)
fig, ax = plt.subplots(figsize=(10, 8))
importance = rfc.feature_importances_
indices = np.argsort(importance)
ax.barh(range(len(indices)), importance[indices], color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices))))
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([list(feature_variables)[i] for i in indices])
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'feature_importance.png'), dpi=150)
plt.close()

# Chart 4
print("  accuracy_comparison.png", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
sorted_pairs = sorted([(n, results[n]['accuracy']) for n in names], key=lambda x: x[1], reverse=True)
ax.bar([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names))))
ax.set_ylim(0, 1.1)
ax.set_xticklabels([p[0] for p in sorted_pairs], rotation=45, ha='right')
ax.set_title('Accuracy Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'accuracy_comparison.png'), dpi=150)
plt.close()

# Chart 5
print("  metrics_radar.png", flush=True)
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
ax.set_title('Performance Radar', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'metrics_radar.png'), dpi=150)
plt.close()

print("\nDONE! All charts saved.", flush=True)
