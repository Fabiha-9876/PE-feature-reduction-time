#!/usr/bin/env python3
"""
This script uses the EXACT same code from Thesis Implementation.ipynb
Only the folder path is updated and chart generation is added at the end.
"""

import os
import pefile
import pandas as pd
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')


folder_path = "/Users/fabihajalal/Desktop/Undergrad Thesis/extracted"
OUTPUT_FOLDER = "/Users/fabihajalal/Desktop/Undergrad Thesis"

# Define the headers to extract
operational_header_features = ["AddressOfEntryPoint","ImageBase","SectionAlignment","FileAlignment","Subsystem","DllCharacteristics","MajorOperatingSystemVersion", "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion", "SizeOfImage", "SizeOfHeaders"]
file_header_features = ["Machine", "NumberOfSections", "Characteristics"]

# Initialize an empty list to store the extracted features for each file
features_list = []

# Loop through the executable files in the folder
print("="*60)
print("STEP 1: FEATURE EXTRACTION (Same as notebook)")
print("="*60)
print(f"Extracting features from: {folder_path}")

start_time1 = time.time()
file_count = 0
error_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".exe"):
        # Load the PE file using the pefile library
        file_path = os.path.join(folder_path, filename)
        try:
            pe = pefile.PE(file_path)

            # Extract the operational header features
            operational_header_values = [getattr(pe.OPTIONAL_HEADER, feature) for feature in operational_header_features]

            # Extract the file header features
            file_header_values = [getattr(pe.FILE_HEADER, feature) for feature in file_header_features]

            # Combine the extracted features into a single list for the file
            file_features = [filename] + operational_header_values + file_header_values

            # Append the file features to the overall list
            features_list.append(file_features)
            file_count += 1

            if file_count % 100 == 0:
                print(f"  Processed {file_count} files...")
        except Exception as e:
            error_count += 1
            continue

# Define the column names for the DataFrame
columns = ["Filename"] + operational_header_features + file_header_features

end_time1 = time.time()
time_needed = round(end_time1 - start_time1, 2)
print(f"\nRun time: {time_needed} seconds")
print(f"Successful: {file_count} files")
print(f"Errors: {error_count} files")
print(f"Speed: {file_count/time_needed:.2f} files/second")

# Create the DataFrame from the list of features
df = pd.DataFrame(features_list, columns=columns)

# Write the DataFrame to a CSV file
output_file_path = os.path.join(OUTPUT_FOLDER, "output_file_final.csv")
df.to_csv(output_file_path, index=False)

# Print confirmation message
print(f"Features extracted from {len(features_list)} files and saved to {output_file_path}")

print("\n" + "="*60)
print("STEP 2: ML TRAINING AND TESTING (Same as notebook)")
print("="*60)

# Load the dataset
df = pd.read_csv(output_file_path)

# Drop the filename column
df.drop(columns=["Filename"], inplace=True)

# Define the target variable and the feature variables
target_variable = "Characteristics"
feature_variables = df.columns.difference([target_variable])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_variables], df[target_variable], test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Store results for charts
results = {}

# Train and test a Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
f1_rfc = f1_score(y_test, y_pred_rfc, average='macro')
precision_rfc = precision_score(y_test, y_pred_rfc, average='macro')
recall_rfc = recall_score(y_test, y_pred_rfc, average='macro')
print("Random Forest Classifier Metrics:")
print("Accuracy:", accuracy_rfc)
print("F1 Score:", f1_rfc)
print("Precision Score:", precision_rfc)
print("Recall Score:", recall_rfc)

results['Random Forest'] = {
    'model': rfc,
    'accuracy': accuracy_rfc,
    'f1': f1_rfc,
    'precision': precision_rfc,
    'recall': recall_rfc,
    'y_pred': y_pred_rfc,
    'cv_scores': cross_val_score(rfc, X_train, y_train, cv=3)
}

# Train and test a K Nearest Neighbor Classifier
print("\nTraining K Nearest Neighbor Classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='macro')
precision_knn = precision_score(y_test, y_pred_knn, average='macro')
recall_knn = recall_score(y_test, y_pred_knn, average='macro')
print("K Nearest Neighbor Classifier Metrics:")
print("Accuracy:", accuracy_knn)
print("F1 Score:", f1_knn)
print("Precision Score:", precision_knn)
print("Recall Score:", recall_knn)

results['KNN'] = {
    'model': knn,
    'accuracy': accuracy_knn,
    'f1': f1_knn,
    'precision': precision_knn,
    'recall': recall_knn,
    'y_pred': y_pred_knn,
    'cv_scores': cross_val_score(knn, X_train, y_train, cv=3)
}

# Train and test a Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
f1_dtc = f1_score(y_test, y_pred_dtc, average='macro')
precision_dtc = precision_score(y_test, y_pred_dtc, average='macro')
recall_dtc = recall_score(y_test, y_pred_dtc, average='macro')
print("Decision Tree Classifier Metrics:")
print("Accuracy:", accuracy_dtc)
print("F1 Score:", f1_dtc)
print("Precision Score:", precision_dtc)
print("Recall Score:", recall_dtc)

results['Decision Tree'] = {
    'model': dtc,
    'accuracy': accuracy_dtc,
    'f1': f1_dtc,
    'precision': precision_dtc,
    'recall': recall_dtc,
    'y_pred': y_pred_dtc,
    'cv_scores': cross_val_score(dtc, X_train, y_train, cv=3)
}

# Train and test a Support Vector Machine Classifier
print("\nTraining Support Vector Machine Classifier...")
svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
print("Support Vector Machine Classifier Metrics:")
print("Accuracy:", accuracy_svm)
print("F1 Score:", f1_svm)
print("Precision Score:", precision_svm)
print("Recall Score:", recall_svm)

results['SVM'] = {
    'model': svm,
    'accuracy': accuracy_svm,
    'f1': f1_svm,
    'precision': precision_svm,
    'recall': recall_svm,
    'y_pred': y_pred_svm,
    'cv_scores': cross_val_score(svm, X_train, y_train, cv=3)
}

# Train and test a Logistic Regression Classifier
print("\nTraining Logistic Regression Classifier...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
precision_lr = precision_score(y_test, y_pred_lr, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')
print("Logistic Regression Classifier Metrics:")
print("Accuracy:", accuracy_lr)
print("F1 Score:", f1_lr)
print("Precision Score:", precision_lr)
print("Recall Score:", recall_lr)

results['Logistic Regression'] = {
    'model': lr,
    'accuracy': accuracy_lr,
    'f1': f1_lr,
    'precision': precision_lr,
    'recall': recall_lr,
    'y_pred': y_pred_lr,
    'cv_scores': cross_val_score(lr, X_train, y_train, cv=3)
}

# Train and test a Gradient Boosting Classifier
print("\nTraining Gradient Boosting Classifier...")
gbc = GradientBoostingClassifier(n_estimators=50, random_state=42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
f1_gbc = f1_score(y_test, y_pred_gbc, average='macro')
precision_gbc = precision_score(y_test, y_pred_gbc, average='macro')
recall_gbc = recall_score(y_test, y_pred_gbc, average='macro')
print("Gradient Boosting Classifier Metrics:")
print("Accuracy:", accuracy_gbc)
print("F1 Score:", f1_gbc)
print("Precision Score:", precision_gbc)
print("Recall Score:", recall_gbc)

results['Gradient Boosting'] = {
    'model': gbc,
    'accuracy': accuracy_gbc,
    'f1': f1_gbc,
    'precision': precision_gbc,
    'recall': recall_gbc,
    'y_pred': y_pred_gbc,
    'cv_scores': cross_val_score(gbc, X_train, y_train, cv=3)
}

# ============================================================================
# CHART GENERATION 
# ============================================================================
print("\n" + "="*60)
print("STEP 3: GENERATING CHARTS (New Addition)")
print("="*60)

# Chart 1: Classifier Performance Comparison
def plot_classifier_comparison(results, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())

    # 1. Bar chart - All metrics comparison
    ax1 = axes[0, 0]
    x = np.arange(len(names))
    width = 0.2

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[name][metric] for name in names]
        ax1.bar(x + i*width, values, width, label=metric.capitalize(), color=color)

    ax1.set_xlabel('Classifier', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Classifier Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Cross-validation scores with error bars
    ax2 = axes[0, 1]
    cv_means = [results[name]['cv_scores'].mean() for name in names]
    cv_stds = [results[name]['cv_scores'].std() for name in names]

    bars = ax2.bar(names, cv_means, yerr=cv_stds, capsize=5,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax2.set_xlabel('Classifier', fontsize=11)
    ax2.set_ylabel('CV Accuracy', fontsize=11)
    ax2.set_title('3-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, cv_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. F1 Score horizontal bar chart
    ax3 = axes[1, 0]
    f1_scores = [results[name]['f1'] for name in names]
    colors = ['#27ae60' if f >= 0.7 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_scores]

    bars = ax3.barh(names, f1_scores, color=colors)
    ax3.set_xlabel('F1 Score', fontsize=11)
    ax3.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, f1_scores):
        ax3.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for name in names:
        table_data.append([
            name,
            f"{results[name]['accuracy']:.3f}",
            f"{results[name]['precision']:.3f}",
            f"{results[name]['recall']:.3f}",
            f"{results[name]['f1']:.3f}",
            f"{results[name]['cv_scores'].mean():.3f}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'CV Mean'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Performance Summary Table', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'classifier_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Chart 2: Confusion Matrices
def plot_confusion_matrices(results, y_test, output_folder):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        if idx >= len(axes):
            break

        cm = confusion_matrix(y_test, data['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=True, square=True)
        axes[idx].set_title(f'{name}\nAccuracy: {data["accuracy"]:.3f}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)

    plt.suptitle('Confusion Matrices for All Classifiers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Chart 3: Feature Importance
def plot_feature_importance(rf_model, feature_names, output_folder):
    fig, ax = plt.subplots(figsize=(10, 8))

    importance = rf_model.feature_importances_
    indices = np.argsort(importance)

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))

    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Chart 4: Accuracy Comparison
def plot_accuracy_comparison(results, output_folder):
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]

    sorted_pairs = sorted(zip(names, accuracies), key=lambda x: x[1], reverse=True)
    names_sorted = [p[0] for p in sorted_pairs]
    acc_sorted = [p[1] for p in sorted_pairs]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names_sorted)))

    bars = ax.bar(names_sorted, acc_sorted, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Accuracy Comparison (Sorted)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, acc_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'accuracy_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Chart 5: Radar Chart
def plot_metrics_radar(results, output_folder):
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        values = [data['accuracy'], data['precision'], data['recall'], data['f1']]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Classifier Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'metrics_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# Generate all charts
plot_classifier_comparison(results, OUTPUT_FOLDER)
plot_confusion_matrices(results, y_test, OUTPUT_FOLDER)
plot_feature_importance(results['Random Forest']['model'], list(feature_variables), OUTPUT_FOLDER)
plot_accuracy_comparison(results, OUTPUT_FOLDER)
plot_metrics_radar(results, OUTPUT_FOLDER)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nExtraction Time: {time_needed} seconds")
print(f"Files Processed: {file_count}")
print(f"Speed: {file_count/time_needed:.2f} files/second")

print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
    "Classifier", "Accuracy", "Precision", "Recall", "F1 Score"))
print("-"*60)

for name, data in results.items():
    print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        name, data['accuracy'], data['precision'], data['recall'], data['f1']))

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print("-"*60)
print(f"\nBest Classifier: {best[0]} (Accuracy: {best[1]['accuracy']:.4f})")

print("\n" + "="*60)
print("CHARTS SAVED:")
print("="*60)
print(f"  - {OUTPUT_FOLDER}/classifier_comparison.png")
print(f"  - {OUTPUT_FOLDER}/confusion_matrices.png")
print(f"  - {OUTPUT_FOLDER}/feature_importance.png")
print(f"  - {OUTPUT_FOLDER}/accuracy_comparison.png")
print(f"  - {OUTPUT_FOLDER}/metrics_radar.png")
