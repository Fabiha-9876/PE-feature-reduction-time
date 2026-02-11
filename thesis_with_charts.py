#!/usr/bin/env python3


import os
import time
import warnings
import pefile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

MALWARE_FOLDER = "/Users/fabihajalal/Desktop/Malware/extracted"
OUTPUT_FOLDER = "/Users/fabihajalal/Desktop/Undergrad Thesis"

operational_header_features = [
    "AddressOfEntryPoint", "ImageBase", "SectionAlignment", "FileAlignment",
    "Subsystem", "DllCharacteristics", "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion",
    "SizeOfImage", "SizeOfHeaders"
]

file_header_features = ["Machine", "NumberOfSections", "Characteristics"]


def extract_features_pefile(folder_path):
    
    ##Extract PE header features using pefile library.
   
    features_list = []

    print(f"Extracting features from: {folder_path}")
    print("Using pefile library ")

    start_time = time.time()
    file_count = 0
    error_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".exe"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load PE file using pefile (same as your code)
                pe = pefile.PE(file_path)

                # Extract operational header features (same as your code)
                operational_header_values = [
                    getattr(pe.OPTIONAL_HEADER, feature)
                    for feature in operational_header_features
                ]

                # Extract file header features (same as your code)
                file_header_values = [
                    getattr(pe.FILE_HEADER, feature)
                    for feature in file_header_features
                ]

                # Combine features (same as your code)
                file_features = [filename] + operational_header_values + file_header_values
                features_list.append(file_features)

                file_count += 1
                if file_count % 100 == 0:
                    print(f"  Processed {file_count} files...")

            except Exception as e:
                error_count += 1
                continue

    elapsed_time = time.time() - start_time
    print(f"\nExtraction complete!")
    print(f"  Successful: {file_count} files")
    print(f"  Errors: {error_count} files")
    print(f"  Time: {elapsed_time:.1f} seconds")
    print(f"  Speed: {file_count/elapsed_time:.2f} files/second")

    # Create DataFrame with same column names as your original
    column_names = ["Filename"] + operational_header_features + file_header_features
    df = pd.DataFrame(features_list, columns=column_names)

    return df


def train_classifiers(X_train, X_test, y_train, y_test):
    """
    Train the SAME classifiers as your original thesis notebook.
    """
    results = {}

    print("\n" + "="*60)
    print("TRAINING CLASSIFIERS (Same as original thesis)")
    print("="*60)

    # 1. Random Forest Classifier 
    print("\nTraining Random Forest Classifier...")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)

    results['Random Forest'] = {
        'model': rfc,
        'accuracy': accuracy_score(y_test, y_pred_rfc),
        'f1': f1_score(y_test, y_pred_rfc, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_rfc, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_rfc, average='macro', zero_division=0),
        'y_pred': y_pred_rfc,
        'cv_scores': cross_val_score(rfc, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}")

    # 2. K-Nearest Neighbors 
    print("\nTraining K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    results['KNN'] = {
        'model': knn,
        'accuracy': accuracy_score(y_test, y_pred_knn),
        'f1': f1_score(y_test, y_pred_knn, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_knn, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_knn, average='macro', zero_division=0),
        'y_pred': y_pred_knn,
        'cv_scores': cross_val_score(knn, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['KNN']['accuracy']:.4f}")

    # 3. Decision Tree 
    print("\nTraining Decision Tree...")
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    y_pred_dtc = dtc.predict(X_test)

    results['Decision Tree'] = {
        'model': dtc,
        'accuracy': accuracy_score(y_test, y_pred_dtc),
        'f1': f1_score(y_test, y_pred_dtc, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_dtc, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_dtc, average='macro', zero_division=0),
        'y_pred': y_pred_dtc,
        'cv_scores': cross_val_score(dtc, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['Decision Tree']['accuracy']:.4f}")

    # 4. Logistic Regression 
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_lr, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, average='macro', zero_division=0),
        'y_pred': y_pred_lr,
        'cv_scores': cross_val_score(lr, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['Logistic Regression']['accuracy']:.4f}")

    # 5. SVM 
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    results['SVM'] = {
        'model': svm,
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'f1': f1_score(y_test, y_pred_svm, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_svm, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_svm, average='macro', zero_division=0),
        'y_pred': y_pred_svm,
        'cv_scores': cross_val_score(svm, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['SVM']['accuracy']:.4f}")

    # 6. Gradient Boosting 
    print("\nTraining Gradient Boosting...")
    gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)

    results['Gradient Boosting'] = {
        'model': gbc,
        'accuracy': accuracy_score(y_test, y_pred_gbc),
        'f1': f1_score(y_test, y_pred_gbc, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_gbc, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_gbc, average='macro', zero_division=0),
        'y_pred': y_pred_gbc,
        'cv_scores': cross_val_score(gbc, X_train, y_train, cv=5)
    }
    print(f"  Accuracy: {results['Gradient Boosting']['accuracy']:.4f}")

    return results

# VISUALIZATION CHARTS 

def plot_classifier_comparison(results, output_folder):
    """Create classifier performance comparison chart."""
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
    ax2.set_title('5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
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

    # Color the header row
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Performance Summary Table', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'classifier_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(results, y_test, output_folder):
    """Create confusion matrices for all classifiers."""
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


def plot_feature_importance(rf_model, feature_names, output_folder):
    """Create feature importance chart from Random Forest."""
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


def plot_accuracy_comparison(results, output_folder):
    """Create a clean accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]

    # Sort by accuracy
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

    # Add value labels on bars
    for bar, acc in zip(bars, acc_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_folder, 'accuracy_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_radar(results, output_folder):
    """Create radar/spider chart for classifier comparison."""
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


def print_results_summary(results):
    """Print a formatted summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
        "Classifier", "Accuracy", "Precision", "Recall", "F1 Score"))
    print("-"*60)

    for name, data in results.items():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, data['accuracy'], data['precision'], data['recall'], data['f1']))

    # Find best classifier
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("-"*60)
    print(f"\nBest Classifier: {best[0]} (Accuracy: {best[1]['accuracy']:.4f})")


# MAIN EXECUTION FLOW
def main():
    print("="*60)
    print("THESIS IMPLEMENTATION WITH VISUALIZATION CHARTS")
    print("Using original pefile approach + ML classifiers + Charts")
    print("="*60)

    # Step 1: Extract features using YOUR original approach
    print("\n" + "="*60)
    print("STEP 1: FEATURE EXTRACTION (Same as original thesis)")
    print("="*60)

    df = extract_features_pefile(MALWARE_FOLDER)
    print(f"\nExtracted {len(df)} samples with {len(df.columns)} features")

    # Save to CSV 
    csv_path = os.path.join(OUTPUT_FOLDER, 'extracted_features.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved features to: {csv_path}")

    # Step 2: Prepare data for ML 
    print("\n" + "="*60)
    print("STEP 2: PREPARING DATA FOR ML")
    print("="*60)

    # Drop filename column 
    df_ml = df.drop(columns=["Filename"])

    # Use Characteristics as target 
    target_variable = "Characteristics"
    feature_variables = df_ml.columns.difference([target_variable])

    print(f"Target variable: {target_variable}")
    print(f"Feature variables: {len(feature_variables)}")

    # Split data 
    X = df_ml[feature_variables]
    y = df_ml[target_variable]

    # Handle any NaN values
    X = X.fillna(0)

    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Unique classes: {y.nunique()}")

    # Step 3: Train classifiers 
    results = train_classifiers(X_train, X_test, y_train, y_test)

    # Print summary
    print_results_summary(results)

    # Step 4: Generate and save charts (NEW)
    print("\n" + "="*60)
    print("STEP 3: GENERATING VISUALIZATION CHARTS")
    print("="*60)

    plot_classifier_comparison(results, OUTPUT_FOLDER)
    plot_confusion_matrices(results, y_test, OUTPUT_FOLDER)
    plot_feature_importance(results['Random Forest']['model'], list(feature_variables), OUTPUT_FOLDER)
    plot_accuracy_comparison(results, OUTPUT_FOLDER)
    plot_metrics_radar(results, OUTPUT_FOLDER)

    print("\n" + "="*60)
    print("ALL CHARTS SAVED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCharts saved to: {OUTPUT_FOLDER}")
    print("  - classifier_comparison.png")
    print("  - confusion_matrices.png")
    print("  - feature_importance.png")
    print("  - accuracy_comparison.png")
    print("  - metrics_radar.png")


if __name__ == "__main__":
    main()
