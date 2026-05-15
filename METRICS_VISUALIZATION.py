#!/usr/bin/env python3
"""
Plant Disease Detection - Standard Metrics Visualization
Comprehensive visualization for accuracy, precision, recall, and F1-score

Based on plant-diseases-detection-cnn-97-acc.ipynb evaluation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_metrics_visualization(model, test_dataset, class_names):
    """
    Create comprehensive visualization of all standard metrics
    """
    print("🔍 Generating Comprehensive Metrics Visualization...")
    
    # Get predictions and true labels
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Metrics Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    overall_metrics = {
        'Accuracy': report['accuracy'],
        'Macro Avg Precision': report['macro avg']['precision'],
        'Macro Avg Recall': report['macro avg']['recall'],
        'Macro Avg F1-Score': report['macro avg']['f1-score']
    }
    
    bars = ax1.bar(overall_metrics.keys(), overall_metrics.values(), 
                   color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_metrics.values()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Per-Class Precision
    ax2 = plt.subplot(2, 3, 2)
    precision_scores = [report[f'{class_name}']['precision'] for class_name in class_names]
    create_per_class_metric_chart(ax2, precision_scores, class_names, 'Precision', '#3498db')
    
    # 3. Per-Class Recall
    ax3 = plt.subplot(2, 3, 3)
    recall_scores = [report[f'{class_name}']['recall'] for class_name in class_names]
    create_per_class_metric_chart(ax3, recall_scores, class_names, 'Recall', '#2ecc71')
    
    # 4. Per-Class F1-Score
    ax4 = plt.subplot(2, 3, 4)
    f1_scores = [report[f'{class_name}']['f1-score'] for class_name in class_names]
    create_per_class_metric_chart(ax4, f1_scores, class_names, 'F1-Score', '#9b59b6')
    
    # 5. Class Distribution (Support)
    ax5 = plt.subplot(2, 3, 5)
    support_scores = [report[f'{class_name}']['support'] for class_name in class_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    ax5.bar(range(len(class_names)), support_scores, color=colors)
    ax5.set_title('Class Distribution (Support)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Class Index', fontsize=10)
    ax5.set_ylabel('Number of Samples', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Confusion Matrix (Sample)
    ax6 = plt.subplot(2, 3, 6)
    create_sample_confusion_matrix(ax6, cm, class_names)
    
    plt.tight_layout()
    plt.savefig('comprehensive_metrics_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return report, cm

def create_per_class_metric_chart(ax, scores, class_names, metric_name, color):
    """Create per-class metric visualization"""
    bars = ax.bar(range(len(class_names)), scores, color=color, alpha=0.7)
    ax.set_title(f'Per-Class {metric_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class Index', fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Highlight best and worst performers
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    bars[best_idx].set_color('#2ecc71')  # Green for best
    bars[worst_idx].set_color('#e74c3c')  # Red for worst

def create_sample_confusion_matrix(ax, cm, class_names):
    """Create a sample confusion matrix visualization"""
    # Show only first 10 classes for clarity
    sample_size = min(10, len(class_names))
    sample_cm = cm[:sample_size, :sample_size]
    sample_classes = class_names[:sample_size]
    
    sns.heatmap(sample_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[c.split('___')[-1] for c in sample_classes],
                yticklabels=[c.split('___')[-1] for c in sample_classes])
    ax.set_title('Confusion Matrix (Sample: First 10 Classes)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

def create_detailed_classification_report(report, class_names):
    """Create detailed classification report visualization"""
    print("\n📊 DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    
    # Create DataFrame for better visualization
    metrics_data = []
    for class_name in class_names:
        metrics_data.append({
            'Class': class_name.split('___')[-1],
            'Precision': report[f'{class_name}']['precision'],
            'Recall': report[f'{class_name}']['recall'],
            'F1-Score': report[f'{class_name}']['f1-score'],
            'Support': report[f'{class_name}']['support']
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 3)
    
    print(df.to_string(index=False))
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Average Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Weighted Average Recall: {report['weighted avg']['recall']:.4f}")
    print(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return df

def create_precision_recall_curve(model, test_dataset, class_names):
    """Create Precision-Recall curves for multi-class classification"""
    print("\n📈 Creating Precision-Recall Curves...")
    
    # Get predictions and true labels
    y_pred_probs = model.predict(test_dataset)
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    # Create PR curves for sample classes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    sample_indices = [0, 10, 20, 30]  # Sample 4 classes
    sample_indices = [i for i in sample_indices if i < len(class_names)]
    
    for idx, ax in enumerate(sample_indices):
        if idx < len(class_names):
            # Binary PR curve for this class
            y_true_binary = (y_true == idx).astype(int)
            y_scores = y_pred_probs[:, idx]
            
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            avg_precision = average_precision_score(y_true_binary, y_scores)
            
            ax.plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}')
            ax.fill_between(recall, precision, alpha=0.2)
            ax.set_title(f'{class_names[idx].split("___")[-1]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_roc_curves(model, test_dataset, class_names):
    """Create ROC curves for multi-class classification"""
    print("\n📈 Creating ROC Curves...")
    
    # Get predictions and true labels
    y_pred_probs = model.predict(test_dataset)
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    # Create ROC curves for sample classes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    sample_indices = [0, 10, 20, 30]  # Sample 4 classes
    sample_indices = [i for i in sample_indices if i < len(class_names)]
    
    for idx, ax in enumerate(sample_indices):
        if idx < len(class_names):
            from sklearn.metrics import roc_curve, auc
            
            # Binary ROC curve for this class
            y_true_binary = (y_true == idx).astype(int)
            y_scores = y_pred_probs[:, idx]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_title(f'{class_names[idx].split("___")[-1]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_learning_curves(training_history):
    """Create comprehensive learning curves visualization"""
    print("\n📈 Creating Learning Curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy curves
    epochs = range(1, len(training_history['accuracy']) + 1)
    ax1.plot(epochs, training_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Loss curves
    ax2.plot(epochs, training_history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_class_performance_analysis(report, class_names):
    """Create detailed class performance analysis"""
    print("\n📊 Creating Class Performance Analysis...")
    
    # Extract metrics
    precision_scores = [report[f'{class_name}']['precision'] for class_name in class_names]
    recall_scores = [report[f'{class_name}']['recall'] for class_name in class_names]
    f1_scores = [report[f'{class_name}']['f1-score'] for class_name in class_names]
    support_scores = [report[f'{class_name}']['support'] for class_name in class_names]
    
    # Create performance DataFrame
    performance_df = pd.DataFrame({
        'Class': [c.split('___')[-1] for c in class_names],
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Support': support_scores
    })
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance by class (sorted by F1-score)
    sorted_df = performance_df.sort_values('F1-Score', ascending=True)
    ax1.barh(range(len(sorted_df)), sorted_df['F1-Score'], color='skyblue')
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df['Class'], fontsize=8)
    ax1.set_xlabel('F1-Score', fontsize=12)
    ax1.set_title('Class Performance (F1-Score)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Precision vs Recall scatter
    ax2.scatter(performance_df['Precision'], performance_df['Recall'], 
               c=performance_df['F1-Score'], cmap='viridis', s=100, alpha=0.7)
    ax2.set_xlabel('Precision', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Precision vs Recall', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Support distribution
    ax3.bar(range(len(performance_df)), performance_df['Support'], color='lightcoral')
    ax3.set_xlabel('Class Index', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Class Support Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Metrics correlation heatmap
    metrics_corr = performance_df[['Precision', 'Recall', 'F1-Score']].corr()
    sns.heatmap(metrics_corr, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Metrics Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return performance_df

def create_error_analysis(cm, class_names):
    """Create detailed error analysis"""
    print("\n🔍 Creating Error Analysis...")
    
    # Calculate per-class errors
    per_class_errors = []
    for i in range(len(class_names)):
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - true_positives
        false_negatives = np.sum(cm[i, :]) - true_positives
        
        error_rate = (false_positives + false_negatives) / np.sum(cm[i, :])
        
        per_class_errors.append({
            'Class': class_names[i].split('___')[-1],
            'True Positives': true_positives,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
            'Error Rate': error_rate
        })
    
    # Create error analysis DataFrame
    error_df = pd.DataFrame(per_class_errors)
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Error rates by class
    sorted_errors = error_df.sort_values('Error Rate', ascending=True)
    ax1.barh(range(len(sorted_errors)), sorted_errors['Error Rate'], color='salmon')
    ax1.set_yticks(range(len(sorted_errors)))
    ax1.set_yticklabels(sorted_errors['Class'], fontsize=8)
    ax1.set_xlabel('Error Rate', fontsize=12)
    ax1.set_title('Error Rate by Class', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Confusion patterns
    ax2.scatter(error_df['False Positives'], error_df['False Negatives'], 
               c=error_df['Error Rate'], cmap='Reds', s=100, alpha=0.7)
    ax2.set_xlabel('False Positives', fontsize=12)
    ax2.set_ylabel('False Negatives', fontsize=12)
    ax2.set_title('Error Patterns', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # True positives distribution
    ax3.bar(range(len(error_df)), error_df['True Positives'], color='lightgreen')
    ax3.set_xlabel('Class Index', fontsize=12)
    ax3.set_ylabel('True Positives', fontsize=12)
    ax3.set_title('True Positives by Class', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Error breakdown
    error_types = ['False Positives', 'False Negatives']
    error_values = [error_df['False Positives'].sum(), error_df['False Negatives'].sum()]
    ax4.pie(error_values, labels=error_types, autopct='%1.1f%%', colors=['lightcoral', 'orange'])
    ax4.set_title('Error Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return error_df

def main():
    """
    Main function to generate all visualizations
    Based on plant-diseases-detection-cnn-97-acc.ipynb structure
    """
    print("🚀 Starting Comprehensive Metrics Visualization")
    print("=" * 60)
    
    # Load the trained model (from plant-diseases-detection-cnn-97-acc.ipynb)
    try:
        model = tf.keras.models.load_model("model_working_copy.keras")
        print("✅ Model loaded successfully")
    except:
        print("❌ Model not found. Please ensure 'model_working_copy.keras' exists.")
        return
    
    # Load test dataset (same as notebook)
    try:
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            'valid',
            labels="inferred",
            label_mode="categorical",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(128, 128),
            shuffle=False,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        print("✅ Test dataset loaded successfully")
    except:
        print("❌ Test dataset not found. Please ensure 'valid' directory exists.")
        return
    
    # Get class names
    class_names = test_dataset.class_names
    print(f"✅ Found {len(class_names)} classes")
    
    # Generate comprehensive metrics visualization
    report, cm = create_comprehensive_metrics_visualization(model, test_dataset, class_names)
    
    # Generate detailed classification report
    df = create_detailed_classification_report(report, class_names)
    
    # Generate class performance analysis
    performance_df = create_class_performance_analysis(report, class_names)
    
    # Generate error analysis
    error_df = create_error_analysis(cm, class_names)
    
    # Generate additional curves if needed
    if input("Generate PR curves? (y/n): ").lower() == 'y':
        create_precision_recall_curve(model, test_dataset, class_names)
    
    if input("Generate ROC curves? (y/n): ").lower() == 'y':
        create_roc_curves(model, test_dataset, class_names)
    
    # Load training history if available
    try:
        import json
        with open("training_hist.json", "r") as f:
            training_history = json.load(f)
        create_learning_curves(training_history)
    except:
        print("⚠️ Training history not found. Skipping learning curves.")
    
    print("\n🎉 All visualizations completed!")
    print("Files saved:")
    print("- comprehensive_metrics_visualization.png")
    print("- class_performance_analysis.png")
    print("- error_analysis.png")
    print("- precision_recall_curves.png (if generated)")
    print("- roc_curves.png (if generated)")
    print("- learning_curves.png (if generated)")

if __name__ == "__main__":
    main()
