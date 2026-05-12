import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import precision_recall_curve, average_precision_score

class ModelVisualizer:
    """Handles the generation and MLflow logging of all model visualizations."""
    
    def __init__(self, model_name, use_smote):
        self.model_name = model_name
        self.strategy = "SMOTE" if use_smote else "No-SMOTE"
        
        # Format strings for titles and filenames
        self.title_suffix = f"{self.model_name} ({self.strategy})"
        self.file_suffix = f"{self.model_name}_{self.strategy}".replace(" ", "_").replace("-", "_")

    def log_confusion_matrix(self, cm):
        """Generates, logs, and cleans up a seaborn confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Inactive (Pred)", "Active (Pred)"],
                    yticklabels=["Inactive (Actual)", "Active (Actual)"])
        
        plt.title(f"Confusion Matrix: {self.title_suffix}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filename = f"confusion_matrix_{self.file_suffix}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close() # Free memory
        
        # Log to MLflow and clean up
        mlflow.log_artifact(filename)
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Logged {filename} to MLflow.")

    def log_pr_curve(self, y_true, y_proba):
        """Calculates, plots, logs, and cleans up the Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Calculate baseline (fraction of positives)
        baseline = sum(y_true) / len(y_true)
        plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision (Positive Predictive Value)')
        plt.title(f"Precision-Recall Curve: {self.title_suffix}")
        plt.legend(loc="lower right")
        
        filename = f"pr_curve_{self.file_suffix}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Log to MLflow and clean up
        mlflow.log_artifact(filename)
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Logged {filename} to MLflow.")