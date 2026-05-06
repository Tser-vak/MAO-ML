import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import roc_curve, auc

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

    def log_roc_curve(self, y_true, y_proba):
        """Calculates, plots, logs, and cleans up the ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal guessing line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Wasted Lab Time)')
        plt.ylabel('True Positive Rate (Correct Actives)')
        plt.title(f"ROC Curve: {self.title_suffix}")
        plt.legend(loc="lower right")
        
        filename = f"roc_curve_{self.file_suffix}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Log to MLflow and clean up
        mlflow.log_artifact(filename)
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Logged {filename} to MLflow.")