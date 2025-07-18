import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Comprehensive model evaluation"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # AUC if probabilities are provided
        if y_pred_proba is not None:
            auc = roc_auc_score(y_true, y_pred_proba)
            results['AUC'] = auc
        
        self.results[model_name] = results
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("No models have been evaluated yet!")
            return None
        
        results_df = pd.DataFrame(list(self.results.values()))
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        return results_df
    
    def plot_model_comparison(self):
        """Plot model comparison"""
        results_df = self.compare_models()
        
        if results_df is None:
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                axes[i].bar(results_df['Model'], results_df[metric])
                axes[i].set_title(metric)
                axes[i].set_xlabel('Models')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()