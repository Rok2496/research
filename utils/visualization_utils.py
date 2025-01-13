import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path

class VisualizationUtils:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_training_history(self, metrics_tracker, metric_name):
        plt.figure(figsize=(10, 5))
        
        train_values = metrics_tracker.get_history(metric_name, "train")
        val_values = metrics_tracker.get_history(metric_name, "val")
        
        plt.plot(train_values, label=f"Train {metric_name}")
        plt.plot(val_values, label=f"Validation {metric_name}")
        
        plt.title(f"{metric_name} Over Time")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        
        plt.savefig(self.save_dir / f"{metric_name}_history.png")
        plt.close()
        
    def plot_confusion_matrix(self, true_labels, predictions, class_names):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        plt.savefig(self.save_dir / "confusion_matrix.png")
        plt.close()
        
    def plot_attention_maps(self, image, attention_maps, save_name):
        num_maps = len(attention_maps)
        fig, axes = plt.subplots(1, num_maps + 1, figsize=(5 * (num_maps + 1), 5))
        
        # Original image
        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Attention maps
        for idx, (name, attention) in enumerate(attention_maps.items(), 1):
            axes[idx].imshow(attention.cpu(), cmap="viridis")
            axes[idx].set_title(name)
            axes[idx].axis("off")
            
        plt.savefig(self.save_dir / f"{save_name}.png")
        plt.close()
        
    def plot_roc_curves(self, true_labels, predictions, class_names):
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_labels[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
            
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        
        plt.savefig(self.save_dir / "roc_curves.png")
        plt.close()
