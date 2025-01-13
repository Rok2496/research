import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

class ClassificationMetrics:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, preds, targets):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
    def compute_metrics(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            "auroc": self._compute_auroc(predictions, targets),
            "average_precision": self._compute_average_precision(predictions, targets),
            "accuracy": self._compute_accuracy(predictions, targets),
            "f1_score": self._compute_f1(predictions, targets)
        }
        return metrics
        
    def _compute_auroc(self, preds, targets):
        aurocs = []
        for i in range(self.num_classes):
            auroc = roc_auc_score(targets[:, i], preds[:, i])
            aurocs.append(auroc)
        return np.mean(aurocs)
    
    def _compute_average_precision(self, preds, targets):
        aps = []
        for i in range(self.num_classes):
            ap = average_precision_score(targets[:, i], preds[:, i])
            aps.append(ap)
        return np.mean(aps)
    
    def _compute_accuracy(self, preds, targets):
        predictions = (preds > 0.5).astype(np.float32)
        return np.mean((predictions == targets).astype(np.float32))
    
    def _compute_f1(self, preds, targets):
        predictions = (preds > 0.5).astype(np.float32)
        tp = np.sum(predictions * targets, axis=0)
        fp = np.sum(predictions * (1 - targets), axis=0)
        fn = np.sum((1 - predictions) * targets, axis=0)
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return np.mean(f1)
