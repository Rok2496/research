import torch
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

class MetricsComputer:
    @staticmethod
    def compute_auroc(outputs, targets):
        return roc_auc_score(targets.cpu(), outputs.cpu())
    
    @staticmethod
    def compute_f1(outputs, targets):
        return f1_score(targets.cpu(), outputs.cpu() > 0.5, average='weighted')