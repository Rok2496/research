import torch
import numpy as np
from collections import defaultdict
from .logger import Logger

class MetricsTracker:
    def __init__(self, metrics_list):
        self.metrics = defaultdict(list)
        self.current_epoch = 0
        self.logger = Logger("MetricsTracker")
        
    def update(self, metrics_dict, phase="train"):
        for metric_name, value in metrics_dict.items():
            self.metrics[f"{phase}_{metric_name}"].append(value)
            
    def get_latest(self, metric_name, phase="train"):
        key = f"{phase}_{metric_name}"
        return self.metrics[key][-1] if self.metrics[key] else None
        
    def get_history(self, metric_name, phase="train"):
        return self.metrics[f"{phase}_{metric_name}"]
        
    def epoch_end(self):
        self.current_epoch += 1
        metrics_summary = {}
        
        for key, values in self.metrics.items():
            if values:
                metrics_summary[key] = values[-1]
                
        self.logger.info(f"Epoch {self.current_epoch} Summary:")
        for key, value in metrics_summary.items():
            self.logger.info(f"{key}: {value:.4f}")
            
    def save_metrics(self, save_path):
        torch.save(self.metrics, save_path)
        
    def load_metrics(self, load_path):
        self.metrics = torch.load(load_path)
