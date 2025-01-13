import torch
from core.models.classification.densenet201 import DenseNet201
from core.models.classification.vit_large import ViTLarge
from core.models.classification.efficientnet_v2 import EfficientNetV2

class ModelRegistry:
    def __init__(self):
        self.models = {
            'efficientnet_v2': EfficientNetV2,
            'densenet201': DenseNet201,
            'vit_large': ViTLarge
        }

    def get_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        return self.models[model_name]()

    def compute_loss(self, output, target):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion(output, target)

    def evaluate(self, model, dataloader, metrics):
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for data, target in dataloader:
                output = model(data)
                # Implement metric computation based on the metrics list
                for metric in metrics:
                    if metric not in val_metrics:
                        val_metrics[metric] = 0
        return val_metrics

    def save_checkpoint(self, state, filename):
        torch.save(state, f"models/checkpoints/{filename}.pth")