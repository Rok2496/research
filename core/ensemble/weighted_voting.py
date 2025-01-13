import torch
import torch.nn as nn

class WeightedVoting(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else torch.ones(len(models))
        self.weights = nn.Parameter(self.weights)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        predictions = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred * self.weights[i])
        
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return self.softmax(ensemble_pred)
