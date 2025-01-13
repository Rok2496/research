import torch
import torch.nn as nn

class StackingClassifier(nn.Module):
    def __init__(self, base_models, meta_model):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        
    def forward(self, x):
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():
                pred = model(x)
                base_preds.append(pred)
        
        meta_features = torch.cat(base_preds, dim=1)
        return self.meta_model(meta_features)

    def train_base_models(self, train_loader, criterion, optimizer):
        for model in self.base_models:
            model.train()
            for batch in train_loader:
                images, labels = batch
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
