from pathlib import Path
import torch
import yaml

class ModelRegistry:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config = self._load_config()
        
    def _load_config(self):
        config_path = Path('../configs/model_config.yaml')
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def save_pretrained(self, model, name):
        path = self.base_path / 'pretrained' / f'{name}.pth'
        torch.save(model.state_dict(), path)
        
    def save_checkpoint(self, state, name):
        path = self.base_path / 'checkpoints' / f'{name}.pth'
        torch.save(state, path)
        
    def save_model(self, model, name):
        path = self.base_path / 'saved' / f'{name}.pth'
        torch.save(model.state_dict(), path)
        
    def load_pretrained(self, name):
        path = self.base_path / 'pretrained' / f'{name}.pth'
        return torch.load(path)
        
    def load_checkpoint(self, name):
        path = self.base_path / 'checkpoints' / f'{name}.pth'
        return torch.load(path)
        
    def load_model(self, name):
        path = self.base_path / 'saved' / f'{name}.pth'
        return torch.load(path)
