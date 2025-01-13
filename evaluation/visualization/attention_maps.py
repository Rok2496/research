import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        
    def register_hooks(self):
        def get_attention_map(name):
            def hook(module, input, output):
                self.attention_maps[name] = output.detach()
            return hook
            
        for name, module in self.model.named_modules():
            if "attention" in name.lower():
                module.register_forward_hook(get_attention_map(name))
                
    def visualize_attention(self, input_image):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_image)
            
        attention_visualizations = {}
        for name, attention in self.attention_maps.items():
            # Average attention weights across heads
            if len(attention.shape) == 4:  # B, H, N, N
                attention = attention.mean(1)
                
            # Reshape and normalize
            B, N, _ = attention.shape
            W = H = int(np.sqrt(N))
            attention = attention.reshape(B, W, H)
            
            # Normalize for visualization
            attention = (attention - attention.min()) / (attention.max() - attention.min())
            
            attention_visualizations[name] = attention
            
        return attention_visualizations
        
    def plot_attention_maps(self, attention_maps, save_path=None):
        num_maps = len(attention_maps)
        fig, axes = plt.subplots(1, num_maps, figsize=(5*num_maps, 5))
        
        for idx, (name, attention) in enumerate(attention_maps.items()):
            if num_maps == 1:
                ax = axes
            else:
                ax = axes[idx]
                
            im = ax.imshow(attention[0].cpu(), cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax)
            
        if save_path:
            plt.savefig(save_path)
        plt.close()
