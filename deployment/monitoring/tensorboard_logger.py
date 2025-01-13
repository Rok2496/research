import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

class TensorboardLogger:
    def __init__(self, log_dir="logs/tensorboard"):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_metrics(self, metrics, step, prefix=""):
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
            elif isinstance(value, np.ndarray):
                self.writer.add_histogram(f"{prefix}/{name}", value, step)

    def log_images(self, images, step, prefix=""):
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        self.writer.add_images(f"{prefix}/images", images, step)

    def log_model_graph(self, model, input_shape=(1, 3, 224, 224)):
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        self.writer.add_graph(model, dummy_input)

    def log_attention_maps(self, attention_maps, step):
        for name, attention in attention_maps.items():
            if isinstance(attention, torch.Tensor):
                attention = attention.cpu().numpy()
            self.writer.add_image(f"attention/{name}", 
                                attention, 
                                step, 
                                dataformats="HW")

    def close(self):
        self.writer.close()
