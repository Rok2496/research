import torch
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

class ModelProfiler:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def profile_inference(self, input_shape=(1, 3, 224, 224), num_runs=100):
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        # Profile
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)
                times.append(time.time() - start)

        self.metrics["inference_time"] = {
            "mean": np.mean(times),
            "std": np.std(times),
            "p95": np.percentile(times, 95)
        }

    def profile_memory(self, input_shape=(1, 3, 224, 224)):
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        torch.cuda.reset_peak_memory_stats()
        _ = self.model(dummy_input)
        
        self.metrics["memory"] = {
            "allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "reserved": torch.cuda.max_memory_reserved() / 1024**2
        }

    def profile_flops(self, input_shape=(1, 3, 224, 224)):
        from thop import profile as thop_profile
        
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        macs, params = thop_profile(self.model, inputs=(dummy_input,))
        
        self.metrics["compute"] = {
            "macs": macs / 1e9,  # GMACs
            "params": params / 1e6  # M
        }

    def get_summary(self):
        return self.metrics
