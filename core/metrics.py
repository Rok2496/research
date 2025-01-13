# core/metrics.py
import torch

def compute_metrics(model, dataloader, metric_names):
    model.eval()
    metrics = {metric: 0.0 for metric in metric_names}
    
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            
            # Compute metrics based on output and target
            for metric in metric_names:
                if metric == 'accuracy':
                    pred = (output > 0.5).float()
                    metrics[metric] += (pred == target).float().mean().item()
                    
    # Average metrics
    for metric in metrics:
        metrics[metric] /= len(dataloader)
        
    return metrics