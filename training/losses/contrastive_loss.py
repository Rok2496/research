import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        Calculates NT-Xent loss for self-supervised learning
        """
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        N = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        
        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)
        
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 2)
        negative_samples = sim[torch.arange(N).repeat(2)].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        
        loss = F.cross_entropy(logits, labels)
        return loss
