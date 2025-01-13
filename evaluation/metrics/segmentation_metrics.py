import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        
    def update(self, preds, targets):
        dice = self._compute_dice(preds, targets)
        iou = self._compute_iou(preds, targets)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        
    def _compute_dice(self, preds, targets):
        smooth = 1e-7
        preds = torch.sigmoid(preds) > 0.5
        intersection = torch.sum(preds * targets, dim=(2,3))
        union = torch.sum(preds, dim=(2,3)) + torch.sum(targets, dim=(2,3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean(dim=0)
    
    def _compute_iou(self, preds, targets):
        smooth = 1e-7
        preds = torch.sigmoid(preds) > 0.5
        intersection = torch.sum(preds * targets, dim=(2,3))
        union = torch.sum(preds + targets, dim=(2,3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean(dim=0)
        
    def compute_metrics(self):
        dice_scores = torch.stack(self.dice_scores)
        iou_scores = torch.stack(self.iou_scores)
        
        return {
            "dice_score": dice_scores.mean().item(),
            "iou_score": iou_scores.mean().item()
        }
