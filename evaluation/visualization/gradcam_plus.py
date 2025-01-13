import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
            
        self.model.zero_grad()
        output = model_output[:, target_class]
        output.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        weights = F.adaptive_avg_pool2d(gradients, (1, 1))
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear', align_corners=False)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().numpy()
