import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class LIMEExplainer:
    def __init__(self, model, num_samples=1000):
        self.model = model
        self.num_samples = num_samples
        self.explainer = lime_image.LimeImageExplainer()
        
    def predict_fn(self, images):
        batch = torch.stack([torch.Tensor(img) for img in images])
        with torch.no_grad():
            logits = self.model(batch.cuda())
        return logits.cpu().numpy()
        
    def explain_prediction(self, image, top_labels=5):
        explanation = self.explainer.explain_instance(
            image.numpy().transpose(1,2,0),
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=self.num_samples
        )
        return explanation
        
    def visualize_explanation(self, image, explanation, label, save_path=None):
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=5,
            hide_rest=True
        )
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(image.permute(1,2,0))
        plt.title('Original Image')
        
        plt.subplot(1,2,2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f'LIME Explanation for Class {label}')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
    def get_feature_importance(self, explanation, label):
        return sorted(
            explanation.local_exp[label],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
    def analyze_multiple_samples(self, dataloader, num_samples=10):
        results = []
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image = images[0]
            explanation = self.explain_prediction(image)
            importance = self.get_feature_importance(
                explanation,
                labels[0].argmax().item()
            )
            results.append({
                'image_idx': i,
                'true_label': labels[0].argmax().item(),
                'importance': importance
            })
            
        return results
