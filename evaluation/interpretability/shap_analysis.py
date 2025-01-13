import torch
import numpy as np
import shap
from ..metrics.classification_metrics import ClassificationMetrics

class ShapAnalyzer:
    def __init__(self, model, background_data=None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def initialize_explainer(self):
        def model_wrapper(x):
            return self.model(torch.FloatTensor(x).cuda()).cpu().detach().numpy()
            
        self.explainer = shap.DeepExplainer(
            model_wrapper,
            self.background_data
        )
        
    def explain_prediction(self, input_image, target_class=None):
        if self.explainer is None:
            self.initialize_explainer()
            
        shap_values = self.explainer.shap_values(input_image)
        
        if target_class is not None:
            return shap_values[target_class]
        return shap_values
        
    def generate_feature_importance(self, input_image):
        shap_values = self.explain_prediction(input_image)
        importance_scores = np.abs(shap_values).mean(0)
        
        return {
            "shap_values": shap_values,
            "importance_scores": importance_scores,
            "global_importance": importance_scores.mean(axis=(1,2))
        }
        
    def plot_feature_importance(self, input_image, save_path=None):
        shap_values = self.explain_prediction(input_image)
        
        shap.image_plot(
            shap_values,
            input_image,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
