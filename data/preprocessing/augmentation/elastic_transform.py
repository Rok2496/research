# data/preprocessing/augmentation/elastic_transform.py
import numpy as np
import cv2

class ElasticTransform:
    def __init__(self, alpha=1000, sigma=30):
        self.alpha = alpha
        self.sigma = sigma
        
    def apply(self, image):
        shape = image.shape
        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        mapx = np.float32(x + self.alpha * dx)
        mapy = np.float32(y + self.alpha * dy)
        
        return cv2.remap(image, mapx, mapy, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT)