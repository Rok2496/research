# data/preprocessing/enhancement/gamma_correction.py
import cv2
import numpy as np

class GammaCorrection:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        
    def apply(self, image):
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)