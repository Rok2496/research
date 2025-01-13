import cv2
import numpy as np

class HistogramEqualization:
    def __init__(self):
        pass
    
    def apply(self, image):
        return cv2.equalizeHist(image)