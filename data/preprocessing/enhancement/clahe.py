import cv2
import numpy as np

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
    
    def apply(self, image):
        return self.clahe.apply(image)