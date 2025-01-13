import numpy as np
import cv2

def normalize_image(image):
    """Normalize image values to [0,1] range"""
    if isinstance(image, np.ndarray):
        return image.astype(np.float32) / 255.0
    return image / 255.0