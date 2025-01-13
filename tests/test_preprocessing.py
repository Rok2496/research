import unittest
import numpy as np
import torch
from core.data.preprocessing.augmentation import ElasticTransform, Mixup, CutMix
from core.data.preprocessing.normalization import HistogramEqualization, AdaptiveContrast
from core.data.preprocessing.enhancement import CLAHE, GammaCorrection

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.image = np.random.rand(224, 224, 3).astype(np.float32)
        self.batch = torch.randn(8, 3, 224, 224)
        self.labels = torch.randint(0, 5, (8,))
        
    def test_elastic_transform(self):
        transform = ElasticTransform()
        transformed = transform(self.image)
        self.assertEqual(transformed.shape, self.image.shape)
        
    def test_mixup(self):
        mixup = Mixup(alpha=0.2)
        mixed_x, y_a, y_b, lam = mixup(self.batch, self.labels)
        
        self.assertEqual(mixed_x.shape, self.batch.shape)
        self.assertTrue(0 <= lam <= 1)
        
class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
    def test_histogram_equalization(self):
        equalizer = HistogramEqualization()
        equalized = equalizer(self.image)
        
        self.assertEqual(equalized.shape, self.image.shape)
        self.assertTrue(np.min(equalized) >= 0)
        self.assertTrue(np.max(equalized) <= 255)
        
    def test_adaptive_contrast(self):
        enhancer = AdaptiveContrast()
        enhanced = enhancer(self.image)
        
        self.assertEqual(enhanced.shape, self.image.shape)
        
class TestEnhancement(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
    def test_clahe(self):
        clahe = CLAHE()
        enhanced = clahe(self.image)
        
        self.assertEqual(enhanced.shape, self.image.shape)
        self.assertTrue(np.min(enhanced) >= 0)
        self.assertTrue(np.max(enhanced) <= 255)

if __name__ == "__main__":
    unittest.main()
