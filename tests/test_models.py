import unittest
import torch
from core.models.classification import EfficientNetV2Classifier, DenseNet201Classifier, ViTClassifier
from core.models.segmentation import UNetPlusPlus, MaskRCNNDetector
from core.models.self_supervised import SimCLR, BYOL

class TestClassificationModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 3
        self.height = 224
        self.width = 224
        self.num_classes = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_efficientnet(self):
        model = EfficientNetV2Classifier(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output, attention = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsNotNone(attention)
        
    def test_densenet(self):
        model = DenseNet201Classifier(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
class TestSegmentationModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_unetplusplus(self):
        model = UNetPlusPlus().to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 1, self.height, self.width))
        
    def test_maskrcnn(self):
        model = MaskRCNNDetector().to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertIn("boxes", output[0])
        self.assertIn("masks", output[0])
        self.assertIn("scores", output[0])

if __name__ == "__main__":
    unittest.main()
