import unittest
import json
from deployment.serving.flask_app import app
from core.data.preprocessing import preprocess_image
import io
import numpy as np
from PIL import Image

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_health_check(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        
    def test_predict_endpoint(self):
        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_io = io.BytesIO()
        img.save(img_io, "PNG")
        img_io.seek(0)
        
        response = self.app.post(
            "/predict",
            content_type="multipart/form-data",
            data={"image": (img_io, "test.png")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 5)  # Number of classes
        
    def test_batch_predict_endpoint(self):
        # Create multiple dummy images
        images = []
        for _ in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_io = io.BytesIO()
            img.save(img_io, "PNG")
            img_io.seek(0)
            images.append(("images", (f"test_{_}.png", img_io, "image/png")))
            
        response = self.app.post(
            "/batch-predict",
            content_type="multipart/form-data",
            data=images
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("batch_results", data)
        self.assertEqual(len(data["batch_results"]), 3)

if __name__ == "__main__":
    unittest.main()
