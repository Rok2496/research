from flask import Flask, request, jsonify, render_template
from pathlib import Path
import torch
from core.models.classification import EfficientNetV2Classifier
from core.data.preprocessing import preprocess_image
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = Path("models/saved/model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = EfficientNetV2Classifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        image = request.files["image"]
        processed_image = preprocess_image(image)
        
        with torch.no_grad():
            output = model(processed_image.to(DEVICE))
            predictions = torch.sigmoid(output)
            
        return jsonify({
            "status": "success",
            "predictions": predictions.cpu().numpy().tolist()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
