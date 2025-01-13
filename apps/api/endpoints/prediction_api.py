from flask import Blueprint, request, jsonify
from ..services.inference_service import InferenceService

prediction_bp = Blueprint('prediction', __name__)
inference_service = InferenceService()

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.files['image']
        patient_id = request.form.get('patient_id')
        
        results = inference_service.run_inference(image_data, patient_id)
        return jsonify({
            'status': 'success',
            'predictions': results,
            'patient_id': patient_id
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@prediction_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        batch_data = request.files.getlist('images')
        results = inference_service.run_batch_inference(batch_data)
        return jsonify({
            'status': 'success',
            'batch_results': results
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
