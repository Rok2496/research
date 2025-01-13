from flask import Blueprint, request, jsonify
from ..services.inference_service import InferenceService

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/stats', methods=['GET'])
def get_analysis_stats():
    try:
        return jsonify({
            'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'values': [85, 87, 89, 91, 92],
            'total_analyses': 500,
            'success_rate': 0.95
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@analysis_bp.route('/patient/<patient_id>', methods=['GET'])
def get_patient_analysis(patient_id):
    try:
        return jsonify({
            'patient_id': patient_id,
            'analyses': [
                {
                    'date': '2023-01-01',
                    'diagnosis': 'Pneumonia',
                    'confidence': 0.95,
                    'regions': ['lower_right_lung']
                }
            ]
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
