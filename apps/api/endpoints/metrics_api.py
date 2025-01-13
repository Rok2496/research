from flask import Blueprint, jsonify
from ..services.inference_service import InferenceService

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/model-performance', methods=['GET'])
def get_model_performance():
    try:
        return jsonify({
            'accuracy': 0.92,
            'precision': 0.94,
            'recall': 0.91,
            'f1_score': 0.925,
            'confusion_matrix': [
                [100, 5],
                [3, 97]
            ]
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@metrics_bp.route('/daily-stats', methods=['GET'])
def get_daily_stats():
    try:
        return jsonify({
            'total_predictions': 150,
            'average_confidence': 0.89,
            'processing_time': 0.45
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
