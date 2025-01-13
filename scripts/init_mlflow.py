import os
import mlflow
from dotenv import load_dotenv

def init_mlflow():
    load_dotenv()
    
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    artifact_uri = os.getenv('MLFLOW_ARTIFACT_ROOT')
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "chexpert_training"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_uri
        )
    mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    init_mlflow()