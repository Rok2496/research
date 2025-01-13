import mlflow
from pathlib import Path

class ModelTracker:
    def __init__(self):
        self.experiment_name = 'chexpert_training'
        mlflow.set_experiment(self.experiment_name)
        
    def log_model(self, model, metrics):
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, 'model')
            
    def load_best_model(self):
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['metrics.val_auroc DESC']
        )
        return mlflow.pytorch.load_model(runs[0].info.artifact_uri + '/model')
