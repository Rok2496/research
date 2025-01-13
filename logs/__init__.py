import os
import json
import mlflow
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

class LogManager:
    def __init__(self):
        load_dotenv()
        self._load_config()
        self._setup_mlflow()
        self._setup_tensorboard()

    def _load_config(self):
        with open('logs/monitoring_config.json', 'r') as f:
            self.config = json.load(f)

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        experiment = mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
        
        if experiment is None:
            mlflow.create_experiment(
                self.config['mlflow']['experiment_name'],
                artifact_location=os.path.abspath(self.config['mlflow']['artifact_location'])
            )
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _setup_tensorboard(self):
        log_dir = os.path.join('logs', self.config['tensorboard']['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.config['tensorboard']['flush_secs']
        )