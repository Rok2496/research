import logging
import os
import json
from torch.utils.tensorboard import SummaryWriter
import mlflow
from dotenv import load_dotenv

class LogManager:
    def __init__(self):
        self.loggers = {}
        load_dotenv()
        self._load_config()
        self._setup_mlflow()
        self._setup_tensorboard()

    def _load_config(self):
        with open('logs/monitoring_config.json', 'r') as f:
            self.config = json.load(f)

    def _setup_mlflow(self):
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI must be set in .env")

        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config['mlflow']['experiment_name'],
                    artifact_location=os.getenv('MLFLOW_ARTIFACT_ROOT')
                )
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_id)
        except Exception as e:
            logger = logging.getLogger('deployment')
            logger.error(f"MLflow setup error: {e}")
            raise

    def _setup_tensorboard(self):
        log_dir = os.path.join('logs', self.config['tensorboard']['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=self.config['tensorboard']['flush_secs']
        )

    def setup_deployment_logger(self):
        logger = logging.getLogger('deployment')
        logger.setLevel(self.config['deployment']['log_level'])
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.loggers['deployment'] = logger
        return logger

    def log_metrics(self, metrics, step):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        mlflow.log_metrics(metrics, step=step)

    def close(self):
        self.writer.close()