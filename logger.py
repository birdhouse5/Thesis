# logger.py
import mlflow
import mlflow.pytorch

class MLflowLogger:
    def __init__(self, experiment_name: str = "default", run_name: str = None):
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_param(self, key: str, value):
        mlflow.log_param(key, value)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metric(self, key: str, value, step: int = None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, file_path: str, artifact_path: str = None):
        mlflow.log_artifact(file_path, artifact_path)

    def end_run(self):
        mlflow.end_run()
