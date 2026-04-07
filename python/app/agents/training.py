class TrainingAgent:
    def __init__(self, valkey, mlflow_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.settings = settings

    async def handle_retrain_job(self, job):
        pass\n