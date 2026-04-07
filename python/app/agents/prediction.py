class PredictionAgent:
    def __init__(self, valkey, mlflow_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.settings = settings

    async def predict(self, job, model_version, model_cache):
        pass\n