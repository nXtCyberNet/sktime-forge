from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://mlflow:5000')
print('experiments', [e.name for e in client.search_experiments()])
model_name = 'ts-forecaster-airline'
versions = client.search_model_versions("name='ts-forecaster-airline'")
print('versions', [{'version': v.version, 'run_id': v.run_id, 'status': v.status, 'source': v.source} for v in versions])
