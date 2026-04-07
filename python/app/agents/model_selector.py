import json
from app.schemas import DataProfile

class ModelSelectorAgent:
    def __init__(self, valkey, mlflow_client, mcp_client, settings):
        self.valkey = valkey
        self.mlflow = mlflow_client
        self.mcp = mcp_client
        self.settings = settings

    async def select(self, job):
        pass

    async def _llm_select(self, profile: DataProfile) -> list[str]:
        pass\n