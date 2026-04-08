from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "openai_compatible"
    llm_api_key: str = ""
    anthropic_api_key: str = ""  # legacy fallback
    llm_api_url: str = "https://api.openai.com/v1/chat/completions"
    llm_api_version: str = "2023-06-01"
    llm_auth_header: str = "Authorization"
    llm_auth_scheme: str = "Bearer"
    llm_extra_headers_json: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 256
    llm_timeout_seconds: float = 30.0
    valkey_url: str = "redis://localhost:6379"
    retrain_lock_ttl_seconds: int = 1800
    max_training_time_seconds: int = 3600
    result_ttl_seconds: int = 60
    no_drift_threshold: float = 0.2
    minor_drift_threshold: float = 0.5
    drift_check_every_n_predictions: int = 50
    drift_check_every_t_minutes: int = 10
    incremental_update_wait_seconds: int = 10
    watchdog_observation_window_seconds: int = 3600
    watchdog_regression_tolerance: float = 0.15
    min_history_length: int = 30
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_uri: str = "s3://sktime-agentic-models"
    class Config:
        env_file = ".env"