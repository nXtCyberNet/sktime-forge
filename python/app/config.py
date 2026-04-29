from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "openai_compatible"
    llm_api_key: str = ""
    anthropic_api_key: str = ""
    llm_api_url: str = "https://api.openai.com/v1/chat/completions"
    llm_api_version: str = "2023-06-01"
    llm_auth_header: str = "Authorization"
    llm_auth_scheme: str = "Bearer"
    llm_extra_headers_json: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 256
    llm_timeout_seconds: float = 30.0
    llm_rationale_timeout_seconds: float = 6.0
    enable_llm_rationale: bool = True

    valkey_url: str = "redis://localhost:6379"

    retrain_lock_ttl_seconds: int = 1800
    max_training_time_seconds: int = 3600
    result_ttl_seconds: int = 60
    admin_api_token: str = ""

    no_drift_threshold: float = 0.2
    minor_drift_threshold: float = 0.35
    major_drift_threshold: float = 0.5

    drift_check_every_n_predictions: int = 50
    drift_check_every_t_minutes: int = 10

    prediction_timeout_seconds: int = 30
    incremental_update_wait_seconds: int = 10
    default_horizon: int = 10
    prediction_interval_coverage: float = 0.9

    cache_ttl_high_frequency_seconds: int = 900
    cache_ttl_daily_seconds: int = 21600
    cache_ttl_low_frequency_seconds: int = 86400

    enable_forecast_worker: bool = True
    enable_retrain_worker: bool = False

    watchdog_poll_interval_s: int = 30
    watchdog_min_observations: int = 50
    watchdog_monitor_ttl_s: int = 3600
    watchdog_degradation_thresh: float = 0.25
    watchdog_observation_window_seconds: int = 3600
    watchdog_regression_tolerance: float = 0.15

    min_history_length: int = 30

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_uri: str = "s3://sktime-agentic-models"

    s3_bucket: str = ""
    s3_region: str = ""

    local_dataset_dir: str = ""
    data_loader: object = None
    memory_loader: object = None
    early_stop_mae: float | None = None

    class Config:
        env_file = ".env"
        arbitrary_types_allowed = True