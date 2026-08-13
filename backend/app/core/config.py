"""
Leaf-Expert — Application Configuration
Pydantic BaseSettings: reads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    app_name: str = "Leaf-Expert API"
    app_version: str = "2.0.0"
    debug: bool = False

    # CORS — comma-separated origins
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Paths (relative to project root; overridable via env)
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    collected_data_dir: Path = Path("data/collected")
    datastore_db_path: Path = Path("data/datastore.db")

    # Training defaults
    default_backbone: str = "efficientnet_v2_s"
    default_epochs: int = 30
    default_batch_size: int = 32
    default_learning_rate: float = 1e-3
    default_patience: int = 7

    # VLM / DataStore
    vlm_default_provider: str = "gemini"
    vlm_training_threshold: int = 30   # images per class to unlock training
    ollama_host: str = "http://localhost:11434"

    # Device
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
