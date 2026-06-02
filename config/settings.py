"""
Configuration centralisée du projet Mina-Français
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """Configuration de l'application"""

    # Chemins des données
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CORPUS_DIR: Path = DATA_DIR / "corpus"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    CACHE_DIR: Path = DATA_DIR / "cache"
    MODEL_DIR: Path = DATA_DIR / "models"
    MODELS_DIR: Path = PROJECT_ROOT / "models"  # Alias pour compatibilité
    TEMP_DIR: Path = DATA_DIR / "temp"

    # Modèles
    WHISPER_MODEL: str = "openai/whisper-small"
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # GPU / VRAM
    MAX_VRAM_USAGE_GB: float = 7.5  # Laisser 0.5 Go de marge
    BATCH_SIZE: int = 4

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CACHE_TTL_SECONDS: int = 3600  # 1 heure

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()