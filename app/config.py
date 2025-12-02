from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration"""

    # Pyannote authentication
    pyannote_auth_token: str

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Model settings (legacy - kept for backward compatibility)
    whisper_model: str = "medium"
    enable_diarization: bool = True

    # Dual Pipeline Configuration
    default_model: str = "tdv1-fast"  # Which pipeline to use by default

    # TDv1 Pipeline (High Quality): Whisper Large-v3 + Diarization + Google Translate
    enable_tdv1: bool = True
    tdv1_whisper_model: str = "large-v3"

    # TDv1-Balanced Pipeline (Balanced): Whisper Medium + Diarization + Google Translate
    enable_tdv1_balanced: bool = True
    tdv1_balanced_whisper_model: str = "medium"

    # TDv1-Fast Pipeline (Real-time): Faster-Whisper Small + Diarization + Google Translate
    enable_tdv1_fast: bool = True
    tdv1_fast_whisper_model: str = "small"

    # Audio processing
    chunk_duration_seconds: float = 2.5
    max_audio_file_size_mb: int = 100

    # VAD settings
    enable_vad: bool = True
    vad_aggressiveness: int = 3  # 0-3, higher = more aggressive

    # Pyannote Diarization settings
    pyannote_min_speakers: int = 1  # Minimum number of speakers (1 = auto-detect)
    pyannote_max_speakers: int = 10  # Maximum number of speakers
    # Clustering threshold: Lower = more speakers detected (0.0-1.0, default 0.7153814901696104)
    # Lower values are more sensitive and will split speakers more aggressively
    pyannote_clustering_threshold: float = 0.5  # For file uploads with full audio context
    pyannote_live_clustering_threshold: float = 0.65  # For live transcription (more conservative)
    # Segmentation threshold: Lower = more speech segments (default varies by model)
    pyannote_segmentation_onset: float = 0.5  # Speech detection sensitivity

    # API Key Authentication
    enable_api_key_auth: bool = True  # Enable/disable API key authentication
    master_api_key: str  # Master key for API key management endpoints

    # Database
    database_path: str = "data/tdvx.db"  # Path to SQLite database

    # Rate Limiting
    default_rate_limit_per_hour: int = 100  # Default rate limit for new API keys
    enable_usage_logging: bool = True  # Enable detailed usage logging

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables not in the model


settings = Settings()
