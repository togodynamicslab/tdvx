from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração da aplicação — lida do .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Autenticação Pyannote (obrigatório)
    pyannote_auth_token: str

    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000

    # Whisper (TDv1-Fast: faster-whisper medium)
    whisper_model: str = "medium"
    cpu_threads: int = 0  # 0 = usa os.cpu_count()

    # Processamento de áudio
    max_audio_file_size_mb: int = 100
    chunk_duration_seconds: float = 10.0
    enable_vad: bool = True
    vad_aggressiveness: int = 1
    speaker_similarity_threshold: float = 0.80

    # Confiança mínima de um segmento Whisper para não ser descartado
    min_segment_confidence: float = 0.30

    # Pyannote — parâmetros de diarização
    pyannote_min_speakers: int = 1   # 0 ou 1 = auto-detect
    pyannote_max_speakers: int = 10
    pyannote_clustering_threshold: float = 0.5
    pyannote_live_clustering_threshold: float = 0.65


settings = Settings()
