from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve o .env sempre a partir da raiz do repositório,
# independente do diretório de trabalho corrente.
_ROOT_ENV = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Configuração da aplicação — lida do .env na raiz do repo."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT_ENV),
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
    whisper_compute_type: Optional[str] = None   # None = auto (int8_float16 GPU / int8 CPU)
    cpu_threads: int = 0  # 0 = usa os.cpu_count()

    @field_validator("whisper_compute_type", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> Optional[str]:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v  # type: ignore[return-value]

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

    # Finetuning dataset
    finetuning_enabled: bool = True
    finetuning_data_dir: str = ""   # vazio = {tdvx_root}/finetuning_data


settings = Settings()
