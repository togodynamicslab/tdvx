from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuração do pipeline TDv1-Fast."""
    name: str
    whisper_model: str
    description: str
    estimated_speed: str


TDV1_FAST_CONFIG = ModelConfig(
    name="TDv1-Fast",
    whisper_model="medium",
    description="Faster-Whisper Medium com quantização int8_float16 (GPU) / int8 (CPU) e speaker re-ID",
    estimated_speed="~4-6s por 10s de áudio",
)


def get_model_config() -> ModelConfig:
    return TDV1_FAST_CONFIG
