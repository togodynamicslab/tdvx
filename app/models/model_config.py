from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


# Repo-local CT2 model directories live under <repo>/models/.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TDV1_CV_PT_PATH = str(_MODELS_DIR / "tdv1-cv-pt-ct2")
TDV3_CV_PT_PATH = str(_MODELS_DIR / "tdv3-cv-pt-ct2")


class ModelType(str, Enum):
    """Enum for available transcription pipelines (all faster-whisper)."""
    TDV1 = "tdv1"                    # high quality: large-v3
    TDV1_MEDIUM = "tdv1-medium"      # balanced: medium
    TDV1_BALANCED = "tdv1-balanced"  # alias of tdv1-medium (backwards compat)
    TDV1_FAST = "tdv1-fast"          # real-time: small
    TDV1_CV_PT = "tdv1-cv-pt"        # fine-tuned medium on CV-PT v1 (local CT2)
    TDV3_CV_PT = "tdv3-cv-pt"        # fine-tuned medium on CV-PT v3 (local CT2)


@dataclass
class ModelConfig:
    """Configuration for a transcription model pipeline."""
    name: str
    whisper_model: str  # size: "large-v3" | "medium" | "small" | local CT2 path
    uses_faster_whisper: bool = True
    description: str = ""
    estimated_speed: str = ""


TDV1_CONFIG = ModelConfig(
    name="TDv1",
    whisper_model=settings.tdv1_whisper_model,
    description="High quality — Faster-Whisper large-v3",
    estimated_speed="~4-6s per 10s of audio",
)

TDV1_MEDIUM_CONFIG = ModelConfig(
    name="TDv1-Medium",
    whisper_model=settings.tdv1_balanced_whisper_model,
    description="Balanced — Faster-Whisper medium",
    estimated_speed="~2-3s per 10s of audio",
)

TDV1_FAST_CONFIG = ModelConfig(
    name="TDv1-Fast",
    whisper_model=settings.tdv1_fast_whisper_model,
    description="Real-time — Faster-Whisper small",
    estimated_speed="~0.5-1s per 10s of audio",
)

TDV1_CV_PT_CONFIG = ModelConfig(
    name="TDv1-CV-PT",
    whisper_model=TDV1_CV_PT_PATH,
    description="Fine-tuned Whisper medium on CommonVoice PT v1 (local CT2, int8)",
    estimated_speed="~2-3s per 10s of audio",
)

TDV3_CV_PT_CONFIG = ModelConfig(
    name="TDv3-CV-PT",
    whisper_model=TDV3_CV_PT_PATH,
    description="Fine-tuned Whisper medium on CommonVoice PT v3 (local CT2, int8)",
    estimated_speed="~2-3s per 10s of audio",
)


_CONFIGS: dict[str, ModelConfig] = {
    ModelType.TDV1: TDV1_CONFIG,
    ModelType.TDV1_MEDIUM: TDV1_MEDIUM_CONFIG,
    ModelType.TDV1_BALANCED: TDV1_MEDIUM_CONFIG,  # alias
    ModelType.TDV1_FAST: TDV1_FAST_CONFIG,
    ModelType.TDV1_CV_PT: TDV1_CV_PT_CONFIG,
    ModelType.TDV3_CV_PT: TDV3_CV_PT_CONFIG,
}


def get_model_config(model_type: str) -> ModelConfig:
    key = model_type.lower()
    if key not in _CONFIGS:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of {list(_CONFIGS)}"
        )
    return _CONFIGS[key]


def get_all_model_configs() -> dict[str, ModelConfig]:
    # Exclude the alias from public listings
    return {
        ModelType.TDV1: TDV1_CONFIG,
        ModelType.TDV1_MEDIUM: TDV1_MEDIUM_CONFIG,
        ModelType.TDV1_FAST: TDV1_FAST_CONFIG,
        ModelType.TDV1_CV_PT: TDV1_CV_PT_CONFIG,
        ModelType.TDV3_CV_PT: TDV3_CV_PT_CONFIG,
    }
