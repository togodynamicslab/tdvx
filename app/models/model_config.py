from dataclasses import dataclass
from typing import Dict

from app.config import settings


class ModelType:
    """Simple string constants used across the codebase.

    The rest of the code expects to compare lowercase strings directly
    (e.g. model_type == ModelType.TDV1), so these are plain string
    constants rather than an Enum.
    """

    TDV1 = "tdv1"
    TDV1_BALANCED = "tdv1-balanced"
    TDV1_FAST = "tdv1-fast"


@dataclass
class ModelConfig:
    name: str
    whisper_model: str
    uses_faster_whisper: bool
    description: str
    estimated_speed: str = "unknown"


def get_all_model_configs() -> Dict[str, ModelConfig]:
    """Return all available model configurations keyed by model_type string."""
    return {
        ModelType.TDV1: ModelConfig(
            name="TDv1 (High quality)",
            whisper_model=getattr(settings, "tdv1_whisper_model", settings.whisper_model),
            uses_faster_whisper=False,
            description="High-quality pipeline (Whisper large-v3)",
            estimated_speed="slow",
        ),
        ModelType.TDV1_BALANCED: ModelConfig(
            name="TDv1-Balanced",
            whisper_model=getattr(settings, "tdv1_balanced_whisper_model", settings.whisper_model),
            uses_faster_whisper=False,
            description="Balanced quality/speed (Whisper medium)",
            estimated_speed="moderate",
        ),
        ModelType.TDV1_FAST: ModelConfig(
            name="TDv1-Fast",
            whisper_model=getattr(settings, "tdv1_fast_whisper_model", settings.whisper_model),
            uses_faster_whisper=True,
            description="Real-time pipeline (Faster-Whisper small)",
            estimated_speed="fast",
        ),
    }


def get_model_config(model_type: str) -> ModelConfig:
    """Return ModelConfig for given model_type string (case-insensitive)."""
    if model_type is None:
        model_type = settings.default_model

    key = model_type.lower()
    configs = get_all_model_configs()
    if key in configs:
        return configs[key]

    raise ValueError(f"Unknown model type: {model_type}")
