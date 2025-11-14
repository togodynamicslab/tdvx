from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ModelType(str, Enum):
    """Enum for available transcription pipelines"""
    TDV1 = "tdv1"  # High quality: Whisper Large-v3
    TDV1_BALANCED = "tdv1-balanced"  # Balanced: Whisper Medium
    TDV1_FAST = "tdv1-fast"  # Real-time: Faster-Whisper Medium


@dataclass
class ModelConfig:
    """Configuration for a transcription model pipeline"""
    name: str
    whisper_model: str  # Model name/size (e.g., "large-v3", "small")
    uses_faster_whisper: bool  # True for faster-whisper, False for openai-whisper
    description: str
    estimated_speed: str  # Human-readable speed estimate


# Model configurations
TDV1_CONFIG = ModelConfig(
    name="TDv1",
    whisper_model="large-v3",
    uses_faster_whisper=False,  # Uses openai-whisper
    description="High quality pipeline with Whisper Large-v3 for accurate file transcription",
    estimated_speed="~15-20s per 10s of audio"
)

TDV1_BALANCED_CONFIG = ModelConfig(
    name="TDv1-Balanced",
    whisper_model="medium",
    uses_faster_whisper=False,  # Uses openai-whisper
    description="Balanced pipeline with Whisper Medium for good quality and speed",
    estimated_speed="~8-12s per 10s of audio"
)

TDV1_FAST_CONFIG = ModelConfig(
    name="TDv1-Fast",
    whisper_model="medium",
    uses_faster_whisper=True,  # Uses faster-whisper
    description="Real-time pipeline with Faster-Whisper Medium for live transcription",
    estimated_speed="~4-6s per 10s of audio"
)


def get_model_config(model_type: str) -> ModelConfig:
    """
    Get model configuration for specified model type.

    Args:
        model_type: Model type string ("tdv1", "tdv1-balanced", or "tdv1-fast")

    Returns:
        ModelConfig for the specified model

    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower()

    if model_type == ModelType.TDV1:
        return TDV1_CONFIG
    elif model_type == ModelType.TDV1_BALANCED:
        return TDV1_BALANCED_CONFIG
    elif model_type == ModelType.TDV1_FAST:
        return TDV1_FAST_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'tdv1', 'tdv1-balanced', or 'tdv1-fast'")


def get_all_model_configs() -> dict[str, ModelConfig]:
    """Get all available model configurations"""
    return {
        ModelType.TDV1: TDV1_CONFIG,
        ModelType.TDV1_BALANCED: TDV1_BALANCED_CONFIG,
        ModelType.TDV1_FAST: TDV1_FAST_CONFIG
    }
