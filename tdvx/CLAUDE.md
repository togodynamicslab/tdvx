# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI-based transcription and diarization service combining:
- **Pyannote**: Speaker diarization (identifying who spoke when)
- **Triple Whisper Pipelines**: Three transcription models optimized for different use cases
  - **TDv1**: Whisper Large-v3 for highest-quality file transcription
  - **TDv1-Balanced**: Whisper Medium for balanced quality and speed
  - **TDv1-Fast**: Faster-Whisper Medium for real-time live transcription
- **Google Translate**: Bidirectional translation (pt-BR ↔ en-US)
- **FastAPI**: REST API framework with WebSocket support

**Supported transcription modes:**
- Single file transcription with model selection
- Live/streaming transcription via WebSocket

## Architecture

This is a Python-based API service with triple transcription pipelines:

### Core Components

1. **Triple Transcription Pipelines**

   **TDv1 (High Quality)**:
   - Uses OpenAI Whisper Large-v3
   - Performance: ~15-20s per 10s of audio
   - Use case: File transcription where accuracy is paramount
   - Best for: Final transcripts, summaries, archival

   **TDv1-Balanced (Balanced)**:
   - Uses OpenAI Whisper Medium
   - Performance: ~8-12s per 10s of audio (2x faster than TDv1)
   - Use case: File transcription with good quality and faster processing
   - Best for: Standard file transcription, batch processing

   **TDv1-Fast (Real-time)**:
   - Uses Faster-Whisper Medium (CTranslate2 optimized)
   - Performance: ~4-6s per 10s of audio (3-4x faster than TDv1)
   - Use case: Live transcription with better accuracy
   - Best for: Real-time streaming, live captions

2. **Diarization Pipeline** (Pyannote)
   - Speaker detection and segmentation
   - Requires authentication token for pyannote models
   - Outputs speaker timestamps and segments
   - Shared by both transcription pipelines

3. **Translation Service** (Google Translate)
   - Bidirectional pt-BR ↔ en-US translation
   - Auto-detects source language
   - Shared by both transcription pipelines

4. **API Layer** (FastAPI)
   - `/transcribe` - Process audio file with model selection
   - `/transcribe/live` - WebSocket for browser audio
   - `/ws/transcribe` - WebSocket for PCM audio streams
   - `/models` - List available models
   - `/health` - Health check

### Project Structure
```
td-v1/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── models/              # Pydantic models and configuration
│   │   ├── model_config.py  # Dual pipeline configuration
│   │   └── response.py      # API response models
│   ├── services/            # Business logic
│   │   ├── whisper_service.py      # Dual Whisper implementation
│   │   ├── diarization_service.py  # Pyannote integration
│   │   ├── translation_service.py  # Google Translate
│   │   ├── processor.py            # Pipeline orchestration
│   │   ├── audio_buffer.py         # Audio chunking
│   │   └── vad_service.py          # Voice activity detection
│   ├── config.py            # Settings (from .env)
│   └── utils/               # Helpers
├── static/                  # Frontend UI
│   ├── index.html           # Live transcription UI
│   └── upload.html          # File upload UI
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── tests/                   # Test files
```

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add PYANNOTE_AUTH_TOKEN
```

### Running the Service
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transcription.py

# Run with coverage
pytest --cov=app tests/
```

## Key Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `openai-whisper` - Original Whisper implementation (TDv1)
- `faster-whisper` - Optimized Whisper via CTranslate2 (TDv1-Fast)
- `pyannote.audio` - Diarization pipeline (v3.1)
- `torch` - Required for both Whisper and Pyannote
- `deep-translator` - Google Translate API wrapper
- `python-multipart` - For file uploads
- `websockets` - For live transcription
- `webrtcvad` - Voice activity detection
- `soundfile` - Audio I/O
- `librosa` - Audio processing

## Important Implementation Notes

### Triple Pipeline Architecture

The system supports three transcription pipelines that can be selected via API parameter:

```python
# Get model configuration
from app.models.model_config import ModelType, get_model_config

# TDv1: Highest quality
config_tdv1 = get_model_config(ModelType.TDV1)
# Returns: large-v3, uses openai-whisper

# TDv1-Balanced: Balanced quality and speed
config_tdv1_balanced = get_model_config(ModelType.TDV1_BALANCED)
# Returns: medium, uses openai-whisper

# TDv1-Fast: Real-time
config_tdv1_fast = get_model_config(ModelType.TDV1_FAST)
# Returns: medium, uses faster-whisper
```

### Model Selection

API endpoints accept a `model` query parameter:
```bash
# Use TDv1 (highest quality)
curl -X POST "http://localhost:8000/transcribe?model=tdv1" -F "file=@audio.mp3"

# Use TDv1-Balanced (balanced quality and speed)
curl -X POST "http://localhost:8000/transcribe?model=tdv1-balanced" -F "file=@audio.mp3"

# Use TDv1-Fast (real-time) - default
curl -X POST "http://localhost:8000/transcribe?model=tdv1-fast" -F "file=@audio.mp3"

# List available models
curl "http://localhost:8000/models"
```

### Whisper Service Factory Pattern

```python
from app.services.whisper_service import get_or_create_whisper_service

# Get or create cached service instance
whisper_svc = get_or_create_whisper_service("tdv1-fast")

# Transcribe
result = whisper_svc.transcribe_audio(audio_data, sample_rate=16000)
```

### Pyannote Authentication

Pyannote models require a Hugging Face token:
```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)
```

Set in `.env`:
```
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxx
```

### Configuration (.env)

```bash
# Model Selection
DEFAULT_MODEL=tdv1-fast              # Which pipeline to use by default
ENABLE_TDV1=true                     # Enable high quality pipeline
ENABLE_TDV1_BALANCED=true            # Enable balanced pipeline
ENABLE_TDV1_FAST=true                # Enable fast pipeline
TDV1_WHISPER_MODEL=large-v3          # Model for TDv1
TDV1_BALANCED_WHISPER_MODEL=medium   # Model for TDv1-Balanced
TDV1_FAST_WHISPER_MODEL=medium       # Model for TDv1-Fast

# Legacy (backward compatibility)
WHISPER_MODEL=medium
ENABLE_DIARIZATION=true

# Audio Processing
CHUNK_DURATION_SECONDS=10.0      # Process audio in chunks
MAX_AUDIO_FILE_SIZE_MB=100       # File size limit

# VAD (Voice Activity Detection)
ENABLE_VAD=true
VAD_AGGRESSIVENESS=1             # 0-3, higher = more aggressive

# Pyannote
PYANNOTE_AUTH_TOKEN=hf_xxxx
```

### Performance Considerations

**TDv1 (Highest Quality)**:
- ~15-20 seconds to process 10 seconds of audio
- Uses OpenAI Whisper Large-v3 (3B parameters)
- Higher VRAM usage (~10GB on GPU)
- Best accuracy, especially for difficult audio

**TDv1-Balanced (Balanced)**:
- ~8-12 seconds to process 10 seconds of audio
- Uses OpenAI Whisper Medium (769M parameters)
- Moderate VRAM usage (~5GB on GPU)
- Good balance between quality and speed

**TDv1-Fast (Real-time)**:
- ~4-6 seconds to process 10 seconds of audio
- Uses Faster-Whisper Medium (769M parameters)
- Moderate VRAM usage (~3GB on GPU)
- 3-4x faster than original Whisper
- 60% less memory usage than original Whisper
- Same accuracy as original Whisper Medium

**General**:
- Both models benefit from GPU (CUDA)
- Diarization adds ~2-5 seconds overhead
- Translation is fast (<100ms per segment)
- Use TDv1-Fast for live transcription
- Use TDv1 for final, high-quality transcripts

### Merging Diarization + Transcription

The processor combines:
- Pyannote output: `[(start_time, end_time, speaker_id), ...]`
- Whisper output: `[{text, start, end}, ...]`

To produce: `[{speaker, text, start, end, translation}, ...]`

Implementation in `app/services/processor.py`:
```python
merged_segments = processor.merge_transcription_and_diarization(
    whisper_segments,
    diarization_segments
)
```

### File Upload Handling

- Supports: WAV, MP3, M4A, FLAC
- FFmpeg required for MP3 decoding
- Validates file size limits (default: 100MB)
- Uses temporary storage for uploaded files
- Auto-cleanup after processing

### WebSocket Protocol

**Browser Audio (/transcribe/live)**:
- Client sends WebM/Opus audio chunks
- Server decodes using FFmpeg
- Returns JSON transcription results

**PCM Audio (/ws/transcribe)**:
- Client sends PCM float32 (16kHz, mono)
- Server processes directly
- Returns JSON transcription results

Both support model selection via WebSocket handshake or messages.

## API Response Format

### File Transcription Response
```json
{
  "timestamp": "2025-11-09T15:30:00",
  "original_language": "pt",
  "target_language": "en",
  "duration": 125.4,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "end": 2.3,
      "text": "Olá, como vai?",
      "translation": "Hello, how are you?"
    }
  ]
}
```

### Live Transcription Chunk
```json
{
  "chunk_id": 1,
  "timestamp": "2025-11-09T15:30:00",
  "original_language": "pt",
  "target_language": "en",
  "segment": {
    "speaker": "SPEAKER_00",
    "start": 0.5,
    "end": 2.3,
    "text": "Olá",
    "translation": "Hello"
  },
  "is_final": false
}
```

### Model List Response
```json
{
  "default_model": "tdv1-fast",
  "available_models": {
    "tdv1": {
      "name": "TDv1",
      "whisper_model": "large-v3",
      "uses_faster_whisper": false,
      "description": "High quality pipeline with Whisper Large-v3 for accurate file transcription",
      "estimated_speed": "~15-20s per 10s of audio"
    },
    "tdv1-fast": {
      "name": "TDv1-Fast",
      "whisper_model": "medium",
      "uses_faster_whisper": true,
      "description": "Real-time pipeline with Faster-Whisper Medium for live transcription",
      "estimated_speed": "~4-6s per 10s of audio"
    }
  }
}
```

## Troubleshooting

### FFmpeg Not Found
If MP3 transcription fails:
1. Install FFmpeg: `winget install ffmpeg`
2. Restart server (PATH is set in `whisper_service.py`)

### CUDA Out of Memory
- Use TDv1-Fast instead of TDv1
- Reduce chunk_duration_seconds
- Process on CPU (slower but works)

### Pyannote Authentication Error
- Verify HF token in `.env`
- Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
- Check token has read access

### Slow Transcription
- Check if GPU is being used (see startup logs)
- Use TDv1-Fast for faster processing
- Consider disabling diarization for speed
