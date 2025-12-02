# Live Transcription API

Real-time audio transcription service with speaker diarization and bilingual translation (pt-BR ↔ en-US).

## Features

- **WebSocket Live Transcription**: Real-time audio streaming with instant transcription
- **Speaker Diarization**: Identifies different speakers using Pyannote
- **Bilingual Translation**: Automatically translates between Portuguese and English
- **Single File Processing**: REST API endpoint for batch file transcription
- **Whisper Medium**: Balanced accuracy and performance

## Tech Stack

- **FastAPI**: Web framework and WebSocket support
- **Whisper Medium**: OpenAI's speech-to-text model
- **Pyannote.audio**: Speaker diarization
- **Deep Translator**: Google Translate integration
- **PyTorch**: ML framework (GPU-accelerated when available)

## Prerequisites

- Python 3.11 or 3.12 (required - Python 3.14+ not yet supported by dependencies)
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster processing)
- Hugging Face account and token

## Setup

### 1. Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Get Hugging Face Token

1. Create account at https://huggingface.co/
2. Go to https://huggingface.co/settings/tokens
3. Create a new token (read access is sufficient)
4. Accept the Pyannote model license:
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"

### 3. Install Python Dependencies

```bash
# Create virtual environment with Python 3.11 or 3.12
# Windows:
py -3.11 -m venv venv
# or: py -3.12 -m venv venv

# Linux/macOS:
python3.11 -m venv venv
# or: python3.12 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

**Important**: Python 3.14+ is not yet supported. If you encounter installation errors, make sure you're using Python 3.11 or 3.12.

**Note**: Installing PyTorch with CUDA support (recommended for GPU):
```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
# PYANNOTE_AUTH_TOKEN=your_token_here
```

## Running the Service

### Development Mode

```bash
# Run from the project root directory with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Important**: Always run uvicorn from the project root directory (where `requirements.txt` is located), not from inside the `app/` folder.

The API will be available at http://localhost:8000

## Usage

### 1. WebSocket Live Transcription

Open `test_client.html` in your browser to use the interactive test client.

**WebSocket Protocol:**
- Endpoint: `ws://localhost:8000/ws/transcribe`
- Send: Binary audio data (PCM Float32, 16kHz, mono)
- Receive: JSON transcription chunks

**Example response:**
```json
{
  "chunk_id": 1,
  "timestamp": "2024-01-15T10:30:00",
  "original_language": "pt",
  "target_language": "en",
  "segment": {
    "speaker": "SPEAKER_00",
    "start": 0.5,
    "end": 2.3,
    "text": "Olá, como vai?",
    "translation": "Hello, how are you?"
  },
  "is_final": false
}
```

### 2. Single File Transcription

**Endpoint:** `POST /transcribe`

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -H "accept: application/json"
```

**Example response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "original_language": "pt",
  "target_language": "en",
  "duration": 10.5,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "end": 2.3,
      "text": "Olá, como vai?",
      "translation": "Hello, how are you?"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 2.5,
      "end": 4.8,
      "text": "Tudo bem, obrigado!",
      "translation": "I'm fine, thank you!"
    }
  ]
}
```

### 3. Health Check

```bash
curl http://localhost:8000/
```

## Configuration

Edit `.env` file:

```bash
# Required: Hugging Face token for Pyannote
PYANNOTE_AUTH_TOKEN=your_token_here

# Server settings
HOST=0.0.0.0
PORT=8000

# Model configuration
WHISPER_MODEL=medium           # Options: tiny, base, small, medium, large
ENABLE_DIARIZATION=true        # Set to false to disable speaker detection

# Audio processing
CHUNK_DURATION_SECONDS=2.5     # Buffer size for live transcription
MAX_AUDIO_FILE_SIZE_MB=100     # Maximum upload size
```

## Architecture

```
app/
├── main.py                    # FastAPI application + WebSocket endpoint
├── config.py                  # Configuration management
├── models/
│   └── response.py            # Pydantic response models
└── services/
    ├── whisper_service.py     # Whisper transcription
    ├── diarization_service.py # Pyannote speaker diarization
    ├── translation_service.py # Translation (pt-BR ↔ en-US)
    ├── audio_buffer.py        # Audio chunk buffering for WebSocket
    └── processor.py           # Main processing pipeline
```

## Performance Tips

1. **Use GPU**: CUDA significantly improves processing speed
2. **Adjust chunk duration**: Larger chunks = better accuracy, higher latency
3. **Disable diarization**: Set `ENABLE_DIARIZATION=false` for faster processing
4. **Use smaller Whisper model**: `tiny` or `base` for real-time on CPU

## Troubleshooting

**"ModuleNotFoundError: No module named 'app'" error:**
- Make sure you're running uvicorn from the project root directory
- Don't use `python app/main.py` - use `uvicorn app.main:app` instead
- Ensure your virtual environment is activated

**"CUDA out of memory" error:**
- Use smaller Whisper model (`base` or `small`)
- Reduce chunk duration
- Close other GPU applications

**Pyannote authentication error:**
- Check your Hugging Face token in `.env`
- Ensure you accepted the model license

**Installation errors with Python 3.14+:**
- Use Python 3.11 or 3.12 instead
- Many ML dependencies don't yet support Python 3.14+

**Audio format errors:**
- WebSocket expects PCM Float32, 16kHz, mono
- File endpoint supports most formats via FFmpeg

**Slow processing:**
- Enable GPU/CUDA
- Use smaller Whisper model
- Disable diarization if not needed

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT License - see LICENSE file for details
