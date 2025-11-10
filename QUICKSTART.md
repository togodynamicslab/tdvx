# Quick Start Guide

Get your live transcription API running in 5 minutes!

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install all dependencies
pip install -r requirements.txt
```

**Important for GPU users:**
If you have an NVIDIA GPU with CUDA, install PyTorch with CUDA support for 10-20x faster processing:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click "Agree and access repository"

## Step 3: Configure Environment

Edit `.env` file and add your token:

```bash
PYANNOTE_AUTH_TOKEN=hf_your_token_here
```

## Step 4: Verify Setup (Optional)

```bash
python setup.py
```

This will check that everything is configured correctly.

## Step 5: Start the Server

```bash
python app/main.py
```

Wait for the models to load (first time takes ~1-2 minutes to download models).

You should see:
```
INFO:     Starting up...
INFO:     Loading Whisper model: medium
INFO:     Whisper will use device: cuda  # or cpu
INFO:     Whisper model loaded successfully
INFO:     Loading Pyannote diarization pipeline...
INFO:     Pyannote pipeline loaded successfully
INFO:     Startup complete!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 6: Test Live Transcription

1. Open `test_client.html` in your browser (Chrome recommended)
2. Click "Start Recording"
3. Speak in Portuguese or English
4. Watch the transcriptions appear with translations!

## Testing with cURL

Test the file upload endpoint:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audio.wav" \
  -H "accept: application/json"
```

## Configuration Options

Edit `.env` to customize:

```bash
# Use smaller/faster model
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large

# Disable speaker diarization for faster processing
ENABLE_DIARIZATION=false

# Adjust chunk size for live transcription
CHUNK_DURATION_SECONDS=2.0  # Smaller = lower latency, less accurate
```

## Troubleshooting

**Models downloading slowly?**
- First time setup downloads ~1.5GB of models
- Be patient, subsequent starts will be instant

**CUDA out of memory?**
- Use smaller model: `WHISPER_MODEL=small` or `base`
- Reduce chunk duration: `CHUNK_DURATION_SECONDS=2.0`

**Pyannote authentication error?**
- Check your token in `.env`
- Make sure you accepted the model license

**WebSocket not connecting?**
- Check server is running on port 8000
- Try `http://localhost:8000/` in browser to verify

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Check the [API docs](http://localhost:8000/docs) when server is running
- Review [CLAUDE.md](CLAUDE.md) for architecture details

## Performance Tips

**For real-time on CPU:**
- Use `WHISPER_MODEL=tiny` or `base`
- Set `ENABLE_DIARIZATION=false`
- Increase `CHUNK_DURATION_SECONDS=3.0`

**For best accuracy (with GPU):**
- Use `WHISPER_MODEL=medium` (default)
- Keep `ENABLE_DIARIZATION=true`
- Use `CHUNK_DURATION_SECONDS=2.5`

Enjoy your live transcription API! 🎉
