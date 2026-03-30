# Vast AI Deployment Guide

This guide explains how to deploy the TDvX transcription service on Vast.ai GPU instances.

## Prerequisites

1. **Vast.ai Account**: Sign up at [https://vast.ai](https://vast.ai)
2. **Hugging Face Token**: Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Accept Pyannote Terms**: Visit [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the model conditions

## Quick Start on Vast.ai

### Step 1: Choose a GPU Instance

1. Go to Vast.ai and click "Search" for GPU instances
2. **Recommended specifications:**
   - **GPU**: NVIDIA RTX 3090, RTX 4090, or A5000 (24GB VRAM minimum)
   - **RAM**: 32GB+ system RAM
   - **Disk**: 100GB+ SSD
   - **Connection**: Good internet speed for model downloads

3. **Filter settings:**
   - GPU RAM: ≥ 24GB
   - Bandwidth: ≥ 100 Mbps
   - Disk Space: ≥ 100GB
   - CUDA version: 12.1+

### Step 2: Deploy with Docker

When creating your instance, use the following configuration:

**Docker Image:**
```
nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

**On-start script:**
```bash
#!/bin/bash

# Update and install dependencies
apt-get update
apt-get install -y python3.10 python3-pip git ffmpeg curl

# Clone the repository
cd /root
git clone https://github.com/togodynamicslab/tdvx.git
cd tdvx

# Install Python dependencies
pip3 install --no-cache-dir -r requirements.txt

# Set your Hugging Face token (REPLACE WITH YOUR TOKEN!)
export PYANNOTE_AUTH_TOKEN="your_huggingface_token_here"

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**OR use Docker Build (Recommended):**
```bash
#!/bin/bash

# Clone the repository
cd /root
git clone https://github.com/togodynamicslab/tdvx.git
cd tdvx

# Build the Docker image
docker build -t tdvx .

# Run the container with your Hugging Face token
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e PYANNOTE_AUTH_TOKEN="your_huggingface_token_here" \
  --name tdvx-service \
  --restart unless-stopped \
  tdvx
```

**Exposed Ports:**
- `8000` (HTTP/TCP) - FastAPI service

### Step 3: Environment Variables

Set these environment variables in your container:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PYANNOTE_AUTH_TOKEN` | Hugging Face token | ✅ Yes | - |
| `DEFAULT_MODEL` | Default transcription model | No | `tdv1-fast` |
| `ENABLE_DIARIZATION` | Enable speaker diarization | No | `true` |
| `PYANNOTE_CLUSTERING_THRESHOLD` | File upload threshold | No | `0.5` |
| `PYANNOTE_LIVE_CLUSTERING_THRESHOLD` | Live transcription threshold | No | `0.65` |
| `CHUNK_DURATION_SECONDS` | Audio chunk duration | No | `10.0` |

### Step 4: Access Your Service

Once the instance is running:

1. Find your instance's IP address in the Vast.ai dashboard
2. Open your browser and go to:
   - **API Docs**: `http://YOUR_INSTANCE_IP:8000/docs`
   - **Health Check**: `http://YOUR_INSTANCE_IP:8000/health`
   - **Web Interface**: `http://YOUR_INSTANCE_IP:8000/`

## Usage

### File Upload Transcription

```bash
curl -X POST "http://YOUR_INSTANCE_IP:8000/transcribe?model=tdv1-fast" \
  -F "file=@your_audio.mp3"
```

### Live Transcription

Connect to WebSocket endpoint:
```
ws://YOUR_INSTANCE_IP:8000/ws/transcribe
```

See the web interface at `http://YOUR_INSTANCE_IP:8000/` for live transcription demo.

## Available Models

| Model | Description | Speed | Quality | Use Case |
|-------|-------------|-------|---------|----------|
| `tdv1` | Whisper Large-v3 | ~15-20s/10s | Highest | Final transcripts |
| `tdv1-balanced` | Whisper Medium | ~8-12s/10s | High | Balanced |
| `tdv1-fast` | Faster-Whisper Medium | ~2-3s/10s | Good | Live/Real-time |

## Performance Expectations

### Recommended GPU Specs:
- **RTX 3090/4090**: ~2-3s per 10s audio (real-time capable)
- **RTX A5000**: ~2-4s per 10s audio
- **RTX 3080**: ~3-5s per 10s audio (minimum recommended)

### First Run:
- Models will download on first use (~5-10GB total)
- Subsequent runs will use cached models

## Cost Optimization

1. **Start with smaller GPU** for testing (RTX 3080 Ti)
2. **Use Spot Instances** for lower costs
3. **Enable auto-shutdown** when idle
4. **Use `tdv1-fast` model** for most use cases

## Troubleshooting

### Container fails to start:
- Check if PYANNOTE_AUTH_TOKEN is set correctly
- Verify you accepted Pyannote model terms on Hugging Face
- Check GPU is available: `nvidia-smi`

### Out of memory errors:
- Use smaller model (`tdv1-fast` instead of `tdv1`)
- Reduce `CHUNK_DURATION_SECONDS` to 5.0
- Choose GPU with more VRAM (24GB+)

### Slow transcription:
- Verify GPU is being used: check `/health` endpoint
- Ensure CUDA 12.1+ is installed
- Try `tdv1-fast` model for better speed

### Connection timeout:
- Check firewall settings on Vast.ai
- Ensure port 8000 is exposed
- Verify instance is running: check instance logs

## Monitoring

Check service health:
```bash
curl http://YOUR_INSTANCE_IP:8000/health
```

View instance logs in Vast.ai dashboard or:
```bash
docker logs tdvx-service
```

## Updating the Service

```bash
# SSH into your instance
cd /root/tdvx
git pull
docker build -t tdvx .
docker stop tdvx-service
docker rm tdvx-service
docker run -d --gpus all -p 8000:8000 \
  -e PYANNOTE_AUTH_TOKEN="your_token" \
  --name tdvx-service --restart unless-stopped tdvx
```

## Security Recommendations

1. **Do not commit** your `.env` file with tokens
2. **Use environment variables** for sensitive data
3. **Set up firewall** rules if exposing publicly
4. **Use HTTPS** in production (consider nginx reverse proxy)
5. **Implement authentication** for production use

## Support

- **Issues**: [GitHub Issues](https://github.com/togodynamicslab/tdvx/issues)
- **Documentation**: See `README.md` and `CLAUDE.md`
- **API Docs**: Available at `/docs` endpoint

## License

See LICENSE file in the repository.
