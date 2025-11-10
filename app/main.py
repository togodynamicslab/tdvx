from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import numpy as np
import json
from datetime import datetime
import tempfile
import os
from pathlib import Path
from typing import Optional
import time

from app.config import settings
from app.services.whisper_service import whisper_service, get_or_create_whisper_service
from app.services.diarization_service import diarization_service
from app.services.audio_buffer import AudioBuffer
from app.services.processor import processor
from app.services.vad_service import vad_service
from app.models.response import TranscriptionResponse, ErrorResponse
from app.models.model_config import ModelType, get_all_model_configs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Live Transcription API",
    description="Real-time transcription with speaker diarization and translation (pt-BR ↔ en-US)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up...")

    # Load default model
    logger.info(f"Default model: {settings.default_model}")
    default_whisper_service = get_or_create_whisper_service(settings.default_model)
    default_whisper_service.load_model()

    # Keep legacy whisper_service for backward compatibility
    logger.info(f"Loading legacy Whisper model: {settings.whisper_model}")
    whisper_service.load_model()

    if settings.enable_diarization:
        logger.info("Loading Pyannote diarization pipeline...")
        diarization_service.load_pipeline()
    else:
        logger.info("Diarization is disabled")

    if settings.enable_vad:
        logger.info(f"VAD enabled (aggressiveness: {settings.vad_aggressiveness})")
    else:
        logger.info("VAD is disabled")

    logger.info("Startup complete!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return """
    <html>
        <body>
            <h1>Live Transcription API</h1>
            <p>Status: Running</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """

@app.get("/upload.html", response_class=HTMLResponse)
async def upload_page():
    """Serve the file upload UI"""
    html_path = Path(__file__).parent.parent / "static" / "upload.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return """
    <html>
        <body>
            <h1>File Upload</h1>
            <p>Upload page not found</p>
            <p><a href="/">Back to Live Transcription</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "running",
        "whisper_model": settings.whisper_model,
        "diarization_enabled": settings.enable_diarization,
        "device": whisper_service.device,
        "default_model": settings.default_model
    }


@app.get("/models")
async def list_models():
    """List available transcription models"""
    models = get_all_model_configs()
    return {
        "default_model": settings.default_model,
        "available_models": {
            model_type: {
                "name": config.name,
                "whisper_model": config.whisper_model,
                "uses_faster_whisper": config.uses_faster_whisper,
                "description": config.description,
                "estimated_speed": config.estimated_speed
            }
            for model_type, config in models.items()
        }
    }


@app.websocket("/transcribe/live")
async def websocket_transcribe_live(websocket: WebSocket):
    """
    WebSocket endpoint for live transcription from browser.

    Client sends WebM/Opus audio chunks.
    Server processes and returns JSON transcription results.
    """
    await websocket.accept()
    logger.info(f"Live transcription WebSocket connected: {websocket.client}")

    # Create audio buffer
    audio_buffer = AudioBuffer(sample_rate=16000)
    processor.reset_counter()

    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]

                if len(audio_bytes) == 0:
                    logger.info("End of stream signal received")
                    break

                # Decode WebM/Opus audio using ffmpeg
                try:
                    import subprocess
                    import io

                    # Use ffmpeg to decode WebM to raw PCM
                    process = subprocess.Popen([
                        'ffmpeg',
                        '-i', 'pipe:0',  # Input from stdin
                        '-f', 'f32le',   # Output format: float32 little-endian
                        '-acodec', 'pcm_f32le',
                        '-ar', '16000',  # Sample rate 16kHz
                        '-ac', '1',      # Mono
                        'pipe:1'         # Output to stdout
                    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    audio_pcm, stderr = process.communicate(input=audio_bytes)

                    if process.returncode != 0:
                        logger.error(f"FFmpeg error: {stderr.decode()}")
                        continue

                    # Convert to numpy array
                    audio_chunk = np.frombuffer(audio_pcm, dtype=np.float32)

                except Exception as e:
                    logger.error(f"Failed to parse audio: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid audio format: {str(e)}"
                    })
                    continue

                # Add to buffer
                processable_chunk = audio_buffer.add_chunk(audio_chunk)

                if processable_chunk is not None:
                    logger.info(f"Processing {len(processable_chunk)/16000:.2f}s of audio")

                    try:
                        # Process
                        chunks = processor.process_audio_chunk(
                            processable_chunk,
                            sample_rate=16000,
                            is_final=False
                        )

                        # Send results
                        for chunk in chunks:
                            response = {
                                "type": "transcription",
                                "speaker": chunk.speaker,
                                "text": chunk.text,
                                "start": chunk.start,
                                "end": chunk.end,
                                "translation": chunk.translation
                            }
                            await websocket.send_json(response)

                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })

            elif "text" in data:
                message = data["text"]
                if message == "end":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Process remaining audio
        remaining = audio_buffer.get_remaining()
        if remaining is not None and len(remaining) > 0:
            try:
                chunks = processor.process_audio_chunk(remaining, sample_rate=16000, is_final=True)
                for chunk in chunks:
                    response = {
                        "type": "transcription",
                        "speaker": chunk.speaker,
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.end,
                        "translation": chunk.translation
                    }
                    await websocket.send_json(response)
            except:
                pass

        logger.info("Live transcription WebSocket closed")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    model: Optional[str] = Query(None, description="Model to use: 'tdv1' or 'tdv1-fast'. Uses default if not specified.")
):
    """
    WebSocket endpoint for live transcription.

    Client should send audio chunks as binary data (PCM float32, 16kHz, mono).
    Server will send back JSON transcription chunks as they're processed.

    Protocol:
    - Client connects with optional ?model=tdv1 or ?model=tdv1-fast query parameter
    - Client sends binary audio chunks
    - Server processes and sends JSON responses
    - Client sends empty message or disconnects to end
    """
    await websocket.accept()

    # Use default model if not specified
    selected_model = model if model else settings.default_model
    logger.info(f"WebSocket connected: {websocket.client} (model: {selected_model})")

    # Create audio buffer for this connection
    audio_buffer = AudioBuffer(sample_rate=16000)
    processor.reset_counter()

    try:
        while True:
            # Receive audio chunk from client
            data = await websocket.receive()

            if "bytes" in data:
                # Binary audio data received
                audio_bytes = data["bytes"]

                if len(audio_bytes) == 0:
                    # Empty message signals end of stream
                    logger.info("End of stream signal received")
                    break

                # Convert bytes to numpy array (assuming float32 PCM)
                try:
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to parse audio chunk: {e}")
                    await websocket.send_json({
                        "error": "Invalid audio format. Expected float32 PCM."
                    })
                    continue

                # Add to buffer
                processable_chunk = audio_buffer.add_chunk(audio_chunk)

                if processable_chunk is not None:
                    # We have enough audio to process
                    duration = len(processable_chunk)/16000

                    # Check for voice activity if VAD is enabled
                    if settings.enable_vad:
                        speech_ratio = vad_service.get_speech_ratio(processable_chunk, sample_rate=16000)
                        has_speech = vad_service.is_speech(processable_chunk, sample_rate=16000)

                        logger.info(f"VAD check: {speech_ratio:.2%} speech ratio, has_speech={has_speech}")

                        if not has_speech:
                            logger.info(f"Skipping {duration:.2f}s chunk (no speech detected)")
                            continue

                    logger.info(f"Processing {len(processable_chunk)} samples ({duration:.2f}s)")

                    try:
                        # Process through pipeline
                        chunks = processor.process_audio_chunk(
                            processable_chunk,
                            sample_rate=16000,
                            is_final=False,
                            model_type=selected_model
                        )

                        # Send each chunk back to client
                        for chunk in chunks:
                            await websocket.send_json(chunk.model_dump(mode='json'))

                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        await websocket.send_json({
                            "error": f"Processing failed: {str(e)}"
                        })

            elif "text" in data:
                # Text message received (could be control message)
                message = data["text"]

                if message == "end":
                    logger.info("End command received")
                    break

                # Handle other text messages if needed
                logger.info(f"Received text message: {message}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Process any remaining audio in buffer
        remaining = audio_buffer.get_remaining()
        if remaining is not None and len(remaining) > 0:
            logger.info(f"Processing remaining {len(remaining)} samples")
            try:
                chunks = processor.process_audio_chunk(
                    remaining,
                    sample_rate=16000,
                    is_final=True,
                    model_type=selected_model
                )
                for chunk in chunks:
                    await websocket.send_json(chunk.model_dump(mode='json'))
            except Exception as e:
                logger.error(f"Error processing remaining audio: {e}")

        logger.info("WebSocket connection closed")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model to use: 'tdv1' or 'tdv1-fast'. Uses default if not specified.")
):
    """
    Transcribe an uploaded audio file.

    Accepts: WAV, MP3, M4A, FLAC, etc.
    Returns: Complete transcription with speaker diarization and translation
    """
    request_start = time.time()

    # Use default model if not specified
    if model is None:
        model = settings.default_model

    logger.info(f"Received file: {file.filename} (model: {model})")

    # Validate file size
    max_size = settings.max_audio_file_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_audio_file_size_mb}MB"
        )

    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Get appropriate whisper service for the model
        whisper_svc = get_or_create_whisper_service(model)

        # Transcribe with Whisper
        logger.info(f"Transcribing file with model: {model}...")
        whisper_result = whisper_svc.transcribe_file(tmp_path)

        if not whisper_result['segments']:
            os.unlink(tmp_path)
            return TranscriptionResponse(
                timestamp=datetime.now(),
                original_language=whisper_result.get('language', 'unknown'),
                target_language='unknown',
                segments=[]
            )

        # Diarize
        logger.info("Performing diarization...")
        diarization_segments = diarization_service.diarize_file(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        # Merge and translate
        from app.services.translation_service import translation_service
        from app.models.response import TranscriptionSegment

        merged_segments = processor.merge_transcription_and_diarization(
            whisper_result['segments'],
            diarization_segments
        )

        original_lang = whisper_result['language']
        target_lang = translation_service.get_target_language(original_lang)

        final_segments = []
        for seg in merged_segments:
            original_text, translated_text = translation_service.translate(
                seg['text'],
                original_lang
            )

            final_segments.append(TranscriptionSegment(
                speaker=seg['speaker'],
                start=seg['start'],
                end=seg['end'],
                text=original_text,
                translation=translated_text
            ))

        total_request_time = time.time() - request_start
        logger.info(f"Total request time: {total_request_time:.2f}s (including upload, processing, and response)")

        return TranscriptionResponse(
            timestamp=datetime.now(),
            original_language=original_lang,
            target_language=target_lang,
            segments=final_segments
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(f"Request failed after {time.time() - request_start:.2f}s")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
