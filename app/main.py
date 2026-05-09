from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
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
from app.services import diarization_session
from app.services.audio_buffer import AudioBuffer
from app.services.processor import processor
from app.services.vad_service import vad_service
from app.models.response import TranscriptionResponse, ErrorResponse
from app.models.model_config import ModelType, get_all_model_configs
from app.services import stress_results
from app.services import eval_results
from app.services import validations as validations_service
from app.services import youtube_service

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

# Mount Vite-built assets at /assets (built React SPA lives under static/dist/).
dist_path = static_path / "dist"
if (dist_path / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="vite-assets")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up...")

    # Load default model (faster-whisper)
    logger.info(f"Default model: {settings.default_model}")
    default_whisper_service = get_or_create_whisper_service(settings.default_model)
    default_whisper_service.load_model()

    if settings.enable_diarization:
        logger.info("Loading Pyannote diarization pipeline...")
        diarization_service.load_pipeline()
    else:
        logger.info("Diarization is disabled")

    # Diarizer backend flag — prototype-only. We always load Pyannote (it's the
    # production path). If sortformer is selected we attempt a best-effort load
    # for early failure detection, but the live endpoints still call Pyannote
    # until the benchmark says otherwise.
    if settings.diarizer_backend == "sortformer":
        logger.warning(
            "DIARIZER_BACKEND=sortformer is a prototype flag. "
            "Live endpoints still use Pyannote; run scripts/bench_diarizers.py to A/B."
        )
        try:
            from app.services.sortformer_service import sortformer_service
            if sortformer_service.is_available():
                logger.info("Sortformer model loaded and ready (prototype path).")
            else:
                logger.warning(
                    "Sortformer requested but unavailable — falling back to Pyannote. "
                    "Install nemo_toolkit[asr] to enable."
                )
        except Exception as e:
            logger.warning(f"Sortformer init raised: {type(e).__name__}: {e}. Falling back to Pyannote.")
    elif settings.diarizer_backend != "pyannote":
        logger.warning(
            f"Unknown DIARIZER_BACKEND={settings.diarizer_backend!r}; defaulting to pyannote."
        )

    if settings.enable_vad:
        logger.info(f"VAD enabled (aggressiveness: {settings.vad_aggressiveness})")
    else:
        logger.info("VAD is disabled")

    logger.info("Startup complete!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the React SPA (built) if available, else the legacy static index."""
    dist_index = Path(__file__).parent.parent / "static" / "dist" / "index.html"
    if dist_index.exists():
        return dist_index.read_text(encoding="utf-8")
    legacy = Path(__file__).parent.parent / "static" / "index.html"
    if legacy.exists():
        return legacy.read_text(encoding="utf-8")
    return """
    <html>
        <body>
            <h1>Live Transcription API</h1>
            <p>Status: Running</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_ui():
    """Original hand-rolled live UI — kept for comparison."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse(status_code=404, content="legacy UI missing")

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


@app.get("/api/runs")
async def api_list_runs():
    """List stress-test runs (summaries only)."""
    return {"runs": stress_results.list_runs()}


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str):
    """Fetch full detail for a single run."""
    detail = stress_results.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return detail


@app.get("/api/runs/{run_id}/summary.md")
async def api_get_run_summary_md(run_id: str):
    """Download the summary.md file for a run."""
    from fastapi.responses import Response
    path = stress_results.RESULTS_ROOT / run_id / "summary.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"summary.md not found for {run_id}")
    return Response(
        content=path.read_text(),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_summary.md"'},
    )


@app.get("/api/runs/{run_id}/requests.ndjson")
async def api_get_run_ndjson(run_id: str):
    """Download the raw requests.ndjson file for a run."""
    from fastapi.responses import Response
    path = stress_results.RESULTS_ROOT / run_id / "requests.ndjson"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"requests.ndjson not found for {run_id}")
    return Response(
        content=path.read_bytes(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_requests.ndjson"'},
    )


# Pipeline-eval read API. Mirrors /api/runs/* but sources data from
# results/eval_history.ndjson (the journal) plus per-run rows.ndjson + summary.md.
# Read-only for now; the "run from UI" endpoints come in step 2.
@app.get("/api/evals")
async def api_list_evals():
    """List all pipeline-eval runs, newest first."""
    return {"evals": eval_results.list_runs()}


@app.get("/api/evals/{run_id}")
async def api_get_eval(run_id: str):
    """Full detail for one eval: journal entry + per-file rows + raw summary.md."""
    detail = eval_results.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"eval {run_id} not found")
    return detail


@app.get("/api/evals/{run_id}/summary.md")
async def api_get_eval_summary_md(run_id: str):
    """Download the summary.md file for an eval run."""
    from fastapi.responses import Response
    path = eval_results.run_dir_for(run_id) / "summary.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"summary.md not found for {run_id}")
    return Response(
        content=path.read_text(),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_summary.md"'},
    )


@app.get("/api/evals/{run_id}/rows.ndjson")
async def api_get_eval_ndjson(run_id: str):
    """Download per-file rows.ndjson for an eval run."""
    from fastapi.responses import Response
    path = eval_results.run_dir_for(run_id) / "rows.ndjson"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"rows.ndjson not found for {run_id}")
    return Response(
        content=path.read_bytes(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_rows.ndjson"'},
    )


# ── Human-validation API ─────────────────────────────────────────
# Lets a human listen to each corpus file and pick which transcript is right
# (ref / hyp / neither / equivalent). Verdicts persist to disk and become the
# real ground-truth eval over time.

from pydantic import BaseModel


class _ValidationIn(BaseModel):
    verdict: str  # "ref" | "hyp" | "neither" | "equivalent"
    corrected_text: Optional[str] = None
    validator: Optional[str] = "human"
    notes: Optional[str] = ""


@app.get("/api/validations/{lang}")
async def api_list_validations(lang: str):
    """All stored verdicts for a language plus aggregate stats."""
    try:
        data = validations_service.list_validations(lang)
        st = validations_service.stats(lang)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"lang": lang, "validations": data, "stats": st}


@app.post("/api/validations/{lang}/{filename}")
async def api_save_validation(lang: str, filename: str, body: _ValidationIn):
    try:
        entry = validations_service.save_validation(
            lang, filename,
            verdict=body.verdict,
            corrected_text=body.corrected_text,
            validator=body.validator or "human",
            notes=body.notes or "",
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "entry": entry}


@app.delete("/api/validations/{lang}/{filename}")
async def api_delete_validation(lang: str, filename: str):
    try:
        removed = validations_service.delete_validation(lang, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "removed": removed}


class _AutoEquivIn(BaseModel):
    # Per-file pair the client already has loaded — saves a server round-trip
    # to re-read eval rows. Same shape as eval rows.ndjson entries.
    pairs: list[dict]


@app.post("/api/validations/{lang}/auto-equivalent")
async def api_auto_equivalent(lang: str, body: _AutoEquivIn):
    """Bulk-mark files as 'equivalent' when ref/hyp normalize to identical text.

    Saves the human from clicking through trivially-equivalent diffs (case,
    punctuation, contractions). Skips any file with an existing verdict.
    """
    try:
        result = validations_service.auto_prefill_equivalent(lang, body.pairs)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, **result}


@app.get("/api/corpus/{lang}/reference/{filename}")
async def api_get_corpus_reference(lang: str, filename: str):
    """Serve the saved Deepgram reference JSON for a corpus file.

    Used by the validate UI to drive word-level highlighting synced with
    audio playback (Deepgram returns per-word start/end times).
    """
    from fastapi.responses import FileResponse
    if "/" in filename or ".." in filename or "/" in lang or ".." in lang:
        raise HTTPException(status_code=400, detail="invalid path")
    corpus_root = Path(__file__).resolve().parent.parent / "tests" / "corpus" / lang
    if not corpus_root.is_dir():
        raise HTTPException(status_code=404, detail=f"corpus {lang} not found")
    # Strip .wav if present, append .json. Keep callers honest about which file.
    stem = filename[:-4] if filename.endswith(".wav") else filename
    target = (corpus_root / ".reference" / f"{stem}.json").resolve()
    if (corpus_root / ".reference").resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="path escape")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"reference for {filename} not found")
    return FileResponse(str(target), media_type="application/json")


@app.get("/api/corpus/{lang}/{filename}")
async def api_get_corpus_audio(lang: str, filename: str):
    """Stream a corpus WAV for inline playback in the eval UI.

    Strict whitelist:
      - lang must match an existing tests/corpus/<lang> directory
      - filename must end in .wav and contain no path separators
      - resolved path must stay inside the corpus directory (path-traversal guard)
    Corpus is non-sensitive but we still validate to avoid serving arbitrary files.
    """
    from fastapi.responses import FileResponse
    if not filename.endswith(".wav") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    if "/" in lang or ".." in lang:
        raise HTTPException(status_code=400, detail="invalid lang")
    corpus_root = Path(__file__).resolve().parent.parent / "tests" / "corpus" / lang
    if not corpus_root.is_dir():
        raise HTTPException(status_code=404, detail=f"corpus {lang} not found")
    target = (corpus_root / filename).resolve()
    if corpus_root.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="path escape")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found in {lang}")
    return FileResponse(str(target), media_type="audio/wav", filename=filename)


@app.post("/youtube/extract")
async def youtube_extract(payload: dict):
    """Resolve a YouTube URL → cached 16 kHz mono PCM WAV.

    The browser cannot fetch YouTube media directly (CORS). The browser keeps
    the YouTube IFrame embed for playback; this endpoint just exists to make
    the audio available to /youtube/stream for transcription.

    Body: {"url": str}
    Returns: {video_id, title, duration_s, channel}
    """
    url = (payload or {}).get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="missing 'url'")
    try:
        loop = asyncio.get_event_loop()
        # yt-dlp blocks on network IO. Run in the default thread pool so
        # we don't stall the event loop while a long video downloads.
        yt = await loop.run_in_executor(None, youtube_service.fetch_youtube_audio, url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return yt.metadata_dict()


@app.websocket("/youtube/stream")
async def youtube_stream(
    websocket: WebSocket,
    url: str = Query(..., description="YouTube URL or video ID. Must already be cached via /youtube/extract."),
    model: Optional[str] = Query(None),
    language: str = Query("pt"),
    diarize: bool = Query(True),
    translate: bool = Query(False),
    speed: float = Query(1.0, ge=0.1, le=64.0,
                         description="Playback speed multiplier. 1.0 = real-time, up to 64.0 for max-throughput QA. Above ~8x the embed plays at 1x and the transcript runs ahead — that's expected."),
    chunk_seconds: float = Query(2.0, ge=0.5, le=10.0,
                                 description="WS chunk duration sent to the pipeline. Smaller = more granular telemetry, larger = fewer chunk overheads."),
    session_id: Optional[str] = Query(None),
):
    """Replay a cached YouTube video's audio through the live pipeline at
    `speed`× wall-clock pace. Emits the same per-chunk transcription objects
    /ws/transcribe does, so the frontend can reuse its live transcript UI.

    Each emitted chunk also carries a `video_offset_s` field so the frontend
    can sync transcript display with the YouTube IFrame's getCurrentTime().

    Backpressure: chunks are processed sequentially. If the pipeline is
    slower than the requested speed, real wall-clock time will exceed the
    target — that's the whole point of telemetry, you'll see it.
    """
    await websocket.accept()

    # Resolve cached audio. Caller must have hit /youtube/extract first.
    try:
        yt = youtube_service.fetch_youtube_audio(url)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    selected_model = model if model else settings.default_model
    sess_id = session_id or diarization_session.new_session_id()
    logger.info(
        f"youtube/stream connected: video_id={yt.video_id} duration={yt.duration_s:.1f}s "
        f"model={selected_model} speed={speed}x chunk={chunk_seconds}s sess={sess_id[:8]}"
    )

    # Send a header so the client knows what it's receiving.
    await websocket.send_json({
        "type": "header",
        "video_id": yt.video_id,
        "title": yt.title,
        "duration_s": yt.duration_s,
        "channel": yt.channel,
        "model": selected_model,
        "speed": speed,
        "chunk_seconds": chunk_seconds,
    })

    audio, sr = youtube_service.load_audio_pcm_f32(yt.audio_path)
    chunk_samples = int(chunk_seconds * sr)
    total_samples = len(audio)
    processor.reset_counter()

    try:
        for i in range(0, total_samples, chunk_samples):
            # Bail out fast if the client disconnected.
            if websocket.client_state.name != "CONNECTED":
                break

            chunk = audio[i : i + chunk_samples]
            if chunk.size == 0:
                continue
            chunk_start_s = i / float(sr)
            chunk_end_s = min((i + chunk.size) / float(sr), yt.duration_s)
            is_final = (i + chunk_samples) >= total_samples

            t0 = time.time()
            try:
                results = await processor.process_audio_chunk_batched(
                    chunk,
                    sample_rate=sr,
                    is_final=is_final,
                    model_type=selected_model,
                    language=language,
                    session_id=sess_id,
                    diarize=diarize,
                    translate=translate,
                )
            except Exception as e:
                logger.error(f"youtube/stream processing error: {e}", exc_info=True)
                await websocket.send_json({"error": f"Processing failed: {e}"})
                continue
            processing_ms = int((time.time() - t0) * 1000)

            for c in results:
                payload = c.model_dump(mode='json')
                # Translate chunk-relative offsets into video-absolute offsets so
                # the frontend can sync against YouTube's getCurrentTime().
                if "segment" in payload and payload["segment"]:
                    seg = payload["segment"]
                    seg["video_start_s"] = chunk_start_s + (seg.get("start") or 0.0)
                    seg["video_end_s"] = chunk_start_s + (seg.get("end") or 0.0)
                payload["video_offset_s"] = chunk_start_s
                payload["video_chunk_end_s"] = chunk_end_s
                payload["pipeline_ms"] = processing_ms
                await websocket.send_json(payload)

            # Pace the next chunk so the apparent playback rate matches `speed`.
            # If processing already took longer than the target chunk duration,
            # don't sleep — just go again. The user will see telemetry slip,
            # which is the point.
            target_wall_s = (chunk.size / float(sr)) / max(speed, 0.01)
            elapsed_s = time.time() - t0
            sleep_s = target_wall_s - elapsed_s
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        # End-of-stream marker so the frontend can stop polling progress.
        try:
            await websocket.send_json({"type": "eos", "video_id": yt.video_id})
        except Exception:
            pass

    except WebSocketDisconnect:
        logger.info(f"youtube/stream client disconnected (video_id={yt.video_id})")
    except Exception as e:
        logger.exception(f"youtube/stream fatal: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


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

    Uses async parallel pipeline: Whisper + Pyannote run concurrently.
    Audio keeps buffering while previous chunk processes.
    """
    await websocket.accept()
    logger.info(f"Live transcription WebSocket connected: {websocket.client}")

    audio_buffer = AudioBuffer(sample_rate=16000)
    processor.reset_counter()
    processing_task: Optional[asyncio.Task] = None

    async def process_and_send(chunk: np.ndarray, is_final: bool = False):
        """Process a chunk and send results back over WebSocket."""
        try:
            chunks = await processor.process_audio_chunk_async(
                chunk,
                sample_rate=16000,
                is_final=is_final
            )
            for c in chunks:
                response = {
                    "type": "transcription",
                    "speaker": c.segment.speaker,
                    "text": c.segment.text,
                    "start": c.segment.start,
                    "end": c.segment.end,
                    "translation": c.segment.translation
                }
                await websocket.send_json(response)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]

                if len(audio_bytes) == 0:
                    logger.info("End of stream signal received")
                    break

                # Decode WebM/Opus audio using ffmpeg
                try:
                    import subprocess

                    process = subprocess.Popen([
                        'ffmpeg',
                        '-i', 'pipe:0',
                        '-f', 'f32le',
                        '-acodec', 'pcm_f32le',
                        '-ar', '16000',
                        '-ac', '1',
                        'pipe:1'
                    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    audio_pcm, stderr = process.communicate(input=audio_bytes)

                    if process.returncode != 0:
                        logger.error(f"FFmpeg error: {stderr.decode()}")
                        continue

                    audio_chunk = np.frombuffer(audio_pcm, dtype=np.float32)

                except Exception as e:
                    logger.error(f"Failed to parse audio: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid audio format: {str(e)}"
                    })
                    continue

                # Add to VAD-aware buffer
                processable_chunk = audio_buffer.add_chunk(audio_chunk)

                if processable_chunk is not None:
                    duration = len(processable_chunk) / 16000
                    logger.info(f"Buffer ready: {duration:.2f}s of audio")

                    # Wait for previous processing to finish before starting new one
                    if processing_task and not processing_task.done():
                        await processing_task

                    processing_task = asyncio.create_task(
                        process_and_send(processable_chunk)
                    )

            elif "text" in data:
                message = data["text"]
                if message == "end":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Wait for in-flight processing
        if processing_task and not processing_task.done():
            await processing_task

        # Process remaining audio
        remaining = audio_buffer.get_remaining()
        if remaining is not None and len(remaining) > 0:
            await process_and_send(remaining, is_final=True)

        logger.info("Live transcription WebSocket closed")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    model: Optional[str] = Query(None, description="Model to use: 'tdv1', 'tdv1-balanced', or 'tdv1-fast'. Uses default if not specified."),
    language: str = Query("pt", description="Source language code: 'pt' or 'en'. Whisper transcribes in this language."),
    session_id: Optional[str] = Query(None, description="Session identifier for cross-chunk speaker registry. Auto-generated if absent."),
    diarize: bool = Query(True, description="Run per-chunk diarization with cross-chunk speaker identity."),
    translate: bool = Query(False, description="Run translation pass on each segment (adds ~100-500ms)."),
):
    """
    WebSocket endpoint for live transcription (PCM audio).

    Now uses the shared cross-request batcher (whisper-s2t) so concurrent WS
    sessions on a worker share GPU forward passes, matching /transcribe-batch
    throughput. Diarization (when enabled) uses the per-session embedding
    registry — same identity model as the batched HTTP endpoint.
    """
    await websocket.accept()

    selected_model = model if model else settings.default_model
    sess_id = session_id or diarization_session.new_session_id()
    logger.info(f"WebSocket connected: {websocket.client} (model: {selected_model}, sess={sess_id[:8]})")

    audio_buffer = AudioBuffer(sample_rate=16000)
    processor.reset_counter()
    processing_task: Optional[asyncio.Task] = None

    async def process_and_send(chunk: np.ndarray, is_final: bool = False):
        """Process a chunk through the batched pipeline and send results."""
        try:
            chunks = await processor.process_audio_chunk_batched(
                chunk,
                sample_rate=16000,
                is_final=is_final,
                model_type=selected_model,
                language=language,
                session_id=sess_id,
                diarize=diarize,
                translate=translate,
            )
            for c in chunks:
                await websocket.send_json(c.model_dump(mode='json'))
        except Exception as e:
            logger.error(f"Processing error: {e}")
            try:
                await websocket.send_json({"error": f"Processing failed: {str(e)}"})
            except:
                pass

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]

                if len(audio_bytes) == 0:
                    logger.info("End of stream signal received")
                    break

                try:
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to parse audio chunk: {e}")
                    await websocket.send_json({
                        "error": "Invalid audio format. Expected float32 PCM."
                    })
                    continue

                # Client now sends VAD-bounded bursts (≥1.5s). Skip the legacy
                # server-side audio buffer — double-buffering adds 5-10s of
                # latency. If the burst is too short, drop it (silence false-positive).
                duration = len(audio_chunk) / 16000
                if duration < 0.5:
                    continue

                # Fire-and-forget: do NOT serialize per-WS. Multiple in-flight
                # bursts on the same session can run concurrently — the
                # batcher fuses them on the GPU. This is the single biggest
                # latency win for live streaming.
                asyncio.create_task(process_and_send(audio_chunk))
                processing_task = None  # no longer the bottleneck

            elif "text" in data:
                message = data["text"]

                if message == "end":
                    logger.info("End command received")
                    break

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
        # Wait for in-flight processing
        if processing_task and not processing_task.done():
            await processing_task

        # Process any remaining audio in buffer
        remaining = audio_buffer.get_remaining()
        if remaining is not None and len(remaining) > 0:
            logger.info(f"Processing remaining {len(remaining)} samples")
            await process_and_send(remaining, is_final=True)

        logger.info("WebSocket connection closed")


async def _run_transcribe(
    file: UploadFile,
    model: Optional[str],
    diarize: bool,
    translate: bool,
) -> TranscriptionResponse:
    """Shared implementation for /transcribe and /transcribe-fast."""
    request_start = time.time()

    if model is None:
        model = settings.default_model

    logger.info(f"Received file: {file.filename} (model={model}, diarize={diarize}, translate={translate})")

    max_size = settings.max_audio_file_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_audio_file_size_mb}MB",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        whisper_svc = get_or_create_whisper_service(model)

        # Whisper is CPU/GPU-bound and blocks the event loop; offload.
        loop = asyncio.get_event_loop()
        whisper_result = await loop.run_in_executor(None, whisper_svc.transcribe_file, tmp_path)

        if not whisper_result["segments"]:
            return TranscriptionResponse(
                timestamp=datetime.now(),
                original_language=whisper_result.get("language", "unknown"),
                target_language="unknown",
                segments=[],
            )

        # Diarization (optional). Pyannote is GPU-bound; offload as well.
        if diarize:
            logger.info("Performing diarization...")
            diarization_segments = await loop.run_in_executor(
                None, diarization_service.diarize_file, tmp_path
            )
            merged_segments = processor.merge_transcription_and_diarization(
                whisper_result["segments"], diarization_segments
            )
        else:
            # Assign every segment to SPEAKER_00 — correct for the vast majority
            # of short chunks and skips the 1-2s diarization fixed cost.
            merged_segments = [
                {"speaker": "SPEAKER_00", **s} for s in whisper_result["segments"]
            ]

        from app.services.translation_service import translation_service
        from app.models.response import TranscriptionSegment

        original_lang = whisper_result["language"]

        if translate:
            target_lang = translation_service.get_target_language(original_lang)
            final_segments = []
            for seg in merged_segments:
                original_text, translated_text = translation_service.translate(
                    seg["text"], original_lang
                )
                final_segments.append(TranscriptionSegment(
                    speaker=seg["speaker"],
                    start=seg["start"],
                    end=seg["end"],
                    text=original_text,
                    translation=translated_text,
                ))
        else:
            target_lang = "unknown"
            final_segments = [
                TranscriptionSegment(
                    speaker=seg["speaker"],
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    translation=None,
                )
                for seg in merged_segments
            ]

        logger.info(f"Request completed in {time.time() - request_start:.2f}s")

        return TranscriptionResponse(
            timestamp=datetime.now(),
            original_language=original_lang,
            target_language=target_lang,
            segments=final_segments,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(f"Request failed after {time.time() - request_start:.2f}s")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model to use (e.g. 'tdv1', 'tdv1-fast'). Uses default if not specified."),
    diarize: bool = Query(True, description="Run speaker diarization. Disable for short single-speaker chunks to save ~1-2s."),
    translate: bool = Query(True, description="Run translation on each segment. Disable to save ~100-500ms per segment."),
):
    """Full transcription pipeline: Whisper + optional diarization + optional translation."""
    return await _run_transcribe(file, model, diarize=diarize, translate=translate)


@app.post("/transcribe-fast", response_model=TranscriptionResponse)
async def transcribe_file_fast(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model to use (e.g. 'tdv1', 'tdv1-fast'). Uses default if not specified."),
):
    """Lean endpoint for short chunks: Whisper only, no diarization, no translation.

    Designed for continuous-listener workloads (e.g. Omi-style wearables) where
    clips are 2-60s, single-speaker, and translation happens client-side.
    """
    return await _run_transcribe(file, model, diarize=False, translate=False)


def _probe_duration_seconds(path: str) -> float:
    """Cheap audio duration probe via `wave` stdlib (PCM WAV only).
    Falls back to 0 on error so caller can skip diarization safely."""
    try:
        import wave as wave_mod
        with wave_mod.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


@app.post("/transcribe-batch", response_model=TranscriptionResponse)
async def transcribe_file_batch(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model to use (e.g. 'tdv1', 'tdv1-fast'). Uses default if not specified."),
    language: str = Query("pt", description="Source language code: 'pt' or 'en'. Whisper transcribes in this language; no auto-detection."),
    diarize: bool = Query(False, description="Run diarization for clips >= diarize_min_seconds. Off by default — most Omi chunks are single-speaker."),
    diarize_min_seconds: float = Query(10.0, description="Threshold: clips shorter than this bypass Pyannote and get SPEAKER_00."),
    session_id: Optional[str] = Query(None, description="Opaque client session identifier. When set + diarize=true, speaker labels persist across chunks via a per-session embedding registry. Without it, each chunk's labels are independent."),
):
    """Cross-request batched transcription via whisper-s2t.

    By default: Whisper only, no diarization, no translation. Requests are
    queued and dispatched in batches, sharing a single GPU forward pass.
    Higher per-request latency at low load (settling window ~75ms) but ~3×
    throughput at high load.

    With ?diarize=true: clips at or above diarize_min_seconds additionally run
    through Pyannote (one audio at a time — diarization doesn't batch). Short
    clips still take the fast path and get SPEAKER_00.
    """
    from app.services.batch_transcriber import get_or_create_batcher
    from app.models.response import TranscriptionSegment, ChunkTelemetry, SpeakerResolution
    from app.models.model_config import get_model_config

    request_start = time.time()
    if model is None:
        model = settings.default_model

    cfg = get_model_config(model.lower())
    lang = (language or "pt").lower()
    if lang not in ("pt", "en"):
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}. Use 'pt' or 'en'.")
    sess_tag = session_id[:8] if session_id else "-"
    logger.info(f"[batch] {file.filename} → model={cfg.whisper_model}  lang={lang}  diarize={diarize}  sess={sess_tag}")

    max_size = settings.max_audio_file_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_audio_file_size_mb}MB")

    notes: list[str] = []
    whisper_ms = diar_ms = embed_ms = align_ms = 0
    diarize_ran = False
    locals_detected = 0
    resolutions_out: list[SpeakerResolution] = []
    registry_size = 0
    chunk_duration = 0.0
    buffer_seconds = 0.0
    speakers_in_buffer = 0
    chunk_offset_seconds = 0.0

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        chunk_duration = _probe_duration_seconds(tmp_path)

        batcher = await get_or_create_batcher(cfg.whisper_model)
        loop = asyncio.get_event_loop()

        # Decide if diarization runs at all up front (so we can fire it in parallel).
        run_diar = (
            diarize
            and settings.enable_diarization
            and chunk_duration >= diarize_min_seconds
        )
        if diarize and not run_diar:
            notes.append(f"diarization skipped: dur {chunk_duration:.2f}s < min {diarize_min_seconds}s")
            logger.info(f"[batch] {file.filename} skipping diarization (dur={chunk_duration:.2f}s < {diarize_min_seconds}s)")

        # Pre-read audio for windowed path (cheap; ~1ms for an 8s clip).
        chunk_audio = None
        chunk_sr = 16000
        if run_diar and session_id:
            import soundfile as sf
            chunk_audio, chunk_sr = sf.read(tmp_path, dtype="float32", always_2d=False)
            if chunk_audio.ndim > 1:
                chunk_audio = chunk_audio.mean(axis=1)

        # Kick off Whisper + diarization in parallel — independent of each other.
        whisper_task = asyncio.create_task(batcher.enqueue(tmp_path, language=lang))

        diar_task = None
        if run_diar:
            if session_id:
                diar_task = loop.run_in_executor(
                    None,
                    diarization_session.assign_global_speakers_windowed,
                    session_id, chunk_audio, int(chunk_sr), diarization_service,
                )
            else:
                diar_task = loop.run_in_executor(
                    None, diarization_service.diarize_file, tmp_path
                )

        t = time.time()
        result = await whisper_task
        whisper_ms = int((time.time() - t) * 1000)

        if result.get("error"):
            if diar_task and not diar_task.done():
                diar_task.cancel()
            raise HTTPException(status_code=500, detail=result["error"])

        whisper_segments = result.get("segments", [])

        if diar_task is not None and whisper_segments:
            try:
                t = time.time()
                diar_result = await diar_task
                diar_ms = int((time.time() - t) * 1000)

                if session_id:
                    windowed = diar_result
                    embed_ms = 0  # lumped into diar_ms in windowed mode
                    diar_segments = windowed.segments
                    registry_size = windowed.registry_size
                    buffer_seconds = windowed.buffer_seconds
                    speakers_in_buffer = windowed.speakers_in_buffer
                    chunk_offset_seconds = windowed.chunk_offset_seconds
                    for r in windowed.resolutions:
                        resolutions_out.append(SpeakerResolution(
                            local_label=r.local_label,
                            global_label=r.global_label,
                            is_new=r.is_new,
                            distance=r.distance,
                            duration_s=r.duration_s,
                        ))
                else:
                    diar_segments = diar_result
                    notes.append("no session_id — labels reset per chunk")

                locals_detected = len({s["speaker"] for s in diar_segments}) if diar_segments else 0

                t = time.time()
                whisper_segments = processor.merge_transcription_and_diarization(
                    whisper_segments, diar_segments
                )
                align_ms = int((time.time() - t) * 1000)
                diarize_ran = True
            except Exception as e:
                notes.append(f"diarization error: {e}")
                logger.warning(f"[batch] diarization failed for {file.filename}: {e}. Falling back to SPEAKER_00.")

        segments = [
            TranscriptionSegment(
                speaker=s.get("speaker", "SPEAKER_00"),
                start=s["start"],
                end=s["end"],
                text=s["text"],
                translation=None,
            )
            for s in whisper_segments
        ]

        total_ms = int((time.time() - request_start) * 1000)
        worker_tag = os.environ.get("WORKER_TAG")  # set by remote-start-multigpu.sh if present

        telemetry = ChunkTelemetry(
            chunk_duration_s=chunk_duration,
            whisper_ms=whisper_ms,
            diarization_ms=diar_ms,
            embedding_ms=embed_ms,
            alignment_ms=align_ms,
            total_ms=total_ms,
            diarize_ran=diarize_ran,
            locals_detected=locals_detected,
            resolutions=resolutions_out,
            registry_size=registry_size,
            model=cfg.whisper_model,
            worker=worker_tag,
            notes=notes,
            buffer_seconds=buffer_seconds,
            speakers_in_buffer=speakers_in_buffer,
            chunk_offset_seconds=chunk_offset_seconds,
        )

        logger.info(
            f"[batch] {file.filename} done in {total_ms}ms "
            f"(whisper={whisper_ms}ms diar={diar_ms}ms embed={embed_ms}ms align={align_ms}ms "
            f"locals={locals_detected} reg={registry_size} segs={len(segments)})"
        )
        return TranscriptionResponse(
            timestamp=datetime.now(),
            original_language=result.get("language", "unknown"),
            target_language="unknown",
            segments=segments,
            telemetry=telemetry,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
