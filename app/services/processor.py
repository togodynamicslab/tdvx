import asyncio
import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.whisper_service import whisper_service, get_or_create_whisper_service
from app.services.diarization_service import diarization_service
from app.services.translation_service import translation_service
from app.services import diarization_session
from app.services.batch_transcriber import get_or_create_batcher
from app.models.response import (
    TranscriptionSegment,
    TranscriptionResponse,
    LiveTranscriptionChunk,
    ChunkTelemetry,
    SpeakerResolution,
)
from app.models.model_config import get_model_config
from app.config import settings
import os
import tempfile
import soundfile as sf

logger = logging.getLogger(__name__)

# Thread pool for running sync inference in parallel
_executor = ThreadPoolExecutor(max_workers=2)

# Whisper hallucinations — common phantom text generated on silence/noise
_HALLUCINATION_PATTERNS = {
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "thank you for listening",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "bye bye",
    "goodbye",
    "obrigado por assistir",
    "obrigada por assistir",
    "inscreva-se no canal",
    "até a próxima",
    "legendas pela comunidade",
    "legendado por",
    "subtítulos por",
}


def _is_hallucination(text: str) -> bool:
    """Check if transcribed text is a known Whisper hallucination."""
    cleaned = text.strip().lower().rstrip(".!?,")
    return cleaned in _HALLUCINATION_PATTERNS


class TranscriptionProcessor:
    """
    Combines Whisper transcription, Pyannote diarization, and translation.

    For live transcription, runs Whisper and Pyannote concurrently via asyncio
    to minimize latency.
    """

    def __init__(self):
        self.chunk_counter = 0

    def merge_transcription_and_diarization(
        self,
        whisper_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Merge Whisper transcription segments with Pyannote diarization.
        """
        merged = []

        for whisper_seg in whisper_segments:
            start = whisper_seg['start']
            end = whisper_seg['end']
            text = whisper_seg['text'].strip()

            if not text:
                continue

            max_overlap = 0
            assigned_speaker = 'SPEAKER_00'

            for diar_seg in diarization_segments:
                overlap_start = max(start, diar_seg['start'])
                overlap_end = min(end, diar_seg['end'])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = diar_seg['speaker']

            merged.append({
                'start': start,
                'end': end,
                'text': text,
                'speaker': assigned_speaker
            })

        return merged

    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        model_type: Optional[str] = None
    ) -> TranscriptionResponse:
        """
        Process audio data through the full pipeline (file upload path).
        Kept synchronous for file upload compatibility.
        """
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate

        if model_type is None:
            model_type = settings.default_model

        whisper_svc = get_or_create_whisper_service(model_type)

        # Step 1: Transcribe with Whisper
        logger.info(f"Transcribing audio using model: {model_type}...")
        transcribe_start = time.time()
        whisper_result = whisper_svc.transcribe_audio(audio_data, sample_rate)
        transcribe_time = time.time() - transcribe_start
        logger.info(f"Transcription completed in {transcribe_time:.2f}s")

        if not whisper_result['segments']:
            logger.warning("No transcription segments found")
            return TranscriptionResponse(
                timestamp=datetime.now(),
                original_language=whisper_result.get('language', 'unknown'),
                target_language='unknown',
                duration=len(audio_data) / sample_rate,
                segments=[]
            )

        # Step 2: Diarize (identify speakers)
        logger.info("Performing speaker diarization...")
        diarization_segments = diarization_service.diarize_audio(audio_data, sample_rate)

        # Step 3: Merge transcription and diarization
        merged_segments = self.merge_transcription_and_diarization(
            whisper_result['segments'],
            diarization_segments
        )

        # Step 4: Translate each segment
        logger.info("Translating segments...")
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

        total_time = time.time() - start_time
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"=" * 60)
        logger.info(f"PERFORMANCE METRICS - Model: {model_type}")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Real-Time Factor (RTF): {rtf:.3f}x")
        logger.info(f"Speed: {1/rtf:.2f}x faster than real-time" if rtf < 1 else f"Speed: {rtf:.2f}x slower than real-time")
        logger.info(f"=" * 60)

        return TranscriptionResponse(
            timestamp=datetime.now(),
            original_language=original_lang,
            target_language=target_lang,
            duration=len(audio_data) / sample_rate,
            segments=final_segments
        )

    async def process_audio_chunk_async(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
        model_type: Optional[str] = None
    ) -> List[LiveTranscriptionChunk]:
        """
        Process audio chunk for live transcription with parallel execution.

        Runs Whisper transcription and Pyannote diarization concurrently
        using a thread pool, then merges results and translates.
        """
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate

        if model_type is None:
            model_type = settings.default_model

        whisper_svc = get_or_create_whisper_service(model_type)
        loop = asyncio.get_event_loop()

        # Run transcription and diarization concurrently in thread pool
        logger.info(f"Processing {audio_duration:.2f}s chunk — Whisper + Pyannote in parallel (model: {model_type})")

        whisper_future = loop.run_in_executor(
            _executor,
            whisper_svc.transcribe_audio,
            audio_data,
            sample_rate
        )
        diarize_future = loop.run_in_executor(
            _executor,
            diarization_service.diarize_audio,
            audio_data,
            sample_rate,
            settings.pyannote_live_clustering_threshold
        )

        whisper_result, diarization_segments = await asyncio.gather(
            whisper_future, diarize_future
        )

        parallel_time = time.time() - start_time
        logger.info(f"Parallel inference completed in {parallel_time:.2f}s")

        if not whisper_result['segments']:
            logger.warning("No transcription segments found")
            return []

        # Merge transcription with diarization
        merged_segments = self.merge_transcription_and_diarization(
            whisper_result['segments'],
            diarization_segments
        )

        # Translate segments
        original_lang = whisper_result['language']
        target_lang = translation_service.get_target_language(original_lang)

        chunks = []
        for seg in merged_segments:
            text = seg['text'].strip()

            if _is_hallucination(text):
                logger.info(f"Filtered hallucination: '{text}'")
                continue

            original_text, translated_text = translation_service.translate(
                text,
                original_lang
            )

            if original_text:
                self.chunk_counter += 1
                segment = TranscriptionSegment(
                    speaker=seg['speaker'],
                    start=seg['start'],
                    end=seg['end'],
                    text=original_text,
                    translation=translated_text
                )
                chunks.append(LiveTranscriptionChunk(
                    chunk_id=self.chunk_counter,
                    timestamp=datetime.now(),
                    original_language=original_lang,
                    target_language=target_lang,
                    segment=segment,
                    is_final=is_final
                ))

        total_time = time.time() - start_time
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"Live chunk done in {total_time:.2f}s (RTF: {rtf:.3f}x, audio: {audio_duration:.2f}s)")

        return chunks

    def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
        model_type: Optional[str] = None
    ) -> List[LiveTranscriptionChunk]:
        """
        Synchronous fallback for process_audio_chunk (file upload endpoint compatibility).
        """
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate

        if model_type is None:
            model_type = settings.default_model

        whisper_svc = get_or_create_whisper_service(model_type)

        logger.info(f"Transcribing audio chunk using model: {model_type} (sync mode)...")
        whisper_result = whisper_svc.transcribe_audio(audio_data, sample_rate)

        if not whisper_result['segments']:
            logger.warning("No transcription segments found")
            return []

        logger.info("Performing diarization on chunk...")
        diarization_segments = diarization_service.diarize_audio(audio_data, sample_rate, clustering_threshold=settings.pyannote_live_clustering_threshold)

        merged_segments = self.merge_transcription_and_diarization(
            whisper_result['segments'],
            diarization_segments
        )

        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"Live chunk processed in {processing_time:.2f}s (RTF: {rtf:.3f}x, audio: {audio_duration:.2f}s)")

        original_lang = whisper_result['language']
        target_lang = translation_service.get_target_language(original_lang)

        chunks = []
        for seg in merged_segments:
            text = seg['text'].strip()

            if _is_hallucination(text):
                logger.info(f"Filtered hallucination: '{text}'")
                continue

            original_text, translated_text = translation_service.translate(
                text,
                original_lang
            )

            if original_text:
                self.chunk_counter += 1
                segment = TranscriptionSegment(
                    speaker=seg['speaker'],
                    start=seg['start'],
                    end=seg['end'],
                    text=original_text,
                    translation=translated_text
                )
                chunks.append(LiveTranscriptionChunk(
                    chunk_id=self.chunk_counter,
                    timestamp=datetime.now(),
                    original_language=original_lang,
                    target_language=target_lang,
                    segment=segment,
                    is_final=is_final
                ))

        return chunks

    def reset_counter(self):
        """Reset chunk counter (for new WebSocket connection)"""
        self.chunk_counter = 0

    async def process_audio_chunk_batched(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
        model_type: Optional[str] = None,
        language: str = "pt",
        session_id: Optional[str] = None,
        diarize: bool = True,
        translate: bool = False,
    ) -> List[LiveTranscriptionChunk]:
        """Live chunk path that shares the cross-request batcher with /transcribe-batch.

        WS sessions enqueue into the same per-worker BatchTranscriber as HTTP
        requests, so multiple concurrent WS clients on one worker get GPU-fused
        Whisper inference instead of serialized calls. Diarization (when on)
        runs out-of-band and resolves speaker IDs against the per-session
        registry — same path as the batched HTTP endpoint.
        """
        chunk_start = time.time()
        audio_duration = len(audio_data) / sample_rate

        if model_type is None:
            model_type = settings.default_model
        cfg = get_model_config(model_type.lower())

        notes: list[str] = []
        whisper_ms = diar_ms = embed_ms = align_ms = 0
        diarize_ran = False
        locals_detected = 0
        resolutions_out: list[SpeakerResolution] = []
        registry_size = 0
        buffer_seconds = 0.0
        speakers_in_buffer = 0
        chunk_offset_seconds = 0.0

        # Whisper-s2t takes file paths. Write a 16-bit PCM WAV to /tmp; small
        # synchronous I/O — single-digit ms for an 8s clip.
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_data, sample_rate, subtype="PCM_16")

            batcher = await get_or_create_batcher(cfg.whisper_model)
            loop = asyncio.get_event_loop()

            # Kick off Whisper + diarization in parallel: they're independent.
            # Both return awaitables; gather waits for the slower one. This saves
            # min(whisper_ms, diar_ms) per chunk vs the old sequential path.
            parallel_t = time.time()
            whisper_task = asyncio.create_task(batcher.enqueue(tmp_path, language=language))

            diar_task = None
            if diarize and settings.enable_diarization:
                if session_id:
                    diar_task = loop.run_in_executor(
                        None,
                        diarization_session.assign_global_speakers_windowed,
                        session_id, audio_data, sample_rate, diarization_service,
                    )
                else:
                    diar_task = loop.run_in_executor(
                        None, diarization_service.diarize_file, tmp_path
                    )

            # Await Whisper first so we can early-return on errors / empty.
            t = time.time()
            wresult = await whisper_task
            whisper_ms = int((time.time() - t) * 1000)

            if wresult.get("error"):
                if diar_task and not diar_task.done():
                    diar_task.cancel()
                logger.warning(f"[ws-batched] whisper error: {wresult['error']}")
                return []

            whisper_segments = wresult.get("segments", [])
            if not whisper_segments:
                if diar_task and not diar_task.done():
                    diar_task.cancel()
                return []

            if diar_task is not None:
                try:
                    t = time.time()
                    diar_result = await diar_task
                    diar_ms = int((time.time() - t) * 1000)
                    embed_ms = 0  # lumped into diar_ms in windowed mode

                    if session_id:
                        windowed = diar_result
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
                    whisper_segments = self.merge_transcription_and_diarization(
                        whisper_segments, diar_segments
                    )
                    align_ms = int((time.time() - t) * 1000)
                    diarize_ran = True
                except Exception as e:
                    notes.append(f"diarization error: {e}")
                    logger.warning(f"[ws-batched] diarization failed: {e}")

            # parallel_t isn't surfaced separately yet; total_ms covers it.
            _ = parallel_t

            original_lang = wresult.get("language", language)
            target_lang = translation_service.get_target_language(original_lang) if translate else "unknown"

            total_ms = int((time.time() - chunk_start) * 1000)
            telemetry = ChunkTelemetry(
                chunk_duration_s=audio_duration,
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
                worker=os.environ.get("WORKER_TAG"),
                notes=notes,
                buffer_seconds=buffer_seconds,
                speakers_in_buffer=speakers_in_buffer,
                chunk_offset_seconds=chunk_offset_seconds,
            )

            chunks: list[LiveTranscriptionChunk] = []
            for seg in whisper_segments:
                text = (seg.get("text") or "").strip()
                if not text or _is_hallucination(text):
                    continue
                if translate:
                    original_text, translated_text = translation_service.translate(text, original_lang)
                else:
                    original_text, translated_text = text, None

                self.chunk_counter += 1
                chunks.append(LiveTranscriptionChunk(
                    chunk_id=self.chunk_counter,
                    timestamp=datetime.now(),
                    original_language=original_lang,
                    target_language=target_lang,
                    segment=TranscriptionSegment(
                        speaker=seg.get("speaker", "SPEAKER_00"),
                        start=seg["start"],
                        end=seg["end"],
                        text=original_text,
                        translation=translated_text,
                    ),
                    is_final=is_final,
                    # Attach telemetry only to the first chunk emitted from this audio
                    # buffer — keeps payload small while letting clients see it once.
                    telemetry=telemetry if not chunks else None,
                ))

            logger.info(
                f"[ws-batched] dur={audio_duration:.2f}s "
                f"whisper={whisper_ms}ms diar={diar_ms}ms embed={embed_ms}ms "
                f"locals={locals_detected} reg={registry_size} segs={len(chunks)}"
            )
            return chunks
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


# Singleton instance
processor = TranscriptionProcessor()
