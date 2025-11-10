import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time

from app.services.whisper_service import whisper_service, get_or_create_whisper_service
from app.services.diarization_service import diarization_service
from app.services.translation_service import translation_service
from app.models.response import TranscriptionSegment, TranscriptionResponse, LiveTranscriptionChunk
from app.config import settings

logger = logging.getLogger(__name__)


class TranscriptionProcessor:
    """
    Combines Whisper transcription, Pyannote diarization, and translation
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

        Args:
            whisper_segments: List of segments from Whisper
            diarization_segments: List of segments from Pyannote

        Returns:
            List of merged segments with speaker labels
        """
        merged = []

        for whisper_seg in whisper_segments:
            start = whisper_seg['start']
            end = whisper_seg['end']
            text = whisper_seg['text'].strip()

            if not text:
                continue

            # Find overlapping speaker
            # Use the speaker with the most overlap
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
        Process audio data through the full pipeline.

        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate
            model_type: Which model pipeline to use ("tdv1" or "tdv1-fast"). Uses default if None.

        Returns:
            TranscriptionResponse with all segments
        """
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate

        # Get appropriate whisper service
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

    def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
        model_type: Optional[str] = None
    ) -> List[LiveTranscriptionChunk]:
        """
        Process audio chunk for live transcription.

        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate
            is_final: Whether this is the final chunk
            model_type: Which model pipeline to use ("tdv1" or "tdv1-fast"). Uses default if None.

        Returns:
            List of LiveTranscriptionChunk objects
        """
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate

        # Get appropriate whisper service
        if model_type is None:
            model_type = settings.default_model

        whisper_svc = get_or_create_whisper_service(model_type)

        # Transcribe audio
        logger.info(f"Transcribing audio chunk using model: {model_type} (live mode with diarization)...")
        whisper_result = whisper_svc.transcribe_audio(audio_data, sample_rate)

        if not whisper_result['segments']:
            logger.warning("No transcription segments found")
            return []

        # Perform diarization on the chunk
        logger.info("Performing diarization on live chunk...")
        diarization_segments = diarization_service.diarize_audio(audio_data, sample_rate, clustering_threshold=settings.pyannote_live_clustering_threshold)

        # Merge transcription with diarization
        merged_segments = self.merge_transcription_and_diarization(
            whisper_result['segments'],
            diarization_segments
        )

        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"Live chunk processed in {processing_time:.2f}s (RTF: {rtf:.3f}x, audio: {audio_duration:.2f}s)")

        # Translate segments with speaker labels
        original_lang = whisper_result['language']
        target_lang = translation_service.get_target_language(original_lang)

        chunks = []
        for seg in merged_segments:
            original_text, translated_text = translation_service.translate(
                seg['text'].strip(),
                original_lang
            )

            if original_text:
                self.chunk_counter += 1
                segment = TranscriptionSegment(
                    speaker=seg['speaker'],  # Use speaker from diarization
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


# Singleton instance
processor = TranscriptionProcessor()
