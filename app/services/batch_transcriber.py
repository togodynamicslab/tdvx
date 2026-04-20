"""
Cross-request batched transcription via whisper-s2t.

Why this exists: faster-whisper's WhisperModel.transcribe() processes one
audio per call. Under load each GPU is stuck in a sequential loop and total
throughput caps at ~1-2 req/s per worker. whisper-s2t wraps the same
CTranslate2 backend but takes a List[audio_paths] and does one batched
encoder + batched beam decode — real cross-request parallelism on the GPU.

Architecture:
  - One BatchTranscriber per model_size per uvicorn worker.
  - Incoming request enqueues (path, Event, result_slot).
  - Background loop wakes every BATCH_WINDOW_MS, pops up to BATCH_SIZE,
    runs one model.transcribe([paths...]) call in an executor thread,
    writes results into slots, sets events.
  - Request handler awaits its event and returns.

Tunables via env (set at worker launch):
  BATCH_SIZE       max batch size (default 8)
  BATCH_WINDOW_MS  settling delay for partial batches (default 75)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "75"))


@dataclass
class _BatchItem:
    audio_path: str
    done_event: asyncio.Event
    language: str = "pt"                 # "pt" | "en"; forwarded to whisper-s2t per-item
    result: dict[str, Any] = field(default_factory=dict)


def _load_s2t_model(model_size: str):
    """Load whisper-s2t model. Imported lazily so the module can be imported
    even if whisper-s2t isn't installed (e.g. in dev without the dep)."""
    import whisper_s2t  # noqa: WPS433 — intentional deferred import

    compute_type = "int8_float16"  # matches faster-whisper default on our GPUs
    logger.info(f"[batcher:{model_size}] loading whisper-s2t (CTranslate2, {compute_type})")
    model = whisper_s2t.load_model(
        model_identifier=model_size,
        backend="CTranslate2",
        compute_type=compute_type,
        device="cuda",
    )
    logger.info(f"[batcher:{model_size}] whisper-s2t model loaded")
    return model


class BatchTranscriber:
    def __init__(self, model_size: str, batch_size: int = BATCH_SIZE, window_ms: int = BATCH_WINDOW_MS):
        self.model_size = model_size
        self.batch_size = batch_size
        self.window_ms = window_ms
        self._model = None
        self._queue: list[_BatchItem] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._stopped = False

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        # Load on the executor so model loading doesn't block the event loop.
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, _load_s2t_model, self.model_size)
        self._loop_task = asyncio.create_task(self._drain_loop(), name=f"batcher-{self.model_size}")

    async def stop(self) -> None:
        self._stopped = True
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _drain_loop(self) -> None:
        while not self._stopped:
            try:
                # Wait for at least one arrival.
                if not self._queue:
                    await asyncio.sleep(self.window_ms / 1000.0)
                    continue

                # Settling window: let more arrivals land if we're not already full.
                if len(self._queue) < self.batch_size:
                    await asyncio.sleep(self.window_ms / 1000.0)

                batch = self._queue[: self.batch_size]
                self._queue = self._queue[self.batch_size :]
                if not batch:
                    continue

                t0 = time.monotonic()
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._run_batch_sync, batch)
                except Exception as e:
                    logger.exception(f"[batcher:{self.model_size}] batch failed: {e}")
                    for item in batch:
                        if not item.done_event.is_set():
                            item.result = {"error": str(e), "text": "", "language": "unknown", "segments": []}
                            item.done_event.set()
                finally:
                    dt = time.monotonic() - t0
                    logger.info(
                        f"[batcher:{self.model_size}] batch={len(batch)} dt={dt:.2f}s "
                        f"per-item={dt / max(1, len(batch)):.2f}s queue_remain={len(self._queue)}"
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(f"[batcher:{self.model_size}] drain loop error")
                await asyncio.sleep(0.1)

    def _run_batch_sync(self, batch: list[_BatchItem]) -> None:
        """Run ONE whisper-s2t transcribe call with all paths in one list.
        Runs in an executor thread — don't touch asyncio state here.
        """
        paths = [item.audio_path for item in batch]
        # whisper-s2t defaults task to "translate" (emits English). Force
        # "transcribe" and pass each item's language through. Items with
        # different languages can coexist in the same batch — whisper-s2t
        # honors per-item lang_codes at the ctranslate2 layer.
        lang_codes = [item.language for item in batch]
        tasks = ["transcribe"] * len(paths)
        try:
            results = self._model.transcribe(
                paths,
                lang_codes=lang_codes,
                tasks=tasks,
                batch_size=self.batch_size,
            )
        except IndexError:
            # whisper-s2t's VAD can produce empty start_ends on silent / very short
            # clips, which then crashes in its post-processing. Treat as "no speech":
            # successful transcription with empty segments. Don't pollute logs as
            # error since it's expected for silent buffers.
            logger.info(f"[batcher:{self.model_size}] silent/short batch ({len(batch)} items) → empty result")
            for item in batch:
                item.result = {"text": "", "language": item.language, "segments": []}
                item.done_event.set()
            return
        except Exception as e:
            # Fan the same error to every item in the batch.
            logger.exception(f"[batcher:{self.model_size}] transcribe failed: {e}")
            for item in batch:
                item.result = {"error": str(e), "text": "", "language": "unknown", "segments": []}
                item.done_event.set()
            return

        # whisper-s2t returns List[List[segment_dict]], one per input file.
        # Each segment dict has at least {start_time, end_time, text, ...}.
        for item, segs in zip(batch, results):
            try:
                normalized = []
                text_parts = []
                for s in segs or []:
                    normalized.append({
                        "start": s.get("start_time", s.get("start", 0.0)),
                        "end":   s.get("end_time",   s.get("end",   0.0)),
                        "text":  s.get("text", ""),
                    })
                    text_parts.append(s.get("text", ""))
                item.result = {
                    "text": " ".join(t.strip() for t in text_parts).strip(),
                    "language": item.language,
                    "segments": normalized,
                }
            except Exception as e:
                item.result = {"error": f"post: {e}", "text": "", "language": "unknown", "segments": []}
            finally:
                item.done_event.set()

    async def enqueue(self, audio_path: str, language: str = "pt", timeout_s: float = 120.0) -> dict[str, Any]:
        if self._loop_task is None:
            await self.start()
        item = _BatchItem(audio_path=audio_path, done_event=asyncio.Event(), language=language)
        self._queue.append(item)
        try:
            await asyncio.wait_for(item.done_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {"error": "batch_timeout", "text": "", "language": "unknown", "segments": []}
        return item.result


_BATCHERS: dict[str, BatchTranscriber] = {}
_BATCHERS_LOCK = asyncio.Lock()


async def get_or_create_batcher(model_size: str) -> BatchTranscriber:
    """Lazily create one batcher per model_size within this worker process."""
    async with _BATCHERS_LOCK:
        if model_size not in _BATCHERS:
            b = BatchTranscriber(model_size=model_size)
            await b.start()
            _BATCHERS[model_size] = b
        return _BATCHERS[model_size]
