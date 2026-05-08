# TODOs

Items deferred from active branches. Add a date and a rough owner when picking
something up. Move it under "Done" with a short outcome line when shipped.

## Diarization swap (from /plan-eng-review on feat/batching-ws-evals-validate-20260420)

### High priority

- **DX: surface `?diarizer=` in LiveConsole.tsx** — backend ships with API
  versioning; the live console UI still hardcodes the default. Add a dropdown
  alongside the model selector. Cheap follow-up once the API works.

- **Pyannote in-memory waveform refactor** — `app/services/diarization_session.py:282-293`
  writes `/tmp/*.wav` per chunk because `diarize_file_with_embeddings` requires a
  path. Save ~1.3 MB/chunk of disk I/O on every WS session even if pyannote stays.
  Deferred from /plan-eng-review D6 because the user explicitly chose not to invest
  in pyannote fairness.

- **Production audio eval set with consent + redaction** — D4 deferred. Hand-recorded
  internal sessions are v1; this is v2 once the consent flow and PII redaction
  tooling exist. ~2 weeks of legal/process work before any recording.

### Medium priority

- **DiariZen on the file path** (`/transcribe`) — research showed DiariZen beats
  pyannote 3.1 by ~4 DER points on average in batch mode. No streaming, so it
  belongs on the file path only. Defer until Sortformer is shipped on the live
  path and stable. https://arxiv.org/html/2506.13414v1

- **5+ speaker fallback to pyannote** — D3 chose visible `SPEAKER_UNKNOWN` over
  per-session backend fallback. If 5+ speaker sessions become a recurring need
  (hospital with 5+ staff, large meetings), wire detect-and-fallback. Or upgrade
  to a hypothetical Sortformer 8spk variant if NVIDIA ships one.

### Lower priority — Tier 3 tests deferred from D7

- Per-error-path tests on NeMo failures (loader exception, mid-inference CUDA OOM).
  Mock the failure and assert clean degradation.
- GPU memory accounting tests using `torch.cuda.memory_allocated()` deltas — verify
  long-running WS sessions don't leak.
- Concurrent multi-session stress tests (10 sessions in parallel, all making
  progress).
- `bench_diarizers.py` edge cases (zero-length audio, RTTM with overlapping
  segments, RTTM with single speaker).

### Open items still owned by design doc, not yet decided

- **NeMo SHA pinning** (Open Question 1) — needs a real spike: which exact NeMo
  commit installs cleanly on Python 3.12, CUDA 12.8, PyTorch 2.11. Could be 30
  minutes, could be 2 days. Blocks every other Sortformer task.
- **Pass criteria threshold** (Open Question 3) — what DER delta vs pyannote
  3.1 counts as "Sortformer wins"? Suggested ≥ 5 percentage points DER OR ≥ 2×
  RTF improvement at equal quality. Needs explicit owner sign-off before the
  benchmark runs.
