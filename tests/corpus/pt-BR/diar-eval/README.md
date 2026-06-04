# PT-BR Diarization Eval Set

Hand-labeled multi-speaker Portuguese audio for benchmarking diarization
backends (pyannote 3.1 vs NVIDIA Streaming Sortformer).

This corpus is **separate from `tests/corpus/pt-BR/`** which is single-speaker
voice clips for ASR fine-tuning. You cannot measure diarization on
single-speaker audio — that's why this directory exists.

## Layout

```
diar-eval/
  README.md                      ← this file
  rttm_helpers.py                ← RTTM ↔ Annotation converters used by bench
  manifest.json                  ← list of clips with metadata
  clips/
    {clip_id}.wav                ← 16 kHz mono float32 PCM, 3-6 minutes
    {clip_id}.rttm               ← ground truth speaker turns
    {clip_id}.json               ← per-clip metadata (recording context, speakers)
```

Aim for **30-60 minutes total**, **2-4 speakers per clip**, mix of:
- 2-speaker conversational (interview, doctor-patient)
- 3-speaker (family conversation, panel)
- 4-speaker (max for Sortformer v2.1)
- Hard cases: speaker overlap, fast turn-taking, code-switching to English brand
  names, varied recording quality

## RTTM format

The standard NIST RTTM format for ground-truth diarization:

```
SPEAKER {clip_id} 1 {start_seconds} {duration_seconds} <NA> <NA> {speaker_id} <NA> <NA>
```

Example:

```
SPEAKER doctor-patient-01 1 0.500 4.230 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER doctor-patient-01 1 4.730 2.110 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER doctor-patient-01 1 6.840 3.050 <NA> <NA> SPEAKER_00 <NA> <NA>
```

Speaker IDs are stable within a clip (`SPEAKER_00` is the same person every
time). They are **not** stable across clips — `SPEAKER_00` in clip A is not
the same person as `SPEAKER_00` in clip B.

## Labeling tools

Recommended: **Audacity** with the labels track export, then convert to RTTM
via `rttm_helpers.audacity_labels_to_rttm()`.

Alternative: **pyannote.audio's manual labeling** if you already use it.

Whichever tool: the RTTM goes here, source format gets discarded.

## Speaker labels

- `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`, `SPEAKER_03` — speakers in order
  of first appearance.
- No `SPEAKER_UNKNOWN` in ground truth. The `SPEAKER_UNKNOWN` label is a
  prediction-side concept (Sortformer's 5th+ speaker behavior); ground truth
  always names everyone.

## Per-clip metadata

`{clip_id}.json` schema:

```json
{
  "clip_id": "doctor-patient-01",
  "duration_s": 245.3,
  "n_speakers": 2,
  "speakers": [
    {"id": "SPEAKER_00", "role": "doctor", "gender": "F", "approx_age": 35},
    {"id": "SPEAKER_01", "role": "patient", "gender": "M", "approx_age": 60}
  ],
  "recording": {
    "source": "internal | synthetic | production-redacted",
    "device": "macbook-mic | zoom-recording | phone | ...",
    "noise_level": "clean | mild | noisy",
    "overlap": "none | mild | heavy"
  },
  "notes": "Free-form notes — language register, code-switching, tricky cases"
}
```

This metadata lets the bench script slice DER by recording quality / overlap
level / speaker count to find where each backend fails.

## What goes in this corpus

The eval-source decision is owned by the project lead (D4 in the
plan-eng-review). Until that's decided, this directory is the destination,
not the source. See `TODOS.md` → "Production audio eval set with consent +
redaction" for the v2 plan.

For v1, hand-recorded internal sessions are the recommended source: low
legal risk, fast to produce, you control the speaker mix.
