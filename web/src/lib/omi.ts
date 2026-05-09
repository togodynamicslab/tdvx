// Simulates an Omi-style client: records the mic, slices it into
// VAD-bounded speech bursts (not clock ticks), encodes each burst as WAV
// (16 kHz mono PCM16), and POSTs it to /transcribe-batch. Flushing on voice
// activity boundaries gives Pyannote a coherent per-speaker sample instead
// of a mid-utterance half-slice.

import { useCallback, useEffect, useRef, useState } from "react"

import type { ChunkTelemetry, ConnectionState, Metrics, TranscriptSegment } from "@/lib/ws"
import { VadChunker, type FlushReason } from "@/lib/vadChunker"

const TELEMETRY_LOG_LIMIT = 200

type OmiOptions = {
  model: string
  language?: "pt" | "en"       // default "pt"
  chunkSeconds?: number        // legacy: interpreted as maxChunkMs upper bound
  endpointPath?: string        // default "/transcribe-batch"
  diarize?: boolean            // default true — enables speaker tagging for clips >= diarizeMinSeconds
  diarizeMinSeconds?: number   // default 10 — short clips still skip Pyannote to stay fast
  // VAD tuning (all optional — the defaults in vadChunker.ts are sensible):
  minChunkMs?: number
  maxChunkMs?: number
  silenceMs?: number
}

const SAMPLE_RATE = 16_000

function floatToPcm16(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
  }
  return out
}

function encodeWav(pcm: Int16Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + pcm.length * 2)
  const view = new DataView(buffer)
  const writeString = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i))
  }
  const byteRate = sampleRate * 2
  writeString(0, "RIFF")
  view.setUint32(4, 36 + pcm.length * 2, true)
  writeString(8, "WAVE")
  writeString(12, "fmt ")
  view.setUint32(16, 16, true)       // fmt chunk size
  view.setUint16(20, 1, true)        // PCM
  view.setUint16(22, 1, true)        // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, 2, true)        // block align
  view.setUint16(34, 16, true)       // bits per sample
  writeString(36, "data")
  view.setUint32(40, pcm.length * 2, true)
  new Int16Array(buffer, 44).set(pcm)
  return new Blob([buffer], { type: "audio/wav" })
}

export function useOmiTranscription() {
  const [segments, setSegments] = useState<TranscriptSegment[]>([])
  const [state, setState] = useState<ConnectionState>("idle")
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics>({ segments: 0, words: 0, latencies: [] })
  const [telemetry, setTelemetry] = useState<ChunkTelemetry[]>([])
  const [sessionId, setSessionId] = useState<string>("")

  const ctxRef = useRef<AudioContext | null>(null)
  const srcRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunkerRef = useRef<VadChunker | null>(null)
  const runningRef = useRef<boolean>(false)
  const abortRef = useRef<AbortController | null>(null)
  const modelRef = useRef<string>("tdv1")
  const languageRef = useRef<"pt" | "en">("pt")
  const endpointRef = useRef<string>("/transcribe-batch")
  const diarizeRef = useRef<boolean>(true)
  const diarizeMinSecondsRef = useRef<number>(10)
  const chunkIndexRef = useRef<number>(0)
  const sessionIdRef = useRef<string>("")

  const flushChunk = useCallback(async (audio: Float32Array, reason: FlushReason) => {
    if (audio.length === 0) return
    const pcm = floatToPcm16(audio)
    const wav = encodeWav(pcm, SAMPLE_RATE)
    const thisIndex = chunkIndexRef.current++

    const fd = new FormData()
    fd.append("file", wav, `chunk-${thisIndex}.wav`)

    const sentAt = performance.now()
    try {
      const params = new URLSearchParams({
        model: modelRef.current,
        language: languageRef.current,
      })
      if (diarizeRef.current) {
        params.set("diarize", "true")
        params.set("diarize_min_seconds", String(diarizeMinSecondsRef.current))
        if (sessionIdRef.current) params.set("session_id", sessionIdRef.current)
      }
      const url = `${endpointRef.current}?${params.toString()}`
      const resp = await fetch(url, {
        method: "POST",
        body: fd,
        signal: abortRef.current?.signal,
      })
      if (!resp.ok) {
        const body = await resp.text()
        console.warn(`omi chunk ${thisIndex} → ${resp.status}`, body.slice(0, 200))
        return
      }
      const data = await resp.json()
      const now = performance.now()
      const latency = now - sentAt
      const segs: TranscriptSegment[] = (data.segments || []).map((s: {
        speaker?: string; start?: number; end?: number; text?: string; translation?: string
      }, i: number) => ({
        id: `omi-${thisIndex}-${i}`,
        speaker: s.speaker ?? "SPEAKER_00",
        text: s.text ?? "",
        translation: s.translation && s.translation !== s.text ? s.translation : undefined,
        start: s.start,
        end: s.end,
        receivedAt: now,
        latencyMs: latency,
      }))
      // Capture per-chunk telemetry — even if segments is empty (silence chunks
      // still tell us the model ran and what it found).
      if (data.telemetry) {
        const tel: ChunkTelemetry = {
          ...data.telemetry,
          receivedAt: now,
          chunkIndex: thisIndex,
          flushReason: reason,
        }
        setTelemetry((prev) => {
          const next = [...prev, tel]
          return next.length > TELEMETRY_LOG_LIMIT ? next.slice(-TELEMETRY_LOG_LIMIT) : next
        })
      }
      if (segs.length === 0) return
      setSegments((prev) => [...prev, ...segs])
      setMetrics((m) => ({
        ...m,
        segments: m.segments + segs.length,
        words: m.words + segs.reduce((acc, s) => acc + (s.text?.split(/\s+/).filter(Boolean).length ?? 0), 0),
        firstSegmentAt: m.firstSegmentAt ?? now,
        lastSegmentAt: now,
        latencies: [...m.latencies, latency],
      }))
    } catch (err) {
      if ((err as { name?: string })?.name === "AbortError") return
      console.warn(`omi chunk ${thisIndex} failed`, err)
    }
  }, [])

  const stop = useCallback(() => {
    runningRef.current = false
    // Drain any pending audio through the normal flush pipeline.
    try {
      chunkerRef.current?.flushNow("stop")
    } catch {
      /* noop */
    }
    abortRef.current?.abort()
    abortRef.current = null
    try {
      procRef.current?.disconnect()
      srcRef.current?.disconnect()
      streamRef.current?.getTracks().forEach((t) => t.stop())
      ctxRef.current?.close()
    } catch {
      /* noop */
    }
    procRef.current = null
    srcRef.current = null
    streamRef.current = null
    ctxRef.current = null
    chunkerRef.current = null
    setState("stopped")
  }, [])

  const start = useCallback(async (opts: OmiOptions) => {
    setError(null)
    setSegments([])
    setMetrics({ segments: 0, words: 0, latencies: [] })
    setTelemetry([])
    setState("connecting")
    modelRef.current = opts.model
    languageRef.current = opts.language ?? "pt"
    endpointRef.current = opts.endpointPath ?? "/transcribe-batch"
    diarizeRef.current = opts.diarize ?? true
    diarizeMinSecondsRef.current = opts.diarizeMinSeconds ?? 10
    chunkIndexRef.current = 0
    sessionIdRef.current = (typeof crypto !== "undefined" && crypto.randomUUID)
      ? crypto.randomUUID()
      : `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
    setSessionId(sessionIdRef.current)
    abortRef.current = new AbortController()

    // Derive VAD bounds. Treat the legacy chunkSeconds as a max cap hint if
    // the caller didn't pass explicit VAD params.
    const legacyCapMs = opts.chunkSeconds ? opts.chunkSeconds * 1000 : undefined
    const chunker = new VadChunker({
      sampleRate: SAMPLE_RATE,
      minChunkMs: opts.minChunkMs ?? 1500,
      maxChunkMs: opts.maxChunkMs ?? legacyCapMs ?? 8000,
      silenceMs: opts.silenceMs ?? 400,
    })
    chunker.setOnChunk((buf, reason) => {
      // Fire-and-forget — network IO happens outside the audio callback.
      void flushChunk(buf, reason)
    })
    chunkerRef.current = chunker

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: SAMPLE_RATE },
      })
      streamRef.current = stream

      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new AC({ sampleRate: SAMPLE_RATE })
      ctxRef.current = ctx

      const src = ctx.createMediaStreamSource(stream)
      srcRef.current = src
      const proc = ctx.createScriptProcessor(4096, 1, 1)
      procRef.current = proc
      src.connect(proc)
      proc.connect(ctx.destination)

      runningRef.current = true
      setMetrics((m) => ({ ...m, connectedAt: performance.now() }))

      proc.onaudioprocess = (e) => {
        if (!runningRef.current) return
        const ch = e.inputBuffer.getChannelData(0)
        // Copy since the underlying buffer is reused between callbacks.
        const copy = new Float32Array(ch.length)
        copy.set(ch)
        chunkerRef.current?.feed(copy)
      }

      setState("recording")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access denied")
      setState("error")
      stop()
    }
  }, [flushChunk, stop])

  useEffect(() => () => {
    runningRef.current = false
    abortRef.current?.abort()
  }, [])

  return { segments, state, error, metrics, telemetry, sessionId, start, stop, flushChunk }
}
