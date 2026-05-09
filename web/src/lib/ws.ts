import { useCallback, useEffect, useRef, useState } from "react"

import { VadChunker, type FlushReason } from "@/lib/vadChunker"

const TELEMETRY_LOG_LIMIT = 200

export type TranscriptSegment = {
  id: string
  speaker: string
  text: string
  translation?: string
  start?: number
  end?: number
  receivedAt: number
  latencyMs?: number
}

export type ConnectionState = "idle" | "connecting" | "recording" | "stopped" | "error"

export type Metrics = {
  segments: number
  words: number
  connectedAt?: number
  firstSegmentAt?: number
  lastSegmentAt?: number
  latencies: number[]
}

export type SpeakerResolution = {
  local_label: string
  global_label: string
  is_new: boolean
  distance: number | null
  duration_s: number
}

export type ChunkTelemetry = {
  chunk_duration_s: number
  whisper_ms: number
  diarization_ms: number
  embedding_ms: number
  alignment_ms: number
  total_ms: number
  diarize_ran: boolean
  locals_detected: number
  resolutions: SpeakerResolution[]
  registry_size: number
  model?: string | null
  worker?: string | null
  notes: string[]
  // Rolling-buffer diarization telemetry (optional; 0/undefined when buffer mode not in use)
  buffer_seconds?: number
  speakers_in_buffer?: number
  chunk_offset_seconds?: number
  flush_reason?: "vad" | "cap" | "stop" | null
  // Client-augmented fields (not on server response)
  receivedAt?: number
  chunkIndex?: number
  // Why the client decided to flush this chunk:
  //   "vad"  — trailing silence after voice activity (natural boundary)
  //   "cap"  — hit maxChunkMs hard cap (mid-utterance split)
  //   "stop" — user pressed stop, drain pending audio
  flushReason?: "vad" | "cap" | "stop"
}

type Options = {
  url: string
  language: "pt" | "en"
  model: string
  // VAD tuning (optional — defaults in vadChunker.ts are sensible)
  minChunkMs?: number
  maxChunkMs?: number
  silenceMs?: number
}

const wsUrl = (path: string) => {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
  return `${proto}//${window.location.host}${path}`
}

export const useTranscription = () => {
  const [segments, setSegments] = useState<TranscriptSegment[]>([])
  const [state, setState] = useState<ConnectionState>("idle")
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics>({ segments: 0, words: 0, latencies: [] })
  const [telemetry, setTelemetry] = useState<ChunkTelemetry[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const srcRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunkerRef = useRef<VadChunker | null>(null)
  const lastSendRef = useRef<number>(0)
  // FIFO of flushReason values for bursts we've sent but haven't seen
  // telemetry for yet. Each incoming telemetry event consumes one entry.
  const pendingFlushReasonsRef = useRef<FlushReason[]>([])
  const burstIndexRef = useRef<number>(0)

  const sendBurst = useCallback((buf: Float32Array, reason: FlushReason) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    // Server reads float32 PCM frames — send a single big buffer per burst.
    // Copy into a fresh ArrayBuffer so we don't ship the chunker's internal ref.
    const out = new Float32Array(buf.length)
    out.set(buf)
    ws.send(out.buffer)
    lastSendRef.current = performance.now()
    pendingFlushReasonsRef.current.push(reason)
    burstIndexRef.current += 1
  }, [])

  const stop = useCallback(() => {
    // Drain any pending audio through the normal send pipeline first.
    try {
      chunkerRef.current?.flushNow("stop")
    } catch {
      /* noop */
    }
    try {
      procRef.current?.disconnect()
      srcRef.current?.disconnect()
      streamRef.current?.getTracks().forEach((t) => t.stop())
      ctxRef.current?.close()
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send("end")
        wsRef.current.close()
      }
    } catch {
      /* noop */
    }
    procRef.current = null
    srcRef.current = null
    streamRef.current = null
    ctxRef.current = null
    chunkerRef.current = null
    wsRef.current = null
    setState("stopped")
  }, [])

  const start = useCallback(async (opts: Options) => {
    setError(null)
    setSegments([])
    setMetrics({ segments: 0, words: 0, latencies: [] })
    setTelemetry([])
    pendingFlushReasonsRef.current = []
    burstIndexRef.current = 0

    setState("connecting")

    const params = new URLSearchParams({
      language: opts.language,
      model: opts.model,
    })
    const ws = new WebSocket(`${wsUrl(opts.url)}?${params.toString()}`)
    wsRef.current = ws

    ws.onopen = async () => {
      setMetrics((m) => ({ ...m, connectedAt: performance.now() }))
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
        })
        streamRef.current = stream

        const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
        const ctx = new AC({ sampleRate: 16000 })
        ctxRef.current = ctx

        const src = ctx.createMediaStreamSource(stream)
        srcRef.current = src
        const proc = ctx.createScriptProcessor(4096, 1, 1)
        procRef.current = proc
        src.connect(proc)
        proc.connect(ctx.destination)

        const chunker = new VadChunker({
          sampleRate: 16000,
          minChunkMs: opts.minChunkMs ?? 1500,
          maxChunkMs: opts.maxChunkMs ?? 8000,
          silenceMs: opts.silenceMs ?? 400,
        })
        chunker.setOnChunk((buf, reason) => sendBurst(buf, reason))
        chunkerRef.current = chunker

        proc.onaudioprocess = (e) => {
          const ch = e.inputBuffer.getChannelData(0)
          // Copy: the underlying buffer is reused between callbacks.
          const copy = new Float32Array(ch.length)
          copy.set(ch)
          chunkerRef.current?.feed(copy)
        }

        setState("recording")
      } catch (err) {
        setError(err instanceof Error ? err.message : "Microphone access denied")
        setState("error")
        ws.close()
      }
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (data.error) {
          setError(data.error)
          setState("error")
          return
        }
        // Dedicated telemetry envelope from the server.
        if (data.telemetry) {
          const now = performance.now()
          const reason = pendingFlushReasonsRef.current.shift()
          const tel: ChunkTelemetry = {
            ...data.telemetry,
            receivedAt: now,
            chunkIndex: burstIndexRef.current - pendingFlushReasonsRef.current.length - 1,
            flushReason: reason,
          }
          setTelemetry((prev) => {
            const next = [...prev, tel]
            return next.length > TELEMETRY_LOG_LIMIT ? next.slice(-TELEMETRY_LOG_LIMIT) : next
          })
        }
        const s = data.segment ?? data
        if (!s.text && !s.speaker) return
        const now = performance.now()
        const lat = lastSendRef.current ? now - lastSendRef.current : undefined
        const seg: TranscriptSegment = {
          id: `${s.start ?? now}-${s.speaker ?? "?"}-${s.text?.slice(0, 16) ?? ""}`,
          speaker: s.speaker ?? "SPEAKER_00",
          text: s.text ?? "",
          translation: s.translation && s.translation !== s.text ? s.translation : undefined,
          start: s.start,
          end: s.end,
          receivedAt: now,
          latencyMs: lat,
        }
        setSegments((prev) => [...prev, seg])
        setMetrics((m) => ({
          ...m,
          segments: m.segments + 1,
          words: m.words + (seg.text?.split(/\s+/).filter(Boolean).length ?? 0),
          firstSegmentAt: m.firstSegmentAt ?? now,
          lastSegmentAt: now,
          latencies: lat != null ? [...m.latencies, lat] : m.latencies,
        }))
      } catch (err) {
        console.error("parse", err)
      }
    }

    ws.onerror = () => {
      setError("Connection error")
      setState("error")
    }

    ws.onclose = () => {
      setState((s) => (s === "recording" ? "stopped" : s))
    }
  }, [sendBurst])

  useEffect(() => () => stop(), [stop])

  return { segments, state, error, metrics, telemetry, start, stop }
}

export const pct = (arr: number[], p: number) => {
  if (!arr.length) return 0
  const s = [...arr].sort((a, b) => a - b)
  const idx = Math.min(s.length - 1, Math.floor((p / 100) * s.length))
  return s[idx]
}
