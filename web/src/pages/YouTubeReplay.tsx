import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Play, Square } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation"
import { TelemetryPanel } from "@/components/TelemetryPanel"
import type { ChunkTelemetry, TranscriptSegment } from "@/lib/ws"

// Keep this list in sync with LiveConsole's MODELS — when one moves, the
// other should too. They both depend on the backend's /models endpoint.
const MODELS = [
  { id: "tdv1-fast", name: "TDv1-Fast", hint: "faster-whisper small · live" },
  { id: "tdv1-medium", name: "TDv1-Medium", hint: "faster-whisper medium · balanced" },
  { id: "tdv1", name: "TDv1", hint: "faster-whisper large-v3 · highest quality" },
  { id: "tdv1-cv-pt", name: "TDv1-CV-PT", hint: "fine-tuned medium · CommonVoice PT (CT2 int8)" },
  { id: "tdv3-cv-pt", name: "TDv3-CV-PT", hint: "fine-tuned medium · CommonVoice PT v3 (CT2 int8)" },
]

const LANGS = [
  { code: "pt" as const, label: "Português (pt-BR)" },
  { code: "en" as const, label: "English (en-US)" },
]

const DIARIZERS = [
  { id: "pyannote" as const, name: "Pyannote 4.0", hint: "production · ~2.4s/window · unbounded speakers" },
  { id: "sortformer" as const, name: "Sortformer v2.1", hint: "NVIDIA streaming · ~100ms/window · capped at 4 speakers" },
]

const SPEEDS = [1, 2, 4, 8, 16] as const

// YouTube exposes 11-character video IDs. Capture the ID from any of the
// common URL shapes so the iframe gets the right embed source.
const YT_ID_RE =
  /(?:youtube\.com\/(?:watch\?(?:.*&)?v=|shorts\/|embed\/|v\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/

function extractVideoId(input: string): string | null {
  const trimmed = input.trim()
  if (!trimmed) return null
  const match = trimmed.match(YT_ID_RE)
  if (match) return match[1]
  if (/^[A-Za-z0-9_-]{11}$/.test(trimmed)) return trimmed
  return null
}

const speakerColor = (speaker: string) => {
  const n = Number.parseInt(speaker.replace(/\D/g, ""), 10) || 0
  const palette = [
    "bg-emerald-500/10 text-emerald-400 border-emerald-500/40",
    "bg-sky-500/10 text-sky-400 border-sky-500/40",
    "bg-violet-500/10 text-violet-400 border-violet-500/40",
    "bg-amber-500/10 text-amber-400 border-amber-500/40",
    "bg-rose-500/10 text-rose-400 border-rose-500/40",
    "bg-cyan-500/10 text-cyan-400 border-cyan-500/40",
  ]
  return palette[n % palette.length]
}

// Transcript segment shape extended with absolute video offsets the backend
// computes from each chunk's start time. Lets us click-to-seek the embed.
type YouTubeSegment = TranscriptSegment & {
  videoStartS?: number
  videoEndS?: number
}

type Header = {
  videoId: string
  title: string
  durationS: number
  channel: string
  model: string
  speed: number
  chunkSeconds: number
}

type Status = "idle" | "extracting" | "ready" | "streaming" | "done" | "error"

const formatTime = (s?: number) =>
  s == null
    ? "—"
    : `${Math.floor(s / 60)
        .toString()
        .padStart(2, "0")}:${Math.floor(s % 60)
        .toString()
        .padStart(2, "0")}`

export default function YouTubeReplay() {
  const [url, setUrl] = useState("")
  const [model, setModel] = useState("tdv1-fast")
  const [lang, setLang] = useState<"pt" | "en">("pt")
  const [speed, setSpeed] = useState<number>(1)
  const [diarize, setDiarize] = useState(true)
  const [diarizer, setDiarizer] = useState<"pyannote" | "sortformer">("pyannote")

  const [status, setStatus] = useState<Status>("idle")
  const [error, setError] = useState<string | null>(null)
  const [header, setHeader] = useState<Header | null>(null)
  const [segments, setSegments] = useState<YouTubeSegment[]>([])
  const [telemetry, setTelemetry] = useState<ChunkTelemetry[]>([])
  const [progressS, setProgressS] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string>(`yt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`)
  const iframeRef = useRef<HTMLIFrameElement | null>(null)

  const videoId = useMemo(() => extractVideoId(url), [url])

  const isExtracting = status === "extracting"
  const isStreaming = status === "streaming"
  const canExtract = !!videoId && !isExtracting && !isStreaming
  const canStream = status === "ready" && !!header

  // Compute progress as a fraction so the bar renders the same regardless of
  // total duration. Avoid divide-by-zero when the header hasn't arrived yet.
  const progressPct = useMemo(() => {
    if (!header || !header.durationS) return 0
    return Math.min(100, (progressS / header.durationS) * 100)
  }, [progressS, header])

  // YouTube IFrame Player API speaks postMessage when the iframe URL has
  // ?enablejsapi=1. Keep this dep-free instead of pulling in their JS API.
  // Defined up here because handleStream / stopStream below close over it.
  const ytCommand = useCallback((func: string, args: unknown[] = []) => {
    const iframe = iframeRef.current
    if (!iframe || !iframe.contentWindow) return
    iframe.contentWindow.postMessage(
      JSON.stringify({ event: "command", func, args }),
      "*",
    )
  }, [])

  const handleExtract = useCallback(async () => {
    if (!videoId) return
    setError(null)
    setStatus("extracting")
    setHeader(null)
    setSegments([])
    setTelemetry([])
    setProgressS(0)
    sessionIdRef.current = `yt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

    try {
      const res = await fetch("/youtube/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail || (await res.text().catch(() => ""))
        throw new Error(detail || `extract failed: ${res.status}`)
      }
      const meta = await res.json()
      setHeader({
        videoId: meta.video_id,
        title: meta.title,
        durationS: meta.duration_s,
        channel: meta.channel,
        model,
        speed,
        chunkSeconds: 2.0,
      })
      setStatus("ready")
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setStatus("error")
    }
  }, [url, videoId, model, speed])

  const stopStream = useCallback(() => {
    const ws = wsRef.current
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      try {
        ws.close()
      } catch {
        // ignore
      }
    }
    wsRef.current = null
    // Pause the embed when the user explicitly stops streaming. Don't auto-mute;
    // user already chose this state by clicking play, leave it where they had it.
    ytCommand("pauseVideo")
  }, [ytCommand])

  const handleStream = useCallback(() => {
    if (!header) return
    setError(null)
    setSegments([])
    setTelemetry([])
    setProgressS(0)
    setStatus("streaming")

    // Sync embed playback with the streaming pipeline. Seek to 0, unmute,
    // and play. At speed=1 this stays in sync with the chunk pacing on the
    // server. At speed>1 the embed will drift behind the transcript — that's
    // expected and called out in the speed-selector helper text.
    ytCommand("seekTo", [0, true])
    ytCommand("unMute")
    ytCommand("playVideo")

    // Same-origin WS so the Vite proxy / FastAPI mount handles routing.
    const wsProto = window.location.protocol === "https:" ? "wss" : "ws"
    const params = new URLSearchParams({
      url,
      model,
      language: lang,
      speed: String(speed),
      diarize: String(diarize),
      diarizer,
      session_id: sessionIdRef.current,
    })
    const wsUrl = `${wsProto}://${window.location.host}/youtube/stream?${params.toString()}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.error) {
          setError(msg.error)
          return
        }
        if (msg.type === "header") {
          setHeader((prev) =>
            prev
              ? { ...prev, model: msg.model, speed: msg.speed, chunkSeconds: msg.chunk_seconds }
              : prev,
          )
          return
        }
        if (msg.type === "eos") {
          setStatus("done")
          return
        }
        // Chunk shape mirrors LiveTranscriptionChunk + our youtube_offset extras.
        if (msg.segment) {
          const seg = msg.segment
          const next: YouTubeSegment = {
            id: `${msg.chunk_id ?? Date.now()}-${seg.speaker}-${seg.start ?? 0}`,
            speaker: seg.speaker || "SPEAKER_00",
            text: seg.text || "",
            translation: seg.translation || undefined,
            start: seg.start,
            end: seg.end,
            videoStartS: seg.video_start_s,
            videoEndS: seg.video_end_s,
            receivedAt: Date.now(),
            latencyMs: msg.pipeline_ms,
          }
          setSegments((prev) => [...prev, next])
        }
        if (typeof msg.video_chunk_end_s === "number") {
          setProgressS(msg.video_chunk_end_s)
        }
        if (msg.telemetry) {
          setTelemetry((prev) => [
            ...prev.slice(-199),
            { ...msg.telemetry, receivedAt: Date.now(), chunkIndex: prev.length + 1 },
          ])
        }
      } catch (e) {
        console.warn("youtube/stream parse error", e)
      }
    }

    ws.onerror = () => {
      setError("WebSocket error")
      setStatus("error")
    }
    ws.onclose = () => {
      // If we never finished naturally, mark done so the UI shows a final state.
      setStatus((prev) => (prev === "streaming" ? "done" : prev))
      wsRef.current = null
    }
  }, [header, url, model, lang, speed, diarize, diarizer, ytCommand])

  // If the user navigates away mid-stream, drop the WS so we don't leak.
  useEffect(() => () => stopStream(), [stopStream])

  // Click-to-seek: jump the YouTube embed to a segment's start time and play.
  const seekTo = useCallback((seconds: number) => {
    ytCommand("seekTo", [seconds, true])
    ytCommand("playVideo")
  }, [ytCommand])

  return (
    <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[360px_1fr]">
      {/* Left rail — controls + telemetry */}
      <aside className="space-y-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              YouTube replay
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Video URL</label>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                disabled={isExtracting || isStreaming}
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 font-mono text-xs shadow-xs transition-colors placeholder:text-muted-foreground focus-visible:outline-hidden focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
              <div className="text-[11px] font-mono text-muted-foreground">
                {videoId ? `id=${videoId}` : url ? "not a YouTube URL" : "paste any youtube.com / youtu.be link"}
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Model</label>
              <Select value={model} onValueChange={setModel} disabled={isStreaming}>
                <SelectTrigger className="w-full font-mono text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODELS.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      <div className="flex flex-col">
                        <span>{m.name}</span>
                        <span className="text-[10px] text-muted-foreground">{m.hint}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Language</label>
              <Select value={lang} onValueChange={(v) => setLang(v as "pt" | "en")} disabled={isStreaming}>
                <SelectTrigger className="w-full font-mono text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LANGS.map((l) => (
                    <SelectItem key={l.code} value={l.code}>
                      {l.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Diarizer</label>
              <Select
                value={diarizer}
                onValueChange={(v) => setDiarizer(v as "pyannote" | "sortformer")}
                disabled={isStreaming}
              >
                <SelectTrigger className="w-full font-mono text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DIARIZERS.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      <div className="flex flex-col">
                        <span>{d.name}</span>
                        <span className="text-[10px] text-muted-foreground">{d.hint}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Speed</span>
                <span className="font-mono">{speed}×</span>
              </div>
              <div className="flex gap-1">
                {SPEEDS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    disabled={isStreaming}
                    onClick={() => setSpeed(s)}
                    className={`flex-1 rounded-md border px-2 py-1 font-mono text-[11px] transition-colors ${
                      speed === s
                        ? "border-foreground bg-foreground text-background"
                        : "border-border text-muted-foreground hover:border-foreground/40 hover:text-foreground"
                    } ${isStreaming ? "cursor-not-allowed opacity-50" : ""}`}
                  >
                    {s}×
                  </button>
                ))}
              </div>
              <div className="text-[10px] text-muted-foreground">
                1× = real-time dogfood, higher = QA throughput
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                id="yt-diarize"
                checked={diarize}
                onChange={(e) => setDiarize(e.target.checked)}
                disabled={isStreaming}
                className="size-3.5"
              />
              <label htmlFor="yt-diarize">Diarize speakers</label>
            </div>

            <div className="flex flex-col gap-2 pt-1">
              <Button onClick={handleExtract} disabled={!canExtract} variant="default" size="sm">
                {isExtracting ? (
                  <>
                    <Loader2 className="mr-2 size-3.5 animate-spin" /> Extracting…
                  </>
                ) : status === "ready" || status === "done" || status === "error" ? (
                  "Re-extract"
                ) : (
                  "1. Extract audio"
                )}
              </Button>
              {!isStreaming ? (
                <Button onClick={handleStream} disabled={!canStream} variant="secondary" size="sm">
                  <Play className="mr-2 size-3.5" /> 2. Stream transcription
                </Button>
              ) : (
                <Button onClick={stopStream} variant="destructive" size="sm">
                  <Square className="mr-2 size-3.5" /> Stop
                </Button>
              )}
            </div>

            {header && (
              <div className="rounded-md border bg-muted/30 p-3 font-mono text-[11px] leading-relaxed">
                <div className="truncate text-foreground" title={header.title}>
                  {header.title || "(untitled)"}
                </div>
                <div className="text-muted-foreground">{header.channel}</div>
                <div className="mt-1 text-muted-foreground">
                  {formatTime(progressS)} / {formatTime(header.durationS)}
                </div>
                <div className="mt-1 h-1 w-full overflow-hidden rounded bg-muted">
                  <div
                    className="h-full bg-foreground/70 transition-[width] duration-150"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-md border border-destructive/60 bg-destructive/5 p-2 font-mono text-[11px] text-destructive">
                {error}
              </div>
            )}
          </CardContent>
        </Card>
      </aside>

      {/* Right column — embed + transcript + telemetry */}
      <section className="space-y-4">
        <Card>
          <CardContent className="p-0">
            {videoId ? (
              <div className="aspect-video w-full overflow-hidden rounded-md bg-black">
                <iframe
                  ref={iframeRef}
                  // mute=1 keeps the embed quiet until the user explicitly
                  // hits "Stream transcription" — at which point we fire
                  // unMute + playVideo via postMessage so audio + transcript
                  // start together. Click-to-seek into a segment also unmutes
                  // implicitly via playVideo.
                  src={`https://www.youtube.com/embed/${videoId}?enablejsapi=1&rel=0&mute=1`}
                  title="YouTube video player"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="size-full"
                />
              </div>
            ) : (
              <div className="flex aspect-video w-full items-center justify-center bg-muted/30 font-mono text-xs text-muted-foreground">
                Paste a YouTube URL to load the player.
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="min-h-[40vh]">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Transcript
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-[10px]">
                {segments.length} segments
              </Badge>
              <Badge
                variant="outline"
                className={`font-mono text-[10px] ${
                  status === "streaming"
                    ? "border-emerald-500/50 text-emerald-400"
                    : status === "error"
                      ? "border-destructive/60 text-destructive"
                      : "text-muted-foreground"
                }`}
              >
                {status.toUpperCase()}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <Conversation className="h-[55vh]">
              <ConversationContent>
                {segments.length === 0 ? (
                  <div className="flex h-full items-center justify-center font-mono text-xs text-muted-foreground">
                    {status === "streaming"
                      ? "Streaming…"
                      : status === "ready"
                        ? "Ready. Click 'Stream transcription' to begin."
                        : "No transcript yet."}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {segments.map((seg) => (
                      <button
                        key={seg.id}
                        onClick={() => seg.videoStartS != null && seekTo(seg.videoStartS)}
                        className="group block w-full rounded-lg border bg-card px-4 py-3 text-left shadow-xs transition-colors hover:border-foreground/20"
                      >
                        <div className="flex items-baseline justify-between gap-3">
                          <Badge
                            variant="outline"
                            className={`font-mono text-[10px] tracking-wider ${speakerColor(seg.speaker)}`}
                          >
                            {seg.speaker}
                          </Badge>
                          <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
                            {seg.videoStartS != null && (
                              <span>
                                {formatTime(seg.videoStartS)}–{formatTime(seg.videoEndS)}
                              </span>
                            )}
                            {seg.latencyMs != null && (
                              <span title="pipeline processing time for this chunk">
                                {(seg.latencyMs / 1000).toFixed(2)}s
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="mt-2 text-sm leading-relaxed text-foreground">{seg.text}</div>
                        {seg.translation && (
                          <div className="mt-1 text-xs italic text-muted-foreground">{seg.translation}</div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </ConversationContent>
              <ConversationScrollButton />
            </Conversation>
          </CardContent>
        </Card>

        <TelemetryPanel events={telemetry} sessionId={sessionIdRef.current} />
      </section>
    </main>
  )
}
