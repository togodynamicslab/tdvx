import { useMemo, useState } from "react"
import { Mic, Square, Loader2, Waves, Activity, Languages, Cpu, Gauge, Clock, Radio, Package } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
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
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation"

import { pct, useTranscription, type TranscriptSegment } from "@/lib/ws"
import { useOmiTranscription } from "@/lib/omi"
import { TelemetryPanel } from "@/components/TelemetryPanel"

const MODELS = [
  { id: "tdv1-fast", name: "TDv1-Fast", hint: "faster-whisper small · live" },
  { id: "tdv1-medium", name: "TDv1-Medium", hint: "faster-whisper medium · balanced" },
  { id: "tdv1", name: "TDv1", hint: "faster-whisper large-v3 · highest quality" },
  { id: "tdv1-cv-pt", name: "TDv1-CV-PT", hint: "fine-tuned medium · CommonVoice PT (CT2 int8)" },
  { id: "tdv3-cv-pt", name: "TDv3-CV-PT", hint: "fine-tuned medium · CommonVoice PT v3 (CT2 int8)" },
]

const LANGS = [
  { code: "pt", label: "Português (pt-BR)" },
  { code: "en", label: "English (en-US)" },
] as const

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

const formatMs = (v?: number) => (v == null ? "—" : `${(v / 1000).toFixed(2)}s`)

const TranscriptCard = ({ seg }: { seg: TranscriptSegment }) => (
  <div className="group rounded-lg border bg-card px-4 py-3 shadow-xs transition-colors hover:border-foreground/20">
    <div className="flex items-baseline justify-between gap-3">
      <Badge
        variant="outline"
        className={`font-mono text-[10px] tracking-wider ${speakerColor(seg.speaker)}`}
      >
        {seg.speaker}
      </Badge>
      <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
        {seg.start != null && (
          <span>
            {seg.start.toFixed(1)}s–{seg.end?.toFixed(1)}s
          </span>
        )}
        {seg.latencyMs != null && (
          <span title="round-trip from last audio byte sent">
            <Clock className="mr-1 inline size-3" />
            {formatMs(seg.latencyMs)}
          </span>
        )}
      </div>
    </div>
    <p className="mt-2 text-[15px] leading-relaxed text-foreground">{seg.text}</p>
    {seg.translation && (
      <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
        → {seg.translation}
      </p>
    )}
  </div>
)

const StatBlock = ({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  hint?: string
}) => (
  <div className="rounded-lg border bg-card p-4">
    <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
      <Icon className="size-3.5" />
      {label}
    </div>
    <div className="mt-2 font-mono text-2xl font-medium tabular-nums">{value}</div>
    {hint && <div className="mt-1 text-[11px] text-muted-foreground">{hint}</div>}
  </div>
)

type Mode = "live" | "omi"

export default function LiveConsole() {
  const [mode, setMode] = useState<Mode>("omi")
  const [model, setModel] = useState("tdv1")
  const [lang, setLang] = useState<"pt" | "en">("pt")
  const liveHook = useTranscription()
  const omiHook = useOmiTranscription()
  const active = mode === "omi" ? omiHook : liveHook
  const { segments, state, error, metrics, stop } = active

  const isRecording = state === "recording"
  const isConnecting = state === "connecting"

  const stats = useMemo(() => {
    const elapsed =
      metrics.connectedAt && metrics.lastSegmentAt
        ? (metrics.lastSegmentAt - metrics.connectedAt) / 1000
        : 0
    const ttfr =
      metrics.connectedAt && metrics.firstSegmentAt
        ? metrics.firstSegmentAt - metrics.connectedAt
        : undefined
    const latP50 = pct(metrics.latencies, 50)
    const latP95 = pct(metrics.latencies, 95)
    const wpm = elapsed > 0 ? (metrics.words / elapsed) * 60 : 0
    return { elapsed, ttfr, latP50, latP95, wpm }
  }, [metrics])

  const onToggle = () => {
    if (isRecording) {
      stop()
      return
    }
    // Stop the OTHER mode first in case it's mid-connect.
    if (mode === "omi") {
      liveHook.stop()
      omiHook.start({ model, language: lang, endpointPath: "/transcribe-batch", chunkSeconds: 8, diarize: true, diarizeMinSeconds: 0 })
    } else {
      omiHook.stop()
      liveHook.start({ url: "/ws/transcribe", language: lang, model })
    }
  }

  return (
    <>
      <div className="mx-auto flex max-w-7xl items-center justify-end px-6 pt-3">
        <Badge
          variant="outline"
          className={`font-mono text-[10px] ${
            state === "recording"
              ? "border-emerald-500/50 text-emerald-400"
              : state === "error"
                ? "border-destructive/60 text-destructive"
                : "text-muted-foreground"
          }`}
        >
          <span
            className={`mr-1.5 inline-block size-1.5 rounded-full ${
              state === "recording"
                ? "animate-pulse bg-emerald-400"
                : state === "error"
                  ? "bg-destructive"
                  : "bg-muted-foreground"
            }`}
          />
          {state.toUpperCase()}
        </Badge>
      </div>
      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[320px_1fr]">
        <aside className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                Session
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Radio className="size-3.5" />
                  Mode
                </div>
                <div className="flex gap-1 rounded-md border p-0.5">
                  <button
                    type="button"
                    disabled={isRecording || isConnecting}
                    onClick={() => setMode("omi")}
                    className={`flex-1 rounded px-2 py-1.5 text-xs font-mono transition-colors ${
                      mode === "omi"
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground"
                    } disabled:cursor-not-allowed disabled:opacity-40`}
                  >
                    <Package className="mr-1 inline size-3" />
                    Omi (batched)
                  </button>
                  <button
                    type="button"
                    disabled={isRecording || isConnecting}
                    onClick={() => setMode("live")}
                    className={`flex-1 rounded px-2 py-1.5 text-xs font-mono transition-colors ${
                      mode === "live"
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground"
                    } disabled:cursor-not-allowed disabled:opacity-40`}
                  >
                    <Radio className="mr-1 inline size-3" />
                    Live (WS)
                  </button>
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {mode === "omi"
                    ? "8s chunks → /transcribe-batch + per-session speaker registry. Simulates Omi."
                    : "Continuous WebSocket. Streams audio live."}
                </div>
              </div>

              <Separator />

              <div className="space-y-1.5">
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Languages className="size-3.5" />
                  Language
                </label>
                <Select value={lang} onValueChange={(v) => setLang(v as "pt" | "en")}>
                  <SelectTrigger className="w-full">
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
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Cpu className="size-3.5" />
                  Model
                </label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MODELS.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        <div className="flex flex-col">
                          <span className="font-mono text-sm">{m.name}</span>
                          <span className="text-[11px] text-muted-foreground">
                            {m.hint}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <Button
                size="lg"
                variant={isRecording ? "destructive" : "default"}
                onClick={onToggle}
                disabled={isConnecting}
                className="w-full gap-2 font-mono"
              >
                {isConnecting ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : isRecording ? (
                  <Square className="size-4" />
                ) : (
                  <Mic className="size-4" />
                )}
                {isConnecting
                  ? "Connecting…"
                  : isRecording
                    ? "Stop recording"
                    : "Start recording"}
              </Button>

              {error && (
                <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                Metrics
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3">
              <StatBlock
                icon={Clock}
                label="TTFR"
                value={formatMs(stats.ttfr)}
                hint="time-to-first-response"
              />
              <StatBlock
                icon={Gauge}
                label="Lat p95"
                value={formatMs(stats.latP95)}
                hint={`p50 ${formatMs(stats.latP50)}`}
              />
              <StatBlock
                icon={Activity}
                label="Segments"
                value={metrics.segments.toString()}
                hint={`${metrics.words} words`}
              />
              <StatBlock
                icon={Waves}
                label="Rate"
                value={`${stats.wpm.toFixed(0)}`}
                hint="words / min"
              />
            </CardContent>
          </Card>
        </aside>

        <section className="flex min-h-[calc(100vh-120px)] flex-col">
          <Card className="flex flex-1 flex-col overflow-hidden p-0">
            <CardHeader className="border-b px-6 py-4">
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-sm">Transcript</CardTitle>
                <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                  <span>{segments.length} seg</span>
                  <span>·</span>
                  <span>{metrics.words} words</span>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative flex-1 overflow-hidden p-0">
              <Conversation className="h-[calc(100vh-240px)]">
                <ConversationContent className="gap-3">
                  {segments.length === 0 ? (
                    <ConversationEmptyState
                      icon={<Mic className="size-8 text-muted-foreground/60" />}
                      title="Ready when you are"
                      description={
                        state === "recording"
                          ? "Listening… speak into your microphone."
                          : `Click 'Start recording' to stream audio to ${model}.`
                      }
                    />
                  ) : (
                    segments.map((s) => <TranscriptCard key={s.id} seg={s} />)
                  )}
                </ConversationContent>
                <ConversationScrollButton />
              </Conversation>
            </CardContent>
          </Card>

          {mode === "omi" && (
            <div className="mt-4">
              <TelemetryPanel events={omiHook.telemetry} sessionId={omiHook.sessionId} />
            </div>
          )}
        </section>
      </main>
    </>
  )
}
