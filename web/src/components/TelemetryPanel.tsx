import { useMemo, useState } from "react"
import { Activity, ChevronDown, ChevronRight, Cpu, Database, Sparkles, Users } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import type { ChunkTelemetry } from "@/lib/ws"

const speakerColor = (label: string) => {
  const n = Number.parseInt(label.replace(/\D/g, ""), 10) || 0
  const palette = [
    "border-emerald-500/40 text-emerald-400",
    "border-sky-500/40 text-sky-400",
    "border-violet-500/40 text-violet-400",
    "border-amber-500/40 text-amber-400",
    "border-rose-500/40 text-rose-400",
    "border-cyan-500/40 text-cyan-400",
  ]
  return palette[n % palette.length]
}

const fmt = (n: number, d = 0) => n.toFixed(d)
const fmtMs = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${n}ms`)

const flushBadgeClass = (reason: string) => {
  switch (reason) {
    case "vad":
      return "border-emerald-500/50 text-emerald-400"
    case "cap":
      return "border-amber-500/50 text-amber-400"
    case "stop":
      return "border-muted-foreground/40 text-muted-foreground"
    default:
      return "border-muted-foreground/30 text-muted-foreground"
  }
}

type Aggregates = {
  total: number
  diarized: number
  totalLocals: number
  newSpeakerEvents: number
  registryMax: number
  whisperP50: number
  diarP50: number
  embedP50: number
  totalP50: number
  distSamples: number[]
  distMin: number
  distMax: number
  distMean: number
  bufferP50: number
  bufferSamples: number
  speakersInBufferMax: number
}

const percentile = (arr: number[], p: number): number => {
  if (arr.length === 0) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * p))
  return sorted[idx]
}

const computeAggregates = (events: ChunkTelemetry[]): Aggregates => {
  const whisperT: number[] = []
  const diarT: number[] = []
  const embedT: number[] = []
  const totalT: number[] = []
  const dists: number[] = []
  const bufferT: number[] = []
  let diarized = 0
  let totalLocals = 0
  let newSpeakerEvents = 0
  let registryMax = 0
  let speakersInBufferMax = 0
  for (const e of events) {
    whisperT.push(e.whisper_ms)
    if (e.diarize_ran) {
      diarized += 1
      diarT.push(e.diarization_ms)
      embedT.push(e.embedding_ms)
    }
    totalT.push(e.total_ms)
    totalLocals += e.locals_detected
    registryMax = Math.max(registryMax, e.registry_size)
    if (e.buffer_seconds != null && e.buffer_seconds > 0) bufferT.push(e.buffer_seconds)
    if (e.speakers_in_buffer != null) speakersInBufferMax = Math.max(speakersInBufferMax, e.speakers_in_buffer)
    for (const r of e.resolutions) {
      if (r.distance != null) dists.push(r.distance)
      if (r.is_new) newSpeakerEvents += 1
    }
  }
  return {
    total: events.length,
    diarized,
    totalLocals,
    newSpeakerEvents,
    registryMax,
    whisperP50: percentile(whisperT, 0.5),
    diarP50: percentile(diarT, 0.5),
    embedP50: percentile(embedT, 0.5),
    totalP50: percentile(totalT, 0.5),
    distSamples: dists,
    distMin: dists.length ? Math.min(...dists) : 0,
    distMax: dists.length ? Math.max(...dists) : 0,
    distMean: dists.length ? dists.reduce((a, b) => a + b, 0) / dists.length : 0,
    bufferP50: percentile(bufferT, 0.5),
    bufferSamples: bufferT.length,
    speakersInBufferMax,
  }
}

const DistanceHistogram = ({ values }: { values: number[] }) => {
  // 10 buckets across [0, 2] cosine distance range.
  const buckets = useMemo(() => {
    const bins = new Array(10).fill(0) as number[]
    for (const v of values) {
      const idx = Math.min(9, Math.max(0, Math.floor((v / 2) * 10)))
      bins[idx] += 1
    }
    return bins
  }, [values])
  const max = Math.max(...buckets, 1)
  return (
    <div className="flex h-16 items-end gap-px">
      {buckets.map((c, i) => (
        <div key={i} className="flex-1 bg-muted-foreground/15 transition-colors hover:bg-muted-foreground/40"
             style={{ height: `${(c / max) * 100}%` }}
             title={`${(i * 0.2).toFixed(1)}–${((i + 1) * 0.2).toFixed(1)}: ${c}`} />
      ))}
    </div>
  )
}

const ChunkRow = ({ e }: { e: ChunkTelemetry }) => {
  const [open, setOpen] = useState(false)
  const stages = [
    { name: "whisper", ms: e.whisper_ms, color: "bg-sky-500/60" },
    { name: "diar", ms: e.diarization_ms, color: "bg-violet-500/60" },
    { name: "embed", ms: e.embedding_ms, color: "bg-amber-500/60" },
    { name: "align", ms: e.alignment_ms, color: "bg-emerald-500/60" },
  ]
  const totalStages = stages.reduce((a, s) => a + s.ms, 0) || 1
  const isMixed = e.locals_detected > 1
  const flushReason = e.flush_reason ?? e.flushReason ?? null
  const bufferSeconds = e.buffer_seconds ?? 0
  return (
    <div
      className={`rounded-md border bg-card/50 px-3 py-2 ${
        isMixed ? "border-l-2 border-l-violet-500/40" : ""
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          {open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
          <span className="text-foreground">#{e.chunkIndex ?? "?"}</span>
          <span>{fmt(e.chunk_duration_s, 1)}s in</span>
          <span>·</span>
          <span>{fmtMs(e.total_ms)} total</span>
          {!e.diarize_ran && <Badge variant="outline" className="ml-1 border-amber-500/40 text-[10px] text-amber-400">no-diar</Badge>}
          {flushReason && (
            <Badge variant="outline" className={`text-[10px] ${flushBadgeClass(flushReason)}`}>
              {flushReason}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
          <span>locals {e.locals_detected}</span>
          {bufferSeconds > 0 && (
            <>
              <span>·</span>
              <span>buf {bufferSeconds.toFixed(1)}s</span>
            </>
          )}
          <span>·</span>
          <span>reg {e.registry_size}</span>
          {e.resolutions.length > 0 && (
            <div className="flex gap-1">
              {e.resolutions.map((r, i) => (
                <Badge
                  key={i}
                  variant="outline"
                  className={`text-[10px] ${speakerColor(r.global_label)} ${r.is_new ? "ring-1 ring-amber-500/60" : ""}`}
                >
                  {r.global_label.replace("SPEAKER_", "S")}
                  {r.distance != null && (
                    <span className="ml-1 opacity-60">{r.distance.toFixed(2)}</span>
                  )}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </button>
      {open && (
        <div className="mt-2 space-y-2 border-t pt-2">
          <div className="flex h-2 w-full overflow-hidden rounded-sm bg-muted">
            {stages.map((s) => s.ms > 0 && (
              <div
                key={s.name}
                className={s.color}
                style={{ width: `${(s.ms / totalStages) * 100}%` }}
                title={`${s.name}: ${s.ms}ms`}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted-foreground">
            {stages.map((s) => (
              <span key={s.name}>
                {s.name}: <span className="text-foreground">{s.ms}ms</span>
              </span>
            ))}
            {e.model && <span>model: <span className="text-foreground">{e.model}</span></span>}
            {e.worker && <span>worker: <span className="text-foreground">{e.worker}</span></span>}
            {bufferSeconds > 0 && (
              <span>
                buffer: <span className="text-foreground">{bufferSeconds.toFixed(2)}s</span>
              </span>
            )}
            {e.speakers_in_buffer != null && (
              <span>
                speakers_in_buffer: <span className="text-foreground">{e.speakers_in_buffer}</span>
              </span>
            )}
            {e.chunk_offset_seconds != null && e.chunk_offset_seconds > 0 && (
              <span>
                offset: <span className="text-foreground">{e.chunk_offset_seconds.toFixed(2)}s</span>
              </span>
            )}
          </div>
          {e.resolutions.length > 0 && (
            <div className="space-y-0.5 font-mono text-[10px] text-muted-foreground">
              {e.resolutions.map((r, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span>{r.local_label}</span>
                  <span>→</span>
                  <Badge variant="outline" className={`text-[10px] ${speakerColor(r.global_label)}`}>
                    {r.global_label}
                  </Badge>
                  <span>·</span>
                  <span>dist {r.distance != null ? r.distance.toFixed(3) : "—"}</span>
                  <span>·</span>
                  <span>{r.duration_s.toFixed(2)}s</span>
                  {r.is_new && <Badge variant="outline" className="border-amber-500/50 text-[10px] text-amber-400">NEW</Badge>}
                </div>
              ))}
            </div>
          )}
          {e.notes.length > 0 && (
            <div className="space-y-0.5 font-mono text-[10px] text-amber-400/80">
              {e.notes.map((n, i) => <div key={i}>! {n}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export const TelemetryPanel = ({ events, sessionId }: { events: ChunkTelemetry[]; sessionId?: string }) => {
  const [open, setOpen] = useState(true)
  const agg = useMemo(() => computeAggregates(events), [events])
  // Show most recent first.
  const recent = useMemo(() => [...events].slice(-50).reverse(), [events])

  return (
    <div className="space-y-3 rounded-lg border bg-card/40 p-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2"
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown className="size-4 text-muted-foreground" /> : <ChevronRight className="size-4 text-muted-foreground" />}
          <Activity className="size-3.5 text-muted-foreground" />
          <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Model Telemetry</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
          <span>{agg.total} chunks</span>
          <span>·</span>
          <span>{agg.diarized} diarized</span>
          <span>·</span>
          <span>reg {agg.registryMax}</span>
          {sessionId && (
            <Badge variant="outline" className="font-mono text-[10px]">sess {sessionId.slice(0, 8)}</Badge>
          )}
        </div>
      </button>

      {open && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <div className="rounded border bg-card/60 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <Cpu className="size-3" /> Whisper p50
              </div>
              <div className="mt-1 font-mono text-base tabular-nums">{fmtMs(agg.whisperP50)}</div>
            </div>
            <div className="rounded border bg-card/60 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <Users className="size-3" /> Diar p50
              </div>
              <div className="mt-1 font-mono text-base tabular-nums">{fmtMs(agg.diarP50)}</div>
              <div className="mt-0.5 text-[10px] text-muted-foreground">embed {fmtMs(agg.embedP50)}</div>
            </div>
            <div className="rounded border bg-card/60 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <Sparkles className="size-3" /> New spk events
              </div>
              <div className="mt-1 font-mono text-base tabular-nums">{agg.newSpeakerEvents}</div>
              <div className="mt-0.5 text-[10px] text-muted-foreground">{agg.totalLocals} locals total</div>
            </div>
            <div className="rounded border bg-card/60 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <Activity className="size-3" /> Match dist
              </div>
              <div className="mt-1 font-mono text-base tabular-nums">
                {agg.distSamples.length ? agg.distMean.toFixed(2) : "—"}
              </div>
              <div className="mt-0.5 text-[10px] text-muted-foreground">
                {agg.distSamples.length ? `min ${agg.distMin.toFixed(2)} · max ${agg.distMax.toFixed(2)}` : "no embeddings yet"}
              </div>
            </div>
            <div className="rounded border bg-card/60 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                <Database className="size-3" /> Buffer p50
              </div>
              <div className="mt-1 font-mono text-base tabular-nums">
                {agg.bufferSamples ? `${agg.bufferP50.toFixed(1)}s` : "—"}
              </div>
              <div className="mt-0.5 text-[10px] text-muted-foreground">
                {agg.bufferSamples
                  ? `max spk in buf ${agg.speakersInBufferMax}`
                  : "no buffer yet"}
              </div>
            </div>
          </div>

          {agg.distSamples.length > 0 && (
            <div>
              <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Distance distribution (0 → 2 cosine)
              </div>
              <DistanceHistogram values={agg.distSamples} />
            </div>
          )}

          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Recent chunks (newest first)
              </span>
            </div>
            <div className="max-h-[420px] space-y-1.5 overflow-y-auto pr-1">
              {recent.length === 0 ? (
                <div className="rounded border border-dashed bg-card/30 px-3 py-6 text-center font-mono text-[11px] text-muted-foreground">
                  No telemetry yet — chunks will appear here as they're processed.
                </div>
              ) : (
                recent.map((e, i) => <ChunkRow key={`${e.chunkIndex}-${i}`} e={e} />)
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
