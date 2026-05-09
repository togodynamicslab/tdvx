import { useMemo, useState } from "react"
import type { EvalSummary } from "@/lib/evals"

// Tiny SVG sparkline-ish chart. Three series: WER p50, WER p95, server p50.
// Pure SVG, no chart library — keeps bundle size flat and behavior predictable.
//
// Y-axis is dual-scaled: WER on the left (%, 0-30 typical), server_p50 on
// the right (ms). Horizontal axis is run order, oldest → newest.
//
// Hover a point → tooltip with run tag + values. Optional onPick lets the
// parent scroll the list to that run.

type Series = "wer_p50" | "wer_p95" | "server_p50"

const SERIES: { key: Series; label: string; color: string; axis: "wer" | "ms" }[] = [
  { key: "wer_p50",    label: "WER p50",   color: "#22c55e", axis: "wer" }, // emerald
  { key: "wer_p95",    label: "WER p95",   color: "#f59e0b", axis: "wer" }, // amber
  { key: "server_p50", label: "server p50 (ms)", color: "#0ea5e9", axis: "ms" }, // sky
]

const W = 720
const H = 180
const PAD_L = 36
const PAD_R = 44
const PAD_T = 14
const PAD_B = 28

export function EvalTrend({
  evals,
  onPick,
}: {
  evals: EvalSummary[]
  onPick?: (runId: string) => void
}) {
  // Oldest first along the X axis (we receive newest first from the API).
  const ordered = useMemo(() => [...evals].reverse(), [evals])
  const n = ordered.length

  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  if (n === 0) {
    return (
      <div className="rounded-md border bg-card/40 p-6 text-center font-mono text-[11px] text-muted-foreground">
        No runs yet.
      </div>
    )
  }
  if (n === 1) {
    return (
      <div className="rounded-md border bg-card/40 p-6 text-center font-mono text-[11px] text-muted-foreground">
        Trend appears once you have 2+ runs. Current: WER p50 {fmtPct(ordered[0].wer_p50)}, server p50 {fmtMs(ordered[0].server_p50)}.
      </div>
    )
  }

  // Domain calc — pad ranges so points don't sit on the axis.
  const werVals = ordered.flatMap((e) => [e.wer_p50, e.wer_p95]).filter((v): v is number => v != null)
  const msVals = ordered.map((e) => e.server_p50).filter((v): v is number => v != null)
  const werMax = Math.max(0.05, ...werVals) * 1.15
  const werMin = 0
  const msMax = Math.max(50, ...msVals) * 1.15
  const msMin = 0

  const xFor = (i: number) => PAD_L + (n === 1 ? 0 : (i / (n - 1)) * (W - PAD_L - PAD_R))
  const yWer = (v: number) => PAD_T + (1 - (v - werMin) / Math.max(1e-9, werMax - werMin)) * (H - PAD_T - PAD_B)
  const yMs  = (v: number) => PAD_T + (1 - (v - msMin)  / Math.max(1e-9, msMax  - msMin))  * (H - PAD_T - PAD_B)

  const linePath = (key: Series, axis: "wer" | "ms") =>
    ordered
      .map((e, i) => {
        const v = e[key] as number | null | undefined
        if (v == null) return null
        const y = axis === "wer" ? yWer(v) : yMs(v)
        return { i, x: xFor(i), y, v }
      })
      .filter((p): p is { i: number; x: number; y: number; v: number } => p != null)

  const drawLine = (pts: { x: number; y: number }[]) =>
    pts.length === 0 ? "" : `M ${pts[0].x} ${pts[0].y} ` + pts.slice(1).map((p) => `L ${p.x} ${p.y}`).join(" ")

  const hovered = hoverIdx != null ? ordered[hoverIdx] : null

  return (
    <div className="rounded-md border bg-card/40 p-3">
      <div className="mb-2 flex items-center justify-between font-mono text-[10px] text-muted-foreground">
        <span>{n} runs · oldest → newest</span>
        <div className="flex flex-wrap gap-3">
          {SERIES.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1">
              <span className="inline-block size-2 rounded" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-44 w-full">
          {/* Y gridlines (4 horizontal) */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = PAD_T + t * (H - PAD_T - PAD_B)
            return <line key={t} x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} stroke="currentColor" strokeOpacity={0.08} />
          })}

          {/* Y-axis labels (left = WER%, right = ms) */}
          {[0, 0.5, 1].map((t) => {
            const y = PAD_T + (1 - t) * (H - PAD_T - PAD_B)
            const werV = werMin + t * (werMax - werMin)
            const msV  = msMin  + t * (msMax  - msMin)
            return (
              <g key={t}>
                <text x={PAD_L - 4} y={y + 3} textAnchor="end" fontSize="9" fill="currentColor" opacity="0.5">
                  {(werV * 100).toFixed(0)}%
                </text>
                <text x={W - PAD_R + 4} y={y + 3} textAnchor="start" fontSize="9" fill="currentColor" opacity="0.5">
                  {Math.round(msV)}
                </text>
              </g>
            )
          })}

          {/* Lines */}
          {SERIES.map((s) => {
            const pts = linePath(s.key, s.axis)
            return (
              <g key={s.key}>
                <path d={drawLine(pts)} fill="none" stroke={s.color} strokeWidth={1.5} strokeOpacity={0.9} />
                {pts.map((p) => (
                  <circle
                    key={p.i}
                    cx={p.x}
                    cy={p.y}
                    r={hoverIdx === p.i ? 4 : 2.5}
                    fill={s.color}
                    fillOpacity={hoverIdx === p.i ? 1 : 0.85}
                  />
                ))}
              </g>
            )
          })}

          {/* Invisible hover targets per run, full column height */}
          {ordered.map((_, i) => {
            const x = xFor(i)
            // Half-step width either side, clamped to drawable area.
            const stepW = n > 1 ? (W - PAD_L - PAD_R) / (n - 1) : (W - PAD_L - PAD_R)
            const halfW = Math.max(8, stepW / 2)
            return (
              <rect
                key={i}
                x={x - halfW}
                y={PAD_T}
                width={halfW * 2}
                height={H - PAD_T - PAD_B}
                fill="transparent"
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx((cur) => (cur === i ? null : cur))}
                onClick={() => onPick?.(ordered[i].run_id)}
                style={{ cursor: onPick ? "pointer" : "default" }}
              />
            )
          })}

          {/* Hover guide line */}
          {hoverIdx != null && (
            <line
              x1={xFor(hoverIdx)}
              y1={PAD_T}
              x2={xFor(hoverIdx)}
              y2={H - PAD_B}
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeDasharray="2 2"
            />
          )}

          {/* X-axis: first/middle/last run timestamps as ticks */}
          {[0, Math.floor((n - 1) / 2), n - 1]
            .filter((i, idx, arr) => arr.indexOf(i) === idx)
            .map((i) => {
              const e = ordered[i]
              const label = e.ts ? e.ts.replace("T", " ").slice(5, 16) : ""
              return (
                <text
                  key={i}
                  x={xFor(i)}
                  y={H - 8}
                  textAnchor="middle"
                  fontSize="9"
                  fill="currentColor"
                  opacity="0.5"
                >
                  {label}
                </text>
              )
            })}
        </svg>

        {hovered && (
          <div className="pointer-events-none absolute right-2 top-2 rounded border bg-background/95 px-2 py-1.5 font-mono text-[10px] shadow-sm">
            <div className="font-semibold">{hovered.tag || hovered.run_id}</div>
            <div className="text-muted-foreground">{hovered.ts?.replace("T", " ").slice(0, 16)}</div>
            <div className="mt-1 space-y-0.5">
              <div>WER p50 <span style={{ color: SERIES[0].color }}>{fmtPct(hovered.wer_p50)}</span></div>
              <div>WER p95 <span style={{ color: SERIES[1].color }}>{fmtPct(hovered.wer_p95)}</span></div>
              <div>server p50 <span style={{ color: SERIES[2].color }}>{fmtMs(hovered.server_p50)}</span></div>
              <div className="text-muted-foreground">files {hovered.files} · {hovered.model}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function fmtPct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`
}
function fmtMs(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v)}ms`
}
