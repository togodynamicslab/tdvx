// Read-only client for /api/evals/*. Mirrors lib/runs.ts but for pipeline-eval data.
//
// One eval row is what scripts/eval_pipeline.py appends to results/eval_history.ndjson:
// a journal entry summarising one A/B run against the Deepgram reference corpus.

export type EvalSummary = {
  run_id: string
  ts: string
  tag: string
  commit: string
  endpoint: string
  model: string
  language: string
  diarize: boolean
  files: number
  wer_mean: number | null
  wer_p50: number | null
  wer_p95: number | null
  ser_mean: number | null
  ser_p50: number | null
  lat_p50: number
  lat_p95: number
  server_p50: number | null
  server_p95: number | null
  summary_path: string
}

export type HypSegment = {
  start: number
  end: number
  text: string
}

export type EvalRow = {
  file: string
  wer: number | null
  speaker_error_rate: number | null
  latency_ms: number
  server_total_ms: number | null
  server_whisper_ms: number | null
  server_diar_ms: number | null
  diarize_ran: boolean | null
  locals_detected: number | null
  registry_size: number | null
  ref_text: string
  hyp_text: string
  hyp_segments?: HypSegment[]  // present in newer eval runs; optional for backward compat
}

export type EvalDetail = EvalSummary & {
  rows: EvalRow[]
  summary_md: string | null
}

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? ""

export async function listEvals(): Promise<EvalSummary[]> {
  const res = await fetch(`${API_BASE}/api/evals`)
  if (!res.ok) throw new Error(`GET /api/evals → ${res.status}`)
  const data = (await res.json()) as { evals: EvalSummary[] }
  return data.evals
}

const detailCache = new Map<string, EvalDetail>()

export async function getEvalDetail(runId: string): Promise<EvalDetail> {
  const cached = detailCache.get(runId)
  if (cached) return cached
  const res = await fetch(`${API_BASE}/api/evals/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(`GET /api/evals/${runId} → ${res.status}`)
  const data = (await res.json()) as EvalDetail
  detailCache.set(runId, data)
  return data
}

// Cosmetic helpers — keep consistent with existing palette in the runs view.
export function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—"
  return `${(v * 100).toFixed(1)}%`
}

export function fmtMs(v: number | null | undefined): string {
  if (v == null) return "—"
  return `${Math.round(v)}ms`
}

// WER bands chosen to roughly match how it feels: <10% green, <20% amber, else red.
export function werClass(v: number | null | undefined): string {
  if (v == null) return "text-muted-foreground"
  if (v < 0.10) return "text-emerald-400"
  if (v < 0.20) return "text-amber-400"
  return "text-destructive"
}
