export type RunSummary = {
  run_id: string
  preset: string
  scenario: string
  verdict: "PASS" | "DEGRADED" | "FAIL"
  started_at: string
  real_duration_min: number
  simulated_hours: number
  compression: number
  model: string
  endpoint: string
  users: number
  audios_per_user_day: number
  total_requests: number
  success_rate_pct: number
  ttfr_p50_s: number | null
  ttfr_p95_s: number | null
  ttfr_p99_s: number | null
  rtf_p50: number | null
  rtf_p95: number | null
  peak_ttfr_p95_s: number | null
  peak_rtf_p95: number | null
  status_counts: Record<string, number>
}

export type TableARow = {
  hour: number
  requests: number
  ttfr_p50_s: number
  ttfr_p95_s: number
  ttfr_p99_s: number
  rtf_p95: number
  success_pct: number
}

export type TableBRow = {
  metric: string
  expected: string
  measured: string | number
  status: "PASS" | "WARN" | "FAIL"
}

export type TableDData = {
  fail_types: string[]
  rows: Array<{ hour: number; total: number; [status: string]: number }>
}

export type RunDetail = RunSummary & {
  config: Record<string, unknown>
  table_a: TableARow[]
  table_b: TableBRow[]
  table_d: TableDData
  duration_histogram: Array<{ bucket: string; count: number }>
  inter_arrival_sample: number[]
}

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? ""

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/api/runs`)
  if (!res.ok) throw new Error(`GET /api/runs → ${res.status}`)
  const data = (await res.json()) as { runs: RunSummary[] }
  return data.runs
}

const detailCache = new Map<string, RunDetail>()

export async function getRunDetail(runId: string): Promise<RunDetail> {
  const cached = detailCache.get(runId)
  if (cached) return cached
  const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(`GET /api/runs/${runId} → ${res.status}`)
  const data = (await res.json()) as RunDetail
  detailCache.set(runId, data)
  return data
}

export function verdictClass(v: string): string {
  if (v === "PASS") return "border-emerald-500/50 text-emerald-400 bg-emerald-500/10"
  if (v === "DEGRADED") return "border-amber-500/50 text-amber-400 bg-amber-500/10"
  return "border-destructive/60 text-destructive bg-destructive/10"
}
