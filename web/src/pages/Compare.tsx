import { getRunDetail, verdictClass, type RunDetail } from "@/lib/runs"
import { navigate } from "@/lib/hashRouter"
import { useAsync } from "@/lib/useAsync"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Loader2 } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"

const COLORS = ["#10b981", "#f59e0b", "#6366f1", "#ef4444", "#ec4899", "#06b6d4"]

export default function Compare({ ids }: { ids: string[] }) {
  const query = useAsync<RunDetail[]>(
    () => Promise.all(ids.map((id) => getRunDetail(id))),
    [ids.join(",")],
  )

  if (query.status === "loading") {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10 text-sm text-muted-foreground">
        <Loader2 className="mr-2 inline size-4 animate-spin" />
        Loading {ids.length} runs…
      </div>
    )
  }
  if (query.status === "error") {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10">
        <Button variant="ghost" onClick={() => navigate("runs")} className="gap-2 font-mono">
          <ArrowLeft className="size-4" /> back
        </Button>
        <div className="mt-8 rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          Failed to load: {query.error}
        </div>
      </div>
    )
  }
  const runs = query.data

  if (runs.length < 2) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10">
        <Button variant="ghost" onClick={() => navigate("runs")} className="gap-2 font-mono">
          <ArrowLeft className="size-4" /> back
        </Button>
        <div className="mt-8 rounded-md border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-400">
          Need at least 2 valid runs to compare. Got {runs.length}.
        </div>
      </div>
    )
  }

  // Build unified latency chart data: one row per sim_hour, one series per run
  const hours = Array.from(new Set(runs.flatMap((r) => r.table_a.map((t) => t.hour)))).sort((a, b) => a - b)
  const latencyData = hours.map((h) => {
    const row: Record<string, number | null> = { hour: h }
    runs.forEach((r) => {
      const t = r.table_a.find((x) => x.hour === h)
      row[`${r.run_id}__p95`] = t ? t.ttfr_p95_s : null
    })
    return row
  })

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate("runs")} className="gap-2 font-mono">
          <ArrowLeft className="size-4" /> back
        </Button>
        <div>
          <h1 className="font-mono text-xl font-semibold tracking-tight">Compare</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">{runs.length} runs side by side</p>
        </div>
      </div>

      {/* Headline comparison */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {runs.map((r, i) => (
          <Card key={r.run_id} className="relative">
            <div
              className="absolute left-0 top-0 h-full w-1 rounded-l-lg"
              style={{ backgroundColor: COLORS[i % COLORS.length] }}
            />
            <CardHeader className="pb-2 pl-5">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="font-mono text-xs">{r.run_id}</CardTitle>
                <Badge variant="outline" className={`font-mono text-[10px] ${verdictClass(r.verdict)}`}>
                  {r.verdict}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 pl-5">
              <KV k="preset" v={`${r.preset} / ${r.scenario}`} />
              <KV k="requests" v={r.total_requests.toLocaleString()} />
              <KV k="success" v={`${r.success_rate_pct.toFixed(2)}%`} />
              <KV k="TTFR p50" v={r.ttfr_p50_s != null ? `${r.ttfr_p50_s.toFixed(2)}s` : "—"} />
              <KV k="TTFR p95" v={r.ttfr_p95_s != null ? `${r.ttfr_p95_s.toFixed(2)}s` : "—"} />
              <KV k="RTF p95" v={r.rtf_p95 != null ? r.rtf_p95.toFixed(3) : "—"} />
              <KV k="peak TTFR p95" v={r.peak_ttfr_p95_s != null ? `${r.peak_ttfr_p95_s.toFixed(2)}s` : "—"} />
              <KV k="model" v={r.model} />
              <KV k="compression" v={`${r.compression}×`} />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Overlay latency p95 chart */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            TTFR p95 by simulated hour — overlay
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full">
            <ResponsiveContainer>
              <LineChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="hour" tick={{ fill: "currentColor", fontSize: 11 }} />
                <YAxis tick={{ fill: "currentColor", fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card, 0 0% 10%))", border: "1px solid hsl(var(--border, 0 0% 20%))" }} />
                <Legend />
                <ReferenceLine y={15} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "SLO 15s", fill: "#ef4444", fontSize: 10 }} />
                {runs.map((r, i) => (
                  <Line
                    key={r.run_id}
                    type="monotone"
                    dataKey={`${r.run_id}__p95`}
                    name={`${r.preset}/${r.scenario} p95`}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-border/30 py-1 text-xs last:border-0">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-mono tabular-nums">{v}</span>
    </div>
  )
}
