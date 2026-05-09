import { getRunDetail, verdictClass } from "@/lib/runs"
import { navigate } from "@/lib/hashRouter"
import { useAsync } from "@/lib/useAsync"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Loader2, Download } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ReferenceLine,
} from "recharts"

export default function RunDetail({ runId }: { runId: string }) {
  const query = useAsync(() => getRunDetail(runId), [runId])

  if (query.status === "loading") {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10 text-sm text-muted-foreground">
        <Loader2 className="mr-2 inline size-4 animate-spin" />
        Loading run {runId}…
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
          Failed to load run <code className="font-mono">{runId}</code>: {query.error}
        </div>
      </div>
    )
  }
  const run = query.data

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate("runs")} className="gap-2 font-mono">
            <ArrowLeft className="size-4" /> back
          </Button>
          <div>
            <div className="font-mono text-xl font-semibold tracking-tight">{run.run_id}</div>
            <div className="mt-0.5 text-sm text-muted-foreground">
              {run.preset} / {run.scenario} · {run.real_duration_min} min real · {run.simulated_hours}h
              sim · {run.compression}× compression
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="gap-2 font-mono text-xs"
            asChild
          >
            <a
              href={`/api/runs/${encodeURIComponent(run.run_id)}/summary.md`}
              download
            >
              <Download className="size-3.5" />
              summary.md
            </a>
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="gap-2 font-mono text-xs"
            asChild
          >
            <a
              href={`/api/runs/${encodeURIComponent(run.run_id)}/requests.ndjson`}
              download
            >
              <Download className="size-3.5" />
              ndjson
            </a>
          </Button>
          <Badge variant="outline" className={`font-mono text-xs ${verdictClass(run.verdict)}`}>
            {run.verdict}
          </Badge>
        </div>
      </div>

      {/* Executive summary */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Total requests" value={run.total_requests.toLocaleString()} />
        <Stat
          label="Success"
          value={`${run.success_rate_pct.toFixed(2)}%`}
          tone={run.success_rate_pct >= 99.9 ? "ok" : run.success_rate_pct >= 99 ? "warn" : "bad"}
        />
        <Stat
          label="TTFR p95"
          value={run.ttfr_p95_s != null ? `${run.ttfr_p95_s.toFixed(2)}s` : "—"}
          tone={run.ttfr_p95_s != null ? (run.ttfr_p95_s < 15 ? "ok" : run.ttfr_p95_s < 30 ? "warn" : "bad") : undefined}
        />
        <Stat label="RTF p95" value={run.rtf_p95 != null ? run.rtf_p95.toFixed(3) : "—"} />
        <Stat label="Model" value={run.model} mono />
        <Stat label="Users × audios/day" value={`${run.users} × ${run.audios_per_user_day}`} />
        <Stat
          label="Peak (14-18h) TTFR p95"
          value={run.peak_ttfr_p95_s != null ? `${run.peak_ttfr_p95_s.toFixed(2)}s` : "—"}
        />
        <Stat label="Peak RTF p95" value={run.peak_rtf_p95 != null ? run.peak_rtf_p95.toFixed(3) : "—"} />
      </div>

      {/* Latency vs sim hour chart */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Latency vs simulated hour
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-72 w-full">
            <ResponsiveContainer>
              <LineChart data={run.table_a}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="hour" tick={{ fill: "currentColor", fontSize: 11 }} />
                <YAxis tick={{ fill: "currentColor", fontSize: 11 }} label={{ value: "TTFR (s)", angle: -90, position: "insideLeft", fill: "currentColor" }} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card, 0 0% 10%))", border: "1px solid hsl(var(--border, 0 0% 20%))" }} />
                <Legend />
                <ReferenceLine y={15} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "SLO 15s", fill: "#ef4444", fontSize: 10 }} />
                <Line type="monotone" dataKey="ttfr_p50_s" name="p50" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="ttfr_p95_s" name="p95" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="ttfr_p99_s" name="p99" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Duration histogram */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Audio duration histogram (log-normal target)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-56 w-full">
            <ResponsiveContainer>
              <BarChart data={run.duration_histogram}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="bucket" tick={{ fill: "currentColor", fontSize: 10 }} />
                <YAxis tick={{ fill: "currentColor", fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card, 0 0% 10%))", border: "1px solid hsl(var(--border, 0 0% 20%))" }} />
                <Bar dataKey="count" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Table A */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            A) Latency by simulated hour
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="border-b">
                <tr>
                  <TH>Sim hour</TH>
                  <TH>Requests</TH>
                  <TH>TTFR p50 (s)</TH>
                  <TH>TTFR p95 (s)</TH>
                  <TH>TTFR p99 (s)</TH>
                  <TH>RTF p95</TH>
                  <TH>Success %</TH>
                </tr>
              </thead>
              <tbody>
                {run.table_a.map((r) => (
                  <tr key={r.hour} className="border-b last:border-0">
                    <TD mono>{String(r.hour).padStart(2, "0")}</TD>
                    <TD mono>{r.requests}</TD>
                    <TD mono>{r.ttfr_p50_s.toFixed(2)}</TD>
                    <TD mono>{r.ttfr_p95_s.toFixed(2)}</TD>
                    <TD mono>{r.ttfr_p99_s.toFixed(2)}</TD>
                    <TD mono>{r.rtf_p95.toFixed(3)}</TD>
                    <TD mono tone={r.success_pct >= 99 ? "ok" : r.success_pct >= 90 ? "warn" : "bad"}>
                      {r.success_pct.toFixed(1)}%
                    </TD>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Table B */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            B) Stochastic validation
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full border-collapse text-sm">
            <thead className="border-b">
              <tr>
                <TH>Metric</TH>
                <TH>Expected</TH>
                <TH>Measured</TH>
                <TH>Status</TH>
              </tr>
            </thead>
            <tbody>
              {run.table_b.map((r) => (
                <tr key={r.metric} className="border-b last:border-0">
                  <TD>{r.metric}</TD>
                  <TD mono>{r.expected}</TD>
                  <TD mono>{String(r.measured)}</TD>
                  <TD>
                    <Badge variant="outline" className={`font-mono text-[10px] ${verdictClass(r.status)}`}>
                      {r.status}
                    </Badge>
                  </TD>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Table D */}
      {run.table_d.rows.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              D) Failures by type × simulated hour
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead className="border-b">
                  <tr>
                    <TH>Sim hour</TH>
                    {run.table_d.fail_types.map((t) => (
                      <TH key={t}>{t}</TH>
                    ))}
                    <TH>Total</TH>
                  </tr>
                </thead>
                <tbody>
                  {run.table_d.rows.map((r) => (
                    <tr key={r.hour} className="border-b last:border-0">
                      <TD mono>{String(r.hour).padStart(2, "0")}</TD>
                      {run.table_d.fail_types.map((t) => (
                        <TD key={t} mono>
                          {r[t] ?? 0}
                        </TD>
                      ))}
                      <TD mono>{r.total}</TD>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Config */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Config
          </CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md border bg-muted/30 p-3 text-[11px] leading-relaxed">
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}

function Stat({
  label,
  value,
  tone,
  mono,
}: {
  label: string
  value: string
  tone?: "ok" | "warn" | "bad"
  mono?: boolean
}) {
  const toneClass =
    tone === "ok"
      ? "text-emerald-400"
      : tone === "warn"
        ? "text-amber-400"
        : tone === "bad"
          ? "text-destructive"
          : ""
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-2 text-2xl font-medium tabular-nums ${mono ? "font-mono text-lg" : ""} ${toneClass}`}>
        {value}
      </div>
    </div>
  )
}

function TH({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
      {children}
    </th>
  )
}

function TD({
  children,
  mono,
  tone,
}: {
  children: React.ReactNode
  mono?: boolean
  tone?: "ok" | "warn" | "bad"
}) {
  const toneClass =
    tone === "ok"
      ? "text-emerald-400"
      : tone === "warn"
        ? "text-amber-400"
        : tone === "bad"
          ? "text-destructive"
          : ""
  return <td className={`px-3 py-2 tabular-nums ${mono ? "font-mono text-xs" : ""} ${toneClass}`}>{children}</td>
}
