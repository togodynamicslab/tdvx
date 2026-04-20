import { useMemo, useState } from "react"
import { listRuns, verdictClass, type RunSummary } from "@/lib/runs"
import { navigate } from "@/lib/hashRouter"
import { useAsync } from "@/lib/useAsync"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { GitCompare, ArrowUpDown, FileText, Loader2 } from "lucide-react"

type SortKey = keyof RunSummary
type SortDir = "asc" | "desc"

export default function RunsList() {
  const query = useAsync(() => listRuns(), [])
  const runs: RunSummary[] = query.status === "ok" ? query.data : []
  const [sortKey, setSortKey] = useState<SortKey>("started_at")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const sorted = useMemo(() => {
    const copy = [...runs]
    copy.sort((a, b) => {
      const va = a[sortKey]
      const vb = b[sortKey]
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      if (va < vb) return sortDir === "asc" ? -1 : 1
      if (va > vb) return sortDir === "asc" ? 1 : -1
      return 0
    })
    return copy
  }, [runs, sortKey, sortDir])

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    else {
      setSortKey(k)
      setSortDir("desc")
    }
  }

  const toggleSelect = (id: string) => {
    setSelected((s) => {
      const n = new Set(s)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  }

  const goCompare = () => {
    if (selected.size < 2) return
    navigate(`compare?ids=${Array.from(selected).join(",")}`)
  }

  const TH = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      onClick={() => toggleSort(k)}
      className="cursor-pointer select-none px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <ArrowUpDown className="size-3 opacity-40" />
        {sortKey === k && <span className="text-foreground">{sortDir === "asc" ? "↑" : "↓"}</span>}
      </span>
    </th>
  )

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-mono text-2xl font-semibold tracking-tight">Stress test runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {query.status === "ok"
              ? `${runs.length} run${runs.length === 1 ? "" : "s"} · select 2+ to compare`
              : query.status === "loading"
                ? "loading…"
                : "error"}
          </p>
        </div>
        <Button
          onClick={goCompare}
          disabled={selected.size < 2}
          variant="secondary"
          className="gap-2 font-mono"
        >
          <GitCompare className="size-4" />
          Compare ({selected.size})
        </Button>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Runs
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="border-b">
                <tr>
                  <th className="w-10 px-3 py-2"></th>
                  <TH k="started_at" label="Date" />
                  <TH k="preset" label="Preset" />
                  <TH k="scenario" label="Scenario" />
                  <TH k="verdict" label="Verdict" />
                  <TH k="total_requests" label="Reqs" />
                  <TH k="success_rate_pct" label="Success %" />
                  <TH k="ttfr_p95_s" label="TTFR p95" />
                  <TH k="rtf_p95" label="RTF p95" />
                  <TH k="model" label="Model" />
                  <th className="w-10 px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <tr
                    key={r.run_id}
                    className="border-b last:border-0 hover:bg-muted/30"
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={selected.has(r.run_id)}
                        onChange={() => toggleSelect(r.run_id)}
                        className="accent-foreground"
                      />
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {formatDate(r.started_at)}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{r.preset}</td>
                    <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                      {r.scenario}
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant="outline" className={`font-mono text-[10px] ${verdictClass(r.verdict)}`}>
                        {r.verdict}
                      </Badge>
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums">{r.total_requests}</td>
                    <td className="px-3 py-2 font-mono tabular-nums">{r.success_rate_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono tabular-nums">
                      {r.ttfr_p95_s != null ? `${r.ttfr_p95_s.toFixed(2)}s` : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums">
                      {r.rtf_p95 != null ? r.rtf_p95.toFixed(2) : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{r.model}</td>
                    <td className="px-3 py-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 gap-1 px-2 font-mono text-[11px]"
                        onClick={() => navigate(`run/${r.run_id}`)}
                      >
                        <FileText className="size-3" />
                        open
                      </Button>
                    </td>
                  </tr>
                ))}
                {query.status === "loading" && (
                  <tr>
                    <td colSpan={11} className="px-3 py-10 text-center text-sm text-muted-foreground">
                      <Loader2 className="mr-2 inline size-4 animate-spin" />
                      Loading runs…
                    </td>
                  </tr>
                )}
                {query.status === "error" && (
                  <tr>
                    <td colSpan={11} className="px-3 py-10 text-center text-sm text-destructive">
                      Failed to load runs: {query.error}
                    </td>
                  </tr>
                )}
                {query.status === "ok" && sorted.length === 0 && (
                  <tr>
                    <td colSpan={11} className="px-3 py-10 text-center text-sm text-muted-foreground">
                      No runs yet. Trigger a stress test with{" "}
                      <code className="font-mono text-xs">python3 tests/stochastic_test.py --preset smoke</code>.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function formatDate(s: string): string {
  if (s.length < 15) return s
  const y = s.slice(0, 4)
  const m = s.slice(4, 6)
  const d = s.slice(6, 8)
  const hh = s.slice(9, 11)
  const mm = s.slice(11, 13)
  return `${y}-${m}-${d} ${hh}:${mm}`
}
