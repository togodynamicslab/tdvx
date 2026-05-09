import { useMemo, useState } from "react"
import { ArrowUpDown, ChevronDown, ChevronRight, FileText, Loader2 } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useAsync } from "@/lib/useAsync"
import {
  listEvals,
  getEvalDetail,
  fmtPct,
  fmtMs,
  werClass,
  type EvalSummary,
  type EvalDetail,
  type EvalRow,
} from "@/lib/evals"
import { diffWords, type DiffOp } from "@/lib/wordDiff"
import { EvalTrend } from "@/components/EvalTrend"

type SortKey = "ts" | "wer_p50" | "wer_p95" | "ser_mean" | "lat_p50" | "files"
type SortDir = "asc" | "desc"

// Word-level diff between Deepgram reference text and our pipeline output.
// Two parallel rows: top = ref, bottom = hyp. Each token is a small badge
// colored by op kind. Empty placeholders (·) mark insertions/deletions to
// keep the columns aligned visually.
function WordDiff({ refText, hypText }: { refText: string; hypText: string }) {
  const result = useMemo(() => diffWords(refText, hypText), [refText, hypText])
  if (!refText && !hypText) {
    return <div className="px-3 py-2 text-[11px] text-muted-foreground">no text to diff</div>
  }

  const total = result.refTokens.length || 1
  const errs = result.counts.sub + result.counts.del + result.counts.ins
  const wer = errs / total

  const refCells: { tok: string; cls: string; title?: string }[] = []
  const hypCells: { tok: string; cls: string; title?: string }[] = []
  for (const op of result.ops) pushPair(op, refCells, hypCells)

  return (
    <div className="space-y-2 rounded-md border bg-background/60 p-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-muted-foreground">
        <span>WER <span className={werClass(wer)}>{(wer * 100).toFixed(1)}%</span></span>
        <span>·</span>
        <span>equal <span className="text-emerald-400">{result.counts.equal}</span></span>
        <span>sub <span className="text-amber-400">{result.counts.sub}</span></span>
        <span>del <span className="text-rose-400">{result.counts.del}</span></span>
        <span>ins <span className="text-sky-400">{result.counts.ins}</span></span>
        <span>·</span>
        <span>ref tokens <span className="text-foreground">{result.refTokens.length}</span></span>
        <span>hyp tokens <span className="text-foreground">{result.hypTokens.length}</span></span>
      </div>
      <DiffRow label="ref" cells={refCells} />
      <DiffRow label="hyp" cells={hypCells} />
    </div>
  )
}

function pushPair(
  op: DiffOp,
  refCells: { tok: string; cls: string; title?: string }[],
  hypCells: { tok: string; cls: string; title?: string }[],
) {
  // Empty placeholder to keep columns aligned when one side is missing the token.
  const blank = { tok: "·", cls: "border-transparent text-muted-foreground/40" }
  switch (op.kind) {
    case "equal":
      refCells.push({ tok: op.ref, cls: "border-emerald-500/30 text-emerald-300/90" })
      hypCells.push({ tok: op.hyp, cls: "border-emerald-500/30 text-emerald-300/90" })
      break
    case "sub":
      refCells.push({ tok: op.ref, cls: "border-amber-500/50 text-amber-300", title: `sub → ${op.hyp}` })
      hypCells.push({ tok: op.hyp, cls: "border-amber-500/50 text-amber-300", title: `sub from ${op.ref}` })
      break
    case "del":
      refCells.push({ tok: op.ref, cls: "border-rose-500/60 text-rose-300", title: "deleted in hyp" })
      hypCells.push(blank)
      break
    case "ins":
      refCells.push(blank)
      hypCells.push({ tok: op.hyp, cls: "border-sky-500/60 text-sky-300", title: "inserted in hyp" })
      break
  }
}

function DiffRow({
  label,
  cells,
}: {
  label: string
  cells: { tok: string; cls: string; title?: string }[]
}) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-0.5 w-8 shrink-0 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <div className="flex flex-wrap gap-1">
        {cells.length === 0 ? (
          <span className="text-[11px] text-muted-foreground">(empty)</span>
        ) : (
          cells.map((c, i) => (
            <span
              key={i}
              title={c.title}
              className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${c.cls}`}
            >
              {c.tok}
            </span>
          ))
        )}
      </div>
    </div>
  )
}

// Inline detail panel: lazy-loads the per-file rows when a row is expanded.
// Keeps everything on one page so you can scroll quickly between runs.
function EvalDetailPanel({ runId }: { runId: string }) {
  const detailQuery = useAsync(() => getEvalDetail(runId), [runId])

  if (detailQuery.status === "loading") {
    return (
      <div className="flex items-center gap-2 px-4 py-6 text-xs text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> loading detail…
      </div>
    )
  }
  if (detailQuery.status === "error") {
    return (
      <div className="px-4 py-6 text-xs text-destructive">
        failed to load: {detailQuery.error}
      </div>
    )
  }

  const detail = detailQuery.data as EvalDetail
  const rows = detail.rows ?? []
  // Map the journal's "pt"/"en" code back to the corpus subdir name.
  // Keep this in sync with scripts/build_deepgram_reference.py.
  const corpusLang = detail.language === "pt" ? "pt-BR" : detail.language === "en" ? "en-US" : detail.language

  return (
    <div className="space-y-3 border-t bg-muted/20 px-4 py-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
        <span>commit <span className="text-foreground">{detail.commit ?? "—"}</span></span>
        <span>endpoint <span className="text-foreground">{detail.endpoint}</span></span>
        <span>diarize <span className="text-foreground">{String(detail.diarize)}</span></span>
        <span>files <span className="text-foreground">{detail.files}</span></span>
        <a
          href={`/api/evals/${encodeURIComponent(detail.run_id)}/summary.md`}
          className="ml-auto inline-flex items-center gap-1 text-foreground hover:underline"
          target="_blank"
          rel="noreferrer"
        >
          <FileText className="size-3" /> summary.md
        </a>
        <a
          href={`/api/evals/${encodeURIComponent(detail.run_id)}/rows.ndjson`}
          className="inline-flex items-center gap-1 text-foreground hover:underline"
          target="_blank"
          rel="noreferrer"
        >
          <FileText className="size-3" /> rows.ndjson
        </a>
      </div>

      <div className="overflow-x-auto rounded-md border bg-background">
        <table className="min-w-full text-[11px]">
          <thead>
            <tr className="border-b bg-muted/40 font-mono uppercase tracking-wider text-muted-foreground">
              <th className="w-6 px-2 py-1.5"></th>
              <th className="px-2 py-1.5 text-left">file</th>
              <th className="px-2 py-1.5 text-right">WER</th>
              <th className="px-2 py-1.5 text-right">SER</th>
              <th className="px-2 py-1.5 text-right">client ms</th>
              <th className="px-2 py-1.5 text-right">server ms</th>
              <th className="px-2 py-1.5 text-right">locals</th>
              <th className="px-2 py-1.5 text-right">reg</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-2 py-4 text-center text-muted-foreground">
                  no rows
                </td>
              </tr>
            ) : (
              rows.map((r: EvalRow, i) => (
                <FileRow key={r.file + i} row={r} lang={corpusLang} />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// One per-file row + an expandable diff panel beneath it.
function FileRow({ row, lang }: { row: EvalRow; lang: string }) {
  const [open, setOpen] = useState(false)
  const hasDiff = (row.ref_text?.length ?? 0) > 0 || (row.hyp_text?.length ?? 0) > 0
  return (
    <>
      <tr
        onClick={() => hasDiff && setOpen((v) => !v)}
        className={`border-b last:border-0 hover:bg-muted/30 ${hasDiff ? "cursor-pointer" : ""} ${open ? "bg-muted/40" : ""}`}
      >
        <td className="pl-2 align-middle text-muted-foreground">
          {hasDiff ? (
            open ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />
          ) : null}
        </td>
        <td className="px-2 py-1.5 font-mono text-foreground">{row.file}</td>
        <td className={`px-2 py-1.5 text-right font-mono ${werClass(row.wer)}`}>{fmtPct(row.wer)}</td>
        <td className="px-2 py-1.5 text-right font-mono">{fmtPct(row.speaker_error_rate)}</td>
        <td className="px-2 py-1.5 text-right font-mono">{row.latency_ms}</td>
        <td className="px-2 py-1.5 text-right font-mono">{fmtMs(row.server_total_ms)}</td>
        <td className="px-2 py-1.5 text-right font-mono">{row.locals_detected ?? "—"}</td>
        <td className="px-2 py-1.5 text-right font-mono">{row.registry_size ?? "—"}</td>
      </tr>
      {open && (
        <tr>
          <td colSpan={8} className="px-2 pb-3 pt-1">
            <div className="space-y-2">
              <FileAudio file={row.file} lang={lang} />
              <WordDiff refText={row.ref_text} hypText={row.hyp_text} />
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

// Native HTML5 audio element pointing at /api/corpus/<lang>/<filename>.
// Tiny on purpose — no controls beyond what the browser ships.
function FileAudio({ file, lang }: { file: string; lang: string }) {
  const src = `/api/corpus/${encodeURIComponent(lang)}/${encodeURIComponent(file)}`
  return (
    <div className="flex items-center gap-3 rounded-md border bg-background/60 px-3 py-2">
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">audio</span>
      <audio controls preload="none" src={src} className="h-8 flex-1" />
      <a
        href={src}
        download={file}
        className="font-mono text-[10px] text-muted-foreground hover:text-foreground"
      >
        download
      </a>
    </div>
  )
}

export default function EvalsList() {
  const query = useAsync(() => listEvals(), [])
  const evals: EvalSummary[] = query.status === "ok" ? query.data : []
  const [sortKey, setSortKey] = useState<SortKey>("ts")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [openId, setOpenId] = useState<string | null>(null)

  const sorted = useMemo(() => {
    const copy = [...evals]
    copy.sort((a, b) => {
      const va = a[sortKey] as number | string | null
      const vb = b[sortKey] as number | string | null
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      if (va < vb) return sortDir === "asc" ? -1 : 1
      if (va > vb) return sortDir === "asc" ? 1 : -1
      return 0
    })
    return copy
  }, [evals, sortKey, sortDir])

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    else {
      setSortKey(k)
      setSortDir("desc")
    }
  }

  const TH = ({ k, label, align = "left" }: { k: SortKey; label: string; align?: "left" | "right" }) => (
    <th
      onClick={() => toggleSort(k)}
      className={`cursor-pointer select-none px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      <span className={`inline-flex items-center gap-1 ${align === "right" ? "flex-row-reverse" : ""}`}>
        {label}
        <ArrowUpDown className="size-3 opacity-40" />
        {sortKey === k && <span className="text-foreground">{sortDir === "asc" ? "↑" : "↓"}</span>}
      </span>
    </th>
  )

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      <div>
        <h1 className="font-mono text-2xl font-semibold tracking-tight">Pipeline evals</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {query.status === "ok"
            ? `${evals.length} run${evals.length === 1 ? "" : "s"} · scored against the Deepgram reference corpus`
            : query.status === "loading"
              ? "loading…"
              : "error"}
        </p>
      </div>

      {/* Trend chart across runs (oldest → newest). Click a point to focus its row below. */}
      {query.status === "ok" && evals.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Trend
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <EvalTrend evals={evals} onPick={(id) => setOpenId(id)} />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Recent runs
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {query.status === "loading" && (
            <div className="flex items-center gap-2 px-4 py-6 text-xs text-muted-foreground">
              <Loader2 className="size-3 animate-spin" /> loading…
            </div>
          )}
          {query.status === "error" && (
            <div className="px-4 py-6 text-xs text-destructive">
              failed: {query.error}
            </div>
          )}
          {query.status === "ok" && evals.length === 0 && (
            <div className="px-4 py-8 text-center text-xs text-muted-foreground">
              No eval runs yet. Run <code className="font-mono">python3 scripts/eval_pipeline.py --tag baseline</code> to create the first one.
            </div>
          )}
          {query.status === "ok" && evals.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="border-b">
                  <tr>
                    <th className="w-8" />
                    <TH k="ts" label="when" />
                    <th className="px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground">tag</th>
                    <th className="px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground">model</th>
                    <th className="px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground">lang</th>
                    <TH k="files" label="files" align="right" />
                    <TH k="wer_p50" label="WER p50" align="right" />
                    <TH k="wer_p95" label="WER p95" align="right" />
                    <TH k="ser_mean" label="SER mean" align="right" />
                    <TH k="lat_p50" label="lat p50" align="right" />
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((e) => {
                    const isOpen = openId === e.run_id
                    return (
                      <Row
                        key={e.run_id}
                        ev={e}
                        isOpen={isOpen}
                        onToggle={() => setOpenId(isOpen ? null : e.run_id)}
                      />
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function Row({ ev, isOpen, onToggle }: { ev: EvalSummary; isOpen: boolean; onToggle: () => void }) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={`cursor-pointer border-b last:border-0 hover:bg-muted/30 ${isOpen ? "bg-muted/40" : ""}`}
      >
        <td className="pl-3 align-middle text-muted-foreground">
          {isOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        </td>
        <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
          {ev.ts ? ev.ts.replace("T", " ").slice(0, 19) : "—"}
        </td>
        <td className="px-3 py-2">
          {ev.tag ? (
            <Badge variant="outline" className="font-mono text-[10px]">{ev.tag}</Badge>
          ) : (
            <span className="text-[11px] text-muted-foreground">—</span>
          )}
        </td>
        <td className="px-3 py-2 font-mono text-[11px]">{ev.model}</td>
        <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">{ev.language}</td>
        <td className="px-3 py-2 text-right font-mono text-[11px]">{ev.files}</td>
        <td className={`px-3 py-2 text-right font-mono text-[11px] ${werClass(ev.wer_p50)}`}>{fmtPct(ev.wer_p50)}</td>
        <td className={`px-3 py-2 text-right font-mono text-[11px] ${werClass(ev.wer_p95)}`}>{fmtPct(ev.wer_p95)}</td>
        <td className="px-3 py-2 text-right font-mono text-[11px]">{fmtPct(ev.ser_mean)}</td>
        <td className="px-3 py-2 text-right font-mono text-[11px]">{fmtMs(ev.lat_p50)}</td>
      </tr>
      {isOpen && (
        <tr>
          <td colSpan={10} className="p-0">
            <EvalDetailPanel runId={ev.run_id} />
          </td>
        </tr>
      )}
    </>
  )
}
