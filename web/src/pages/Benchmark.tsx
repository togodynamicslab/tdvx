import { useMemo } from "react"
import { Crown, Loader2 } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useAsync } from "@/lib/useAsync"
import { listEvals, getEvalDetail, type EvalRow, type EvalSummary } from "@/lib/evals"
import { listValidations, type ValidationEntry, type Verdict } from "@/lib/validations"

// "Ours vs Deepgram" benchmark, scoreboard-style. The whole point is to
// answer one question at a glance: WHO IS WINNING?
//
//   Section 1 — Headline scoreboard:
//       OURS [N] vs [N] DEEPGRAM     (with ties displayed separately)
//
//   Section 2 — Per-category mini-scoreboard:
//       For each voice tag (Deep, Calm, Energetic, ...) a single horizontal
//       stacked bar where green = ours wins, red = their wins, grey = ties.
//       Tag name on the left, raw counts on the right. Sorted by where we're
//       weakest first, so failure modes pop to the top.
//
//   Section 3 — WER distribution histogram (numerical, secondary signal).
//
// Source data: most-recent eval run + the human verdicts you saved on /validate.

const corpusLangFor = (apiLang: string): string =>
  apiLang === "pt" ? "pt-BR" : apiLang === "en" ? "en-US" : apiLang

export default function Benchmark() {
  const evalsQ = useAsync(() => listEvals(), [])
  const latest: EvalSummary | null =
    evalsQ.status === "ok" && evalsQ.data.length > 0 ? evalsQ.data[0] : null

  const detailQ = useAsync(
    () => (latest ? getEvalDetail(latest.run_id) : Promise.reject("no eval")),
    [latest?.run_id ?? ""],
  )

  const lang = latest ? corpusLangFor(latest.language) : ""
  const valsQ = useAsync(
    () => (lang ? listValidations(lang) : Promise.reject("no lang")),
    [lang],
  )

  if (evalsQ.status === "loading" || detailQ.status === "loading" || valsQ.status === "loading") {
    return <Centered><Loader2 className="size-5 animate-spin" /> loading…</Centered>
  }
  if (!latest) {
    return (
      <Centered>
        <div className="space-y-2 text-center">
          <div>No eval runs yet.</div>
          <div className="font-mono text-xs text-muted-foreground">
            Run <code>python3 scripts/eval_pipeline.py</code> first.
          </div>
        </div>
      </Centered>
    )
  }
  if (detailQ.status === "error" || valsQ.status === "error") {
    return <Centered><div className="text-destructive">Failed to load benchmark data.</div></Centered>
  }

  const rows = detailQ.status === "ok" ? detailQ.data.rows : []
  const validations = valsQ.status === "ok" ? valsQ.data.validations : {}
  const verdictTotals = countVerdicts(validations)
  const totalJudged = Object.values(verdictTotals).reduce((a, b) => a + b, 0)

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      <div>
        <h1 className="font-mono text-2xl font-semibold tracking-tight">Benchmark · Ours vs Deepgram</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          source <Badge variant="outline" className="font-mono text-[10px]">{latest.tag || latest.run_id}</Badge>
          {" · "}{rows.length} files, {totalJudged} judged by you, model {latest.model}, {lang}
        </p>
      </div>

      <Scoreboard
        oursWins={verdictTotals.hyp}
        theirsWins={verdictTotals.ref}
        ties={verdictTotals.equivalent}
        bothWrong={verdictTotals.neither}
        unjudged={Math.max(0, rows.length - totalJudged)}
      />

      <CategoryScoreboards rows={rows} validations={validations} />

      <WerHistogram rows={rows} />
    </div>
  )
}

// ── Section 1 — One-glance scoreboard ─────────────────────────────────
function Scoreboard({
  oursWins, theirsWins, ties, bothWrong, unjudged,
}: {
  oursWins: number; theirsWins: number; ties: number; bothWrong: number; unjudged: number
}) {
  const decided = oursWins + theirsWins
  const winner: "ours" | "theirs" | "tied" =
    oursWins > theirsWins ? "ours" : theirsWins > oursWins ? "theirs" : "tied"
  const margin = Math.abs(oursWins - theirsWins)
  const total = oursWins + theirsWins + ties + bothWrong

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        {/* Big number row */}
        <div className="grid grid-cols-3 items-center gap-4 bg-card px-6 py-8">
          <Side
            label="OURS"
            score={oursWins}
            isWinner={winner === "ours"}
            color="text-emerald-400"
            ring="ring-emerald-500/40"
          />
          <div className="flex flex-col items-center justify-center font-mono text-muted-foreground">
            <div className="text-xs uppercase tracking-wider">vs</div>
            {decided > 0 ? (
              <div className="mt-1 text-[10px]">
                {winner === "tied" ? (
                  "dead heat"
                ) : (
                  <>
                    <span className="text-foreground">+{margin}</span> for{" "}
                    <span className="text-foreground">{winner === "ours" ? "ours" : "Deepgram"}</span>
                  </>
                )}
              </div>
            ) : (
              <div className="mt-1 text-[10px]">no verdicts yet</div>
            )}
          </div>
          <Side
            label="DEEPGRAM"
            score={theirsWins}
            isWinner={winner === "theirs"}
            color="text-sky-400"
            ring="ring-sky-500/40"
          />
        </div>

        {/* Stacked bar — width-proportional view of every verdict bucket. */}
        {total > 0 && (
          <div>
            <div className="flex h-3 w-full overflow-hidden">
              <Seg w={oursWins / total} cls="bg-emerald-500/70" title={`${oursWins} ours`} />
              <Seg w={ties / total} cls="bg-muted-foreground/30" title={`${ties} ties`} />
              <Seg w={bothWrong / total} cls="bg-rose-700/60" title={`${bothWrong} both wrong`} />
              <Seg w={theirsWins / total} cls="bg-sky-500/70" title={`${theirsWins} Deepgram`} />
              <Seg w={unjudged / (total + unjudged)} cls="bg-muted/40" title={`${unjudged} unjudged`} />
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-6 py-2 font-mono text-[10px] text-muted-foreground">
              <Legend cls="bg-emerald-500/70" label={`${oursWins} ours`} />
              <Legend cls="bg-sky-500/70" label={`${theirsWins} Deepgram`} />
              <Legend cls="bg-muted-foreground/30" label={`${ties} tied (equivalent)`} />
              {bothWrong > 0 && <Legend cls="bg-rose-700/60" label={`${bothWrong} both wrong`} />}
              {unjudged > 0 && <Legend cls="bg-muted/40" label={`${unjudged} not judged yet`} />}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Side({
  label, score, isWinner, color, ring,
}: { label: string; score: number; isWinner: boolean; color: string; ring: string }) {
  return (
    <div className={`flex flex-col items-center justify-center rounded-lg p-4 transition ${isWinner ? `bg-foreground/5 ring-2 ${ring}` : ""}`}>
      <div className={`flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider ${color}`}>
        {isWinner && <Crown className="size-3.5" />}
        {label}
      </div>
      <div className={`mt-1 font-mono text-6xl font-bold tabular-nums ${color}`}>
        {score}
      </div>
      <div className="mt-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        wins
      </div>
    </div>
  )
}

function Seg({ w, cls, title }: { w: number; cls: string; title: string }) {
  if (w <= 0) return null
  return <div className={cls} style={{ width: `${w * 100}%` }} title={title} />
}

function Legend({ cls, label }: { cls: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`inline-block size-2 rounded-sm ${cls}`} />
      {label}
    </span>
  )
}

// ── Section 2 — Per-category mini scoreboards ─────────────────────────
function CategoryScoreboards({
  rows, validations,
}: { rows: EvalRow[]; validations: Record<string, ValidationEntry> }) {
  const cats = useMemo(() => buildCategoryStats(rows, validations), [rows, validations])
  // Surface failure modes first: where we lose the most relative to wins.
  const sorted = useMemo(() => {
    return [...cats].sort((a, b) => {
      // "we're losing" score: theirs - ours, larger first.
      const lossA = a.theirsWins - a.oursWins
      const lossB = b.theirsWins - b.oursWins
      if (lossA !== lossB) return lossB - lossA
      return b.judged - a.judged
    })
  }, [cats])

  if (sorted.length === 0) {
    return (
      <Card>
        <CardContent className="px-4 py-6 text-center font-mono text-[11px] text-muted-foreground">
          No category data yet.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
          By voice characteristic
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="font-mono text-[10px] text-muted-foreground">
          tags mined from filenames · longer green = we own this category · longer blue = Deepgram does
        </div>
        {sorted.map((c) => <CatRow key={c.tag} c={c} />)}
      </CardContent>
    </Card>
  )
}

function CatRow({ c }: { c: CategoryStat }) {
  const total = c.oursWins + c.theirsWins + c.ties
  const verdict =
    total === 0 ? null
      : c.oursWins > c.theirsWins ? "ours"
      : c.theirsWins > c.oursWins ? "theirs"
      : "tied"

  return (
    <div className="flex items-center gap-3">
      <div className="flex w-32 shrink-0 items-center justify-between gap-2">
        <span className="font-mono text-xs">{c.tag}</span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {c.count} {c.count === 1 ? "file" : "files"}
        </span>
      </div>
      <div className="relative h-6 flex-1 overflow-hidden rounded bg-muted/30">
        {total === 0 ? (
          <span className="absolute inset-0 flex items-center px-2 font-mono text-[10px] text-muted-foreground">
            no verdicts
          </span>
        ) : (
          <>
            <div className="flex h-full">
              <Seg w={c.oursWins / total} cls="bg-emerald-500/70" title={`${c.oursWins} ours`} />
              <Seg w={c.ties / total} cls="bg-muted-foreground/30" title={`${c.ties} tied`} />
              <Seg w={c.theirsWins / total} cls="bg-sky-500/70" title={`${c.theirsWins} Deepgram`} />
            </div>
            <span className="pointer-events-none absolute inset-0 flex items-center justify-between px-2 font-mono text-[10px] text-foreground">
              <span>
                {c.oursWins > 0 && <span className="text-emerald-300">{c.oursWins}</span>}
                {c.ties > 0 && <span className="ml-1 text-muted-foreground">·{c.ties}</span>}
              </span>
              <span>
                {c.theirsWins > 0 && <span className="text-sky-300">{c.theirsWins}</span>}
              </span>
            </span>
          </>
        )}
      </div>
      <div className="w-20 shrink-0 text-right font-mono text-[10px]">
        {verdict === "ours" && <span className="text-emerald-400">we win</span>}
        {verdict === "theirs" && <span className="text-sky-400">they win</span>}
        {verdict === "tied" && <span className="text-muted-foreground">tied</span>}
        {verdict === null && <span className="text-muted-foreground">—</span>}
      </div>
    </div>
  )
}

// ── Section 3 — WER distribution histogram ────────────────────────────
function WerHistogram({ rows }: { rows: EvalRow[] }) {
  const buckets = useMemo(() => {
    const ranges: Array<[number, number, string]> = [
      [0, 0.02, "0-2%"],
      [0.02, 0.05, "2-5%"],
      [0.05, 0.10, "5-10%"],
      [0.10, 0.20, "10-20%"],
      [0.20, 0.30, "20-30%"],
      [0.30, 1.01, "30%+"],
    ]
    return ranges.map(([lo, hi, label]) => ({
      label,
      count: rows.filter((r) => r.wer != null && r.wer >= lo && r.wer < hi).length,
      lo, hi,
    }))
  }, [rows])
  const max = Math.max(1, ...buckets.map((b) => b.count))
  const total = rows.filter((r) => r.wer != null).length

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
          WER distribution (text-distance vs Deepgram)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="font-mono text-[10px] text-muted-foreground">
          remember: low WER ≠ correct, since Deepgram itself is noisy. The scoreboard above is the trustworthy view.
        </div>
        <div className="mt-3 space-y-1">
          {buckets.map((b) => (
            <div key={b.label} className="flex items-center gap-2 font-mono text-[11px]">
              <span className="w-12 text-right text-muted-foreground">{b.label}</span>
              <div className="relative h-5 flex-1 overflow-hidden rounded bg-muted/40">
                <div
                  className={`h-full ${
                    b.lo < 0.10 ? "bg-emerald-500/60" : b.lo < 0.20 ? "bg-amber-500/60" : "bg-rose-500/60"
                  }`}
                  style={{ width: `${(b.count / max) * 100}%` }}
                />
                <span className="absolute inset-0 flex items-center px-2 text-foreground">
                  {b.count} {b.count === 1 ? "file" : "files"}
                  {total > 0 && <span className="ml-2 text-muted-foreground">({((b.count / total) * 100).toFixed(0)}%)</span>}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// ── helpers ──────────────────────────────────────────────────────────
function countVerdicts(vals: Record<string, ValidationEntry>) {
  const out: Record<Verdict, number> = { ref: 0, hyp: 0, neither: 0, equivalent: 0 }
  for (const v of Object.values(vals)) if (v.verdict in out) out[v.verdict]++
  return out
}

function tagsFromFilename(name: string): string[] {
  const stem = name.replace(/\.wav$/i, "")
  const idx = stem.indexOf("_-_")
  const tail = idx >= 0 ? stem.slice(idx + 3) : ""
  if (!tail) return []
  const raw = tail
    .split(/_|,| /)
    .map((s) => s.trim())
    .filter((s) => s.length > 2 && !["and", "the", "for"].includes(s.toLowerCase()))
  return Array.from(new Set(raw.map((s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase())))
}

type CategoryStat = {
  tag: string
  count: number          // files in this tag
  judged: number         // files in this tag with a human verdict
  oursWins: number
  theirsWins: number
  ties: number           // equivalent
}

function buildCategoryStats(
  rows: EvalRow[],
  validations: Record<string, ValidationEntry>,
): CategoryStat[] {
  const byTag: Map<string, CategoryStat> = new Map()
  for (const r of rows) {
    const tags = tagsFromFilename(r.file)
    for (const tag of tags) {
      let s = byTag.get(tag)
      if (!s) {
        s = { tag, count: 0, judged: 0, oursWins: 0, theirsWins: 0, ties: 0 }
        byTag.set(tag, s)
      }
      s.count += 1
      const v = validations[r.file]
      if (v) {
        s.judged += 1
        if (v.verdict === "hyp") s.oursWins += 1
        else if (v.verdict === "ref") s.theirsWins += 1
        else if (v.verdict === "equivalent") s.ties += 1
      }
    }
  }
  // Drop very-rare tags (1 file) so the table isn't a wall of singletons.
  return Array.from(byTag.values()).filter((c) => c.count >= 2)
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[calc(100vh-120px)] items-center justify-center font-mono text-sm text-muted-foreground">
      <div className="flex items-center gap-2">{children}</div>
    </div>
  )
}
