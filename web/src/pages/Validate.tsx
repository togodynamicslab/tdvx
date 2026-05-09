import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { CheckCircle2, ChevronLeft, ChevronRight, Equal, Loader2, Play, Sparkles, X } from "lucide-react"

import { useAsync } from "@/lib/useAsync"
import { listEvals, getEvalDetail, type EvalRow, type EvalSummary } from "@/lib/evals"
import {
  listValidations,
  saveValidation,
  autoEquivalent,
  type Verdict,
  type ValidationEntry,
} from "@/lib/validations"
import { diffWords, type DiffOp } from "@/lib/wordDiff"
import {
  refTimedWords,
  hypTimedWords,
  activeIndex,
  mapTimedWordsToOps,
  type TimedWord,
} from "@/lib/wordTiming"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

// One-file-at-a-time validation UI. Listen → judge → next.
// Keyboard:
//   1   ref correct        2   hyp correct
//   3   neither            4   equivalent (punctuation/casing only)
//   ←   previous file      →   next file (after save)
//   ↵   save current and advance
//   space  play/pause
//
// Source data: the most recent eval run that has the corpus we want.

type Mode = "loading" | "ready" | "no-evals" | "error"

const corpusLangFor = (apiLang: string): string =>
  apiLang === "pt" ? "pt-BR" : apiLang === "en" ? "en-US" : apiLang

export default function Validate() {
  // 1. Pull the eval list, pick the most recent run for the source corpus.
  const evalsQuery = useAsync(() => listEvals(), [])
  const latestEval: EvalSummary | null =
    evalsQuery.status === "ok" && evalsQuery.data.length > 0 ? evalsQuery.data[0] : null

  // 2. Once we have an eval, pull its detail (the per-file rows with ref/hyp).
  const detailQuery = useAsync(
    () => (latestEval ? getEvalDetail(latestEval.run_id) : Promise.reject("no eval")),
    [latestEval?.run_id ?? ""],
  )

  const lang = latestEval ? corpusLangFor(latestEval.language) : ""

  // 3. Pull existing validations for this lang (so we know which files are done).
  const valsQuery = useAsync(
    () => (lang ? listValidations(lang) : Promise.reject("no lang")),
    [lang],
  )

  const mode: Mode = (() => {
    if (evalsQuery.status === "loading" || detailQuery.status === "loading" || valsQuery.status === "loading") return "loading"
    if (evalsQuery.status === "error" || detailQuery.status === "error" || valsQuery.status === "error") return "error"
    if (!latestEval) return "no-evals"
    return "ready"
  })()

  if (mode === "loading") {
    return <Centered><Loader2 className="size-5 animate-spin" /> loading…</Centered>
  }
  if (mode === "no-evals") {
    return (
      <Centered>
        <div className="space-y-2 text-center">
          <div>No eval runs yet.</div>
          <div className="font-mono text-xs text-muted-foreground">
            Run <code>python3 scripts/eval_pipeline.py --tag baseline</code> to create one.
          </div>
        </div>
      </Centered>
    )
  }
  if (mode === "error") {
    return <Centered><div className="text-destructive">Failed to load eval data.</div></Centered>
  }

  return (
    <Game
      lang={lang}
      rows={detailQuery.status === "ok" ? detailQuery.data.rows : []}
      initialValidations={valsQuery.status === "ok" ? valsQuery.data.validations : {}}
      tag={latestEval!.tag || latestEval!.run_id}
    />
  )
}

function Game({
  lang,
  rows,
  initialValidations,
  tag,
}: {
  lang: string
  rows: EvalRow[]
  initialValidations: Record<string, ValidationEntry>
  tag: string
}) {
  // Local mirror so we can show progress without re-fetching after every save.
  const [validations, setValidations] = useState<Record<string, ValidationEntry>>(initialValidations)
  const [idx, setIdx] = useState(() => {
    // Start at the first un-validated file. If all are done, start at 0.
    const i = rows.findIndex((r) => !initialValidations[r.file])
    return i === -1 ? 0 : i
  })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoBusy, setAutoBusy] = useState(false)
  const [autoMsg, setAutoMsg] = useState<string | null>(null)

  const runAutoEquiv = useCallback(async () => {
    setAutoBusy(true); setAutoMsg(null); setError(null)
    try {
      const pairs = rows.map((r) => ({
        file: r.file,
        ref_text: r.ref_text || "",
        hyp_text: r.hyp_text || "",
      }))
      const res = await autoEquivalent(lang, pairs)
      // Re-fetch our local mirror to reflect the bulk insert.
      const fresh = await listValidations(lang)
      setValidations(fresh.validations)
      // Jump to the first remaining un-validated file so the human keeps moving.
      const i = rows.findIndex((r) => !fresh.validations[r.file])
      if (i >= 0) setIdx(i)
      setAutoMsg(`auto-marked ${res.matched} as equivalent · ${res.skipped_existing} kept (already judged) · ${res.skipped_different} need review`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setAutoBusy(false)
    }
  }, [rows, lang])

  const cur = rows[idx]
  const curVerdict: Verdict | null = cur ? (validations[cur.file]?.verdict as Verdict | undefined) ?? null : null

  const goPrev = useCallback(() => setIdx((i) => Math.max(0, i - 1)), [])
  const goNext = useCallback(() => setIdx((i) => Math.min(rows.length - 1, i + 1)), [rows.length])

  const submit = useCallback(
    async (verdict: Verdict) => {
      if (!cur) return
      setSaving(true)
      setError(null)
      try {
        await saveValidation(lang, cur.file, verdict)
        setValidations((v) => ({
          ...v,
          [cur.file]: {
            verdict,
            corrected_text: null,
            validated_at: new Date().toISOString(),
            validator: "human",
            notes: "",
          },
        }))
        // Auto-advance to the next un-validated file.
        const nextUnvalidated = rows.findIndex((r, i) => i > idx && !validations[r.file] && r.file !== cur.file)
        setIdx(nextUnvalidated === -1 ? Math.min(rows.length - 1, idx + 1) : nextUnvalidated)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setSaving(false)
      }
    },
    [cur, lang, rows, idx, validations],
  )

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore when typing in an input/textarea.
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return
      if (e.key === "1") { e.preventDefault(); submit("ref") }
      else if (e.key === "2") { e.preventDefault(); submit("hyp") }
      else if (e.key === "3") { e.preventDefault(); submit("neither") }
      else if (e.key === "4") { e.preventDefault(); submit("equivalent") }
      else if (e.key === "ArrowLeft") { e.preventDefault(); goPrev() }
      else if (e.key === "ArrowRight") { e.preventDefault(); goNext() }
      else if (e.key === " ") {
        e.preventDefault()
        const a = document.getElementById("validate-audio") as HTMLAudioElement | null
        if (a) {
          if (a.paused) void a.play()
          else a.pause()
        }
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [submit, goPrev, goNext])

  if (!cur) {
    return <Centered>No files in this run.</Centered>
  }

  const validatedCount = Object.keys(validations).length
  const total = rows.length
  const progressPct = total > 0 ? (validatedCount / total) * 100 : 0
  const audioSrc = `/api/corpus/${encodeURIComponent(lang)}/${encodeURIComponent(cur.file)}`

  // Quick visual diff — same algo as the eval page.
  const diff = useMemo(() => diffWords(cur.ref_text, cur.hyp_text), [cur.ref_text, cur.hyp_text])

  // ── Playback-synced word highlighting ──────────────────────────────
  // Pull the saved Deepgram reference for per-word REF timestamps. HYP
  // timing is approximated from segment-level data already in the eval row.
  const refJsonQuery = useAsync(
    () => fetch(`/api/corpus/${encodeURIComponent(lang)}/reference/${encodeURIComponent(cur.file)}`).then((r) => r.ok ? r.json() : Promise.reject(r.status)),
    [lang, cur.file],
  )
  const refTimes: TimedWord[] = useMemo(
    () => (refJsonQuery.status === "ok" ? refTimedWords(refJsonQuery.data) : []),
    [refJsonQuery],
  )
  const hypTimes: TimedWord[] = useMemo(
    () => hypTimedWords(cur.hyp_segments),
    [cur.hyp_segments],
  )
  // Map "timed word index" → "diff op index" for each side, so we can light
  // up the right rendered cell when a word is currently being spoken.
  const refOpIdx = useMemo(() => mapTimedWordsToOps(diff.ops, "ref"), [diff.ops])
  const hypOpIdx = useMemo(() => mapTimedWordsToOps(diff.ops, "hyp"), [diff.ops])

  const currentTime = useAudioCurrentTime("validate-audio", cur.file)
  const refActiveOpIdx = useMemo(() => {
    const wIdx = activeIndex(refTimes, currentTime)
    return wIdx >= 0 && wIdx < refOpIdx.length ? refOpIdx[wIdx] : -1
  }, [refTimes, currentTime, refOpIdx])
  const hypActiveOpIdx = useMemo(() => {
    const wIdx = activeIndex(hypTimes, currentTime)
    return wIdx >= 0 && wIdx < hypOpIdx.length ? hypOpIdx[wIdx] : -1
  }, [hypTimes, currentTime, hypOpIdx])

  return (
    <div className="mx-auto flex min-h-[calc(100vh-60px)] max-w-4xl flex-col gap-4 px-6 py-6">
      {/* Header: progress + run tag */}
      <div className="space-y-2">
        <div className="flex items-center justify-between font-mono text-[11px] text-muted-foreground">
          <span>
            file <span className="text-foreground">{idx + 1}</span> / {total} ·
            validated <span className="text-foreground">{validatedCount}</span> · run <Badge variant="outline" className="ml-1 font-mono text-[10px]">{tag}</Badge>
          </span>
          <span className="flex items-center gap-3">
            <span>
              ref {diff.refTokens.length}w · hyp {diff.hypTokens.length}w · WER{" "}
              <span>
                {((diff.counts.sub + diff.counts.del + diff.counts.ins) / Math.max(1, diff.refTokens.length) * 100).toFixed(1)}%
              </span>
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={runAutoEquiv}
              disabled={autoBusy}
              className="h-6 gap-1 px-2 font-mono text-[10px]"
              title="Auto-mark trivially-equivalent diffs (case/punctuation/contractions) so you only judge the meaningful ones"
            >
              {autoBusy ? <Loader2 className="size-3 animate-spin" /> : <Sparkles className="size-3" />}
              skip equivalents
            </Button>
          </span>
        </div>
        {autoMsg && (
          <div className="rounded border border-emerald-500/30 bg-emerald-500/5 px-2 py-1 font-mono text-[10px] text-emerald-300">
            {autoMsg}
          </div>
        )}
        <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
          <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progressPct}%` }} />
        </div>
      </div>

      {/* Filename + audio */}
      <div className="rounded-lg border bg-card/40 p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="font-mono text-sm">{cur.file}</div>
          {curVerdict && (
            <Badge variant="outline" className="font-mono text-[10px]">
              already: {curVerdict}
            </Badge>
          )}
        </div>
        <audio
          id="validate-audio"
          key={cur.file}
          controls
          autoPlay
          src={audioSrc}
          className="w-full"
        />
      </div>

      {/* Side-by-side ref vs hyp — words colored by diff op so differences pop. */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <DiffPanel
          label="REF (Deepgram)"
          subLabel="press 1 if this is correct"
          ops={diff.ops}
          side="ref"
          accent="border-emerald-500/30"
          highlight={curVerdict === "ref"}
          activeOpIdx={refActiveOpIdx}
        />
        <DiffPanel
          label="HYP (Our pipeline)"
          subLabel="press 2 if this is correct"
          ops={diff.ops}
          side="hyp"
          accent="border-sky-500/30"
          highlight={curVerdict === "hyp"}
          activeOpIdx={hypActiveOpIdx}
        />
      </div>

      {/* Legend: tells you what the word colors mean. */}
      <div className="flex flex-wrap items-center gap-3 px-1 font-mono text-[10px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span className="rounded border border-emerald-500/40 px-1 text-emerald-300">word</span> match
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="rounded border border-amber-500/50 px-1 text-amber-300">word</span> different (substitution)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="rounded border border-rose-500/60 px-1 text-rose-300">word</span> only on this side
        </span>
        <span className="ml-auto">
          equal {diff.counts.equal} · sub {diff.counts.sub} · del {diff.counts.del} · ins {diff.counts.ins}
        </span>
      </div>

      {/* Verdict buttons */}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <ChoiceBtn
          label="REF correct"
          hotkey="1"
          icon={CheckCircle2}
          color="text-emerald-400"
          active={curVerdict === "ref"}
          onClick={() => submit("ref")}
          disabled={saving}
        />
        <ChoiceBtn
          label="HYP correct"
          hotkey="2"
          icon={CheckCircle2}
          color="text-sky-400"
          active={curVerdict === "hyp"}
          onClick={() => submit("hyp")}
          disabled={saving}
        />
        <ChoiceBtn
          label="Neither"
          hotkey="3"
          icon={X}
          color="text-rose-400"
          active={curVerdict === "neither"}
          onClick={() => submit("neither")}
          disabled={saving}
        />
        <ChoiceBtn
          label="Equivalent"
          hotkey="4"
          icon={Equal}
          color="text-amber-400"
          active={curVerdict === "equivalent"}
          onClick={() => submit("equivalent")}
          disabled={saving}
        />
      </div>

      {/* Nav */}
      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={goPrev} disabled={idx === 0} className="font-mono text-xs">
          <ChevronLeft className="mr-1 size-4" /> prev (←)
        </Button>
        <span className="font-mono text-[10px] text-muted-foreground">
          space = play · 1-4 = verdict · ↵ enter = next
        </span>
        <Button variant="outline" onClick={goNext} disabled={idx === rows.length - 1} className="font-mono text-xs">
          next (→) <ChevronRight className="ml-1 size-4" />
        </Button>
      </div>

      {error && (
        <div className="rounded border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          save failed: {error}
        </div>
      )}

      <p className="mt-2 text-center font-mono text-[10px] text-muted-foreground">
        every verdict you save here becomes ground truth for future eval runs.
      </p>
    </div>
  )
}

// Renders one side of the diff (ref or hyp) as a sequence of colored word
// pills. Words present on both sides → emerald. Substitutions → amber. Words
// only on this side (deletions when rendering ref, insertions when rendering
// hyp) → rose. Words only on the *other* side render as a faint placeholder
// so positions line up visually with the opposite panel.
function DiffPanel({
  label,
  subLabel,
  ops,
  side,
  accent,
  highlight,
  activeOpIdx,
}: {
  label: string
  subLabel: string
  ops: DiffOp[]
  side: "ref" | "hyp"
  accent: string
  highlight: boolean
  activeOpIdx?: number
}) {
  const cells = ops.map((op, i) => {
    if (op.kind === "equal") {
      return { key: i, tok: side === "ref" ? op.ref : op.hyp, cls: "border-emerald-500/40 text-emerald-300/90" }
    }
    if (op.kind === "sub") {
      return {
        key: i,
        tok: side === "ref" ? op.ref : op.hyp,
        cls: "border-amber-500/60 text-amber-300",
        title: side === "ref" ? `→ ${op.hyp}` : `← ${op.ref}`,
      }
    }
    if (op.kind === "del") {
      return side === "ref"
        ? { key: i, tok: op.ref, cls: "border-rose-500/60 text-rose-300", title: "missing in hyp" }
        : { key: i, tok: "·", cls: "border-transparent text-muted-foreground/30" }
    }
    // ins
    return side === "hyp"
      ? { key: i, tok: op.hyp, cls: "border-rose-500/60 text-rose-300", title: "extra in hyp" }
      : { key: i, tok: "·", cls: "border-transparent text-muted-foreground/30" }
  })

  return (
    <div
      className={`rounded-lg border ${accent} bg-card/40 p-4 transition-colors ${highlight ? "ring-2 ring-foreground/40" : ""}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="font-mono text-[9px] text-muted-foreground/70">{subLabel}</div>
      </div>
      <div className="flex flex-wrap gap-1">
        {cells.length === 0 ? (
          <span className="font-mono text-sm text-muted-foreground">(empty)</span>
        ) : (
          cells.map((c) => {
            const isActive = activeOpIdx != null && c.key === activeOpIdx
            return (
              <span
                key={c.key}
                title={c.title}
                className={`rounded border px-1.5 py-0.5 font-mono text-sm transition-colors ${c.cls} ${
                  isActive ? "scale-110 bg-foreground/20 text-foreground shadow-[0_0_0_2px_currentColor]" : ""
                }`}
              >
                {c.tok}
              </span>
            )
          })
        )}
      </div>
    </div>
  )
}

// Subscribe to an audio element's currentTime via timeupdate events. The
// elementId is stable across files; the fileKey resets the value to 0 when
// we navigate to a new file (so highlight doesn't flash stale).
function useAudioCurrentTime(elementId: string, fileKey: string): number {
  const [t, setT] = useState(0)
  const fileRef = useRef(fileKey)
  useEffect(() => {
    fileRef.current = fileKey
    setT(0)
  }, [fileKey])
  useEffect(() => {
    const a = document.getElementById(elementId) as HTMLAudioElement | null
    if (!a) return
    const onTime = () => setT(a.currentTime)
    const onSeek = () => setT(a.currentTime)
    const onEnd = () => setT(0)
    a.addEventListener("timeupdate", onTime)
    a.addEventListener("seeked", onSeek)
    a.addEventListener("ended", onEnd)
    return () => {
      a.removeEventListener("timeupdate", onTime)
      a.removeEventListener("seeked", onSeek)
      a.removeEventListener("ended", onEnd)
    }
  }, [elementId, fileKey])
  return t
}

function ChoiceBtn({
  label,
  hotkey,
  icon: Icon,
  color,
  active,
  onClick,
  disabled,
}: {
  label: string
  hotkey: string
  icon: React.ComponentType<{ className?: string }>
  color: string
  active: boolean
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-between rounded-md border px-3 py-2 font-mono text-xs transition-all hover:border-foreground/40 disabled:opacity-50 ${
        active ? "border-foreground bg-foreground/10" : "border-muted"
      }`}
    >
      <span className="flex items-center gap-2">
        <Icon className={`size-3.5 ${color}`} />
        {label}
      </span>
      <kbd className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">{hotkey}</kbd>
    </button>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[calc(100vh-120px)] items-center justify-center font-mono text-sm text-muted-foreground">
      <div className="flex items-center gap-2">{children}</div>
    </div>
  )
}

// suppress "Play unused" by referencing it (kept for future ai-elements upgrade)
void Play
