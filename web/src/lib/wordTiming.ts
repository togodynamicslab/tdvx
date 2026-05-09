// Helpers to derive word-level timing for the validate UI's REF and HYP panels.
//
// REF: comes from a saved Deepgram response. They give per-word { word, start, end }.
// HYP: we only have segment-level { start, end, text } from Whisper. We split the
//      segment text into words and distribute time linearly across them. Less
//      accurate than real word timestamps but good enough to highlight the
//      currently-spoken token within ~200ms.

import type { HypSegment } from "@/lib/evals"

export type TimedWord = {
  word: string         // raw token (un-normalized — for display matching only)
  start: number
  end: number
}

// Match the normalizer the diff library uses so timed words line up with
// diff-token indices. Keep these two functions in sync with wordDiff.ts.
function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[.,!?;:""''`´()\[\]{}—–\-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function tokenize(s: string): string[] {
  const n = normalize(s)
  return n ? n.split(" ") : []
}

// ── Deepgram reference shape ─────────────────────────────────────────────
// We only depend on the bits we use. Other fields are present but ignored.
type DeepgramWord = { word: string; start: number; end: number; punctuated_word?: string }
type DeepgramAlt = { transcript?: string; words?: DeepgramWord[] }
type DeepgramShape = {
  results?: { channels?: Array<{ alternatives?: DeepgramAlt[] }> }
}

export function refTimedWords(deepgramJson: unknown): TimedWord[] {
  const dg = deepgramJson as DeepgramShape
  const alt = dg?.results?.channels?.[0]?.alternatives?.[0]
  const words = alt?.words ?? []
  return words.map((w) => ({
    word: w.punctuated_word ?? w.word,
    start: w.start,
    end: w.end,
  }))
}

// HYP: linearly distribute each segment's [start, end] over its tokens.
export function hypTimedWords(segments: HypSegment[] | undefined): TimedWord[] {
  if (!segments || segments.length === 0) return []
  const out: TimedWord[] = []
  for (const seg of segments) {
    const toks = (seg.text || "").trim().split(/\s+/).filter(Boolean)
    if (toks.length === 0) continue
    const dur = Math.max(0.001, seg.end - seg.start)
    const per = dur / toks.length
    for (let i = 0; i < toks.length; i++) {
      out.push({
        word: toks[i],
        start: seg.start + i * per,
        end: seg.start + (i + 1) * per,
      })
    }
  }
  return out
}

// Given a sequence of TimedWord and the current playback time, return the
// index of the active word, or -1 if none is currently active.
// We bias slightly toward "still active for ~150ms after end" so brief gaps
// between words don't make the highlight flicker.
export function activeIndex(words: TimedWord[], t: number, holdoverMs = 150): number {
  if (words.length === 0) return -1
  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    if (t >= w.start && t <= w.end + holdoverMs / 1000) return i
  }
  return -1
}

// Given the diff op list and a timed-word sequence for ONE side, build a
// parallel array (one entry per op) telling us "what diff op index does
// timed word #k correspond to" — so when we know the active word, we can
// highlight the right rendered cell in the DiffPanel.
//
// Strategy: walk the diff ops in order; for each op that produces a token
// on this side, advance one timed-word index. Returns: opIndex[wordIdx].
export function mapTimedWordsToOps(
  ops: ReadonlyArray<{ kind: string }>,
  side: "ref" | "hyp",
): number[] {
  const out: number[] = []
  ops.forEach((op, opIdx) => {
    const hasTokenOnThisSide =
      op.kind === "equal" ||
      op.kind === "sub" ||
      (side === "ref" && op.kind === "del") ||
      (side === "hyp" && op.kind === "ins")
    if (hasTokenOnThisSide) out.push(opIdx)
  })
  return out
}

export { tokenize }
