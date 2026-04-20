// Compute a word-level diff between reference and hypothesis strings.
//
// Uses Wagner-Fischer (Levenshtein on tokens) to recover the alignment, then
// emits an ordered ops list the UI renders as parallel colored rows.
//
// Ops:
//   - "equal": ref word == hyp word
//   - "sub":   ref word ≠ hyp word but they're aligned at the same position
//   - "del":   ref has a word, hyp doesn't (we missed it)
//   - "ins":   hyp has a word, ref doesn't (we added it)
//
// This matches the WER definition jiwer/sclite use, so the count of non-equal
// ops divided by ref-word-count equals the row's WER score.

export type DiffOp =
  | { kind: "equal"; ref: string; hyp: string; refIdx: number; hypIdx: number }
  | { kind: "sub";   ref: string; hyp: string; refIdx: number; hypIdx: number }
  | { kind: "del";   ref: string;              refIdx: number }
  | { kind: "ins";                hyp: string; hypIdx: number }

export type DiffResult = {
  ops: DiffOp[]
  refTokens: string[]
  hypTokens: string[]
  counts: { equal: number; sub: number; del: number; ins: number }
}

// Match the normalization the server applies for WER scoring so visual diff
// agrees with the numeric WER. Lowercase + strip punctuation + collapse spaces.
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

export function diffWords(ref: string, hyp: string): DiffResult {
  const refTokens = tokenize(ref)
  const hypTokens = tokenize(hyp)
  const m = refTokens.length
  const n = hypTokens.length

  // dp[i][j] = edit distance between refTokens[0..i) and hypTokens[0..j)
  // Allocate as flat Int32Array for speed on long sequences.
  const dp = new Int32Array((m + 1) * (n + 1))
  const idx = (i: number, j: number) => i * (n + 1) + j
  for (let i = 0; i <= m; i++) dp[idx(i, 0)] = i
  for (let j = 0; j <= n; j++) dp[idx(0, j)] = j
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const same = refTokens[i - 1] === hypTokens[j - 1]
      const subCost = dp[idx(i - 1, j - 1)] + (same ? 0 : 1)
      const delCost = dp[idx(i - 1, j)] + 1
      const insCost = dp[idx(i, j - 1)] + 1
      dp[idx(i, j)] = Math.min(subCost, delCost, insCost)
    }
  }

  // Backtrack to recover the ordered ops.
  const ops: DiffOp[] = []
  let i = m, j = n
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refTokens[i - 1] === hypTokens[j - 1]) {
      ops.push({ kind: "equal", ref: refTokens[i - 1], hyp: hypTokens[j - 1], refIdx: i - 1, hypIdx: j - 1 })
      i--; j--
      continue
    }
    const here = dp[idx(i, j)]
    if (i > 0 && j > 0 && dp[idx(i - 1, j - 1)] + 1 === here) {
      ops.push({ kind: "sub", ref: refTokens[i - 1], hyp: hypTokens[j - 1], refIdx: i - 1, hypIdx: j - 1 })
      i--; j--
    } else if (i > 0 && dp[idx(i - 1, j)] + 1 === here) {
      ops.push({ kind: "del", ref: refTokens[i - 1], refIdx: i - 1 })
      i--
    } else {
      ops.push({ kind: "ins", hyp: hypTokens[j - 1], hypIdx: j - 1 })
      j--
    }
  }
  ops.reverse()

  const counts = { equal: 0, sub: 0, del: 0, ins: 0 }
  for (const o of ops) counts[o.kind]++

  return { ops, refTokens, hypTokens, counts }
}
