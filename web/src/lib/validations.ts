// Client for /api/validations/*. Stores human verdicts on each corpus file.

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? ""

export type Verdict = "ref" | "hyp" | "neither" | "equivalent"

export type ValidationEntry = {
  verdict: Verdict
  corrected_text: string | null
  validated_at: string
  validator: string
  notes: string
}

export type ValidationStats = {
  total_files: number
  validated: number
  remaining: number
  verdict_counts: Record<Verdict, number>
}

export type ValidationsResponse = {
  lang: string
  validations: Record<string, ValidationEntry>
  stats: ValidationStats
}

export async function listValidations(lang: string): Promise<ValidationsResponse> {
  const res = await fetch(`${API_BASE}/api/validations/${encodeURIComponent(lang)}`)
  if (!res.ok) throw new Error(`GET /api/validations/${lang} → ${res.status}`)
  return res.json()
}

export async function saveValidation(
  lang: string,
  filename: string,
  verdict: Verdict,
  opts?: { corrected_text?: string; validator?: string; notes?: string },
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/validations/${encodeURIComponent(lang)}/${encodeURIComponent(filename)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verdict, ...opts }),
  })
  if (!res.ok) throw new Error(`POST validation → ${res.status}`)
}

export async function deleteValidation(lang: string, filename: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/validations/${encodeURIComponent(lang)}/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error(`DELETE validation → ${res.status}`)
}

export type AutoEquivResult = {
  ok: boolean
  matched: number
  skipped_existing: number
  skipped_different: number
}

export async function autoEquivalent(
  lang: string,
  pairs: Array<{ file: string; ref_text: string; hyp_text: string }>,
): Promise<AutoEquivResult> {
  const res = await fetch(`${API_BASE}/api/validations/${encodeURIComponent(lang)}/auto-equivalent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pairs }),
  })
  if (!res.ok) throw new Error(`POST auto-equivalent → ${res.status}`)
  return res.json()
}
