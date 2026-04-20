import { useEffect, useState } from "react"

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ok"; data: T }

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: ReadonlyArray<unknown>,
): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" })
  useEffect(() => {
    let cancelled = false
    setState({ status: "loading" })
    fn()
      .then((data) => {
        if (!cancelled) setState({ status: "ok", data })
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setState({
            status: "error",
            error: e instanceof Error ? e.message : String(e),
          })
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}
