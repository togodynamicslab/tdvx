import { useEffect, useState } from "react"

export type Route =
  | { name: "live" }
  | { name: "youtube" }
  | { name: "runs" }
  | { name: "run"; id: string }
  | { name: "compare"; ids: string[] }
  | { name: "evals" }
  | { name: "validate" }
  | { name: "benchmark" }

function parse(hash: string): Route {
  const h = hash.replace(/^#\/?/, "")
  if (!h || h === "live") return { name: "live" }
  if (h === "youtube") return { name: "youtube" }
  if (h === "stress" || h === "runs") return { name: "runs" }
  if (h === "evals") return { name: "evals" }
  if (h === "validate") return { name: "validate" }
  if (h === "benchmark") return { name: "benchmark" }
  const runMatch = h.match(/^run\/([^/?]+)/)
  if (runMatch) return { name: "run", id: decodeURIComponent(runMatch[1]) }
  const compareMatch = h.match(/^compare\?ids=(.+)$/)
  if (compareMatch) {
    return {
      name: "compare",
      ids: decodeURIComponent(compareMatch[1]).split(",").filter(Boolean),
    }
  }
  return { name: "live" }
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const onHash = () => setRoute(parse(window.location.hash))
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])
  return route
}

export function navigate(path: string) {
  window.location.hash = path
}
