import { Waves, FlaskConical, Mic, Gauge, Gamepad2, Trophy } from "lucide-react"

import { useRoute, navigate } from "@/lib/hashRouter"
import LiveConsole from "@/pages/LiveConsole"
import RunsList from "@/pages/RunsList"
import RunDetail from "@/pages/RunDetail"
import Compare from "@/pages/Compare"
import EvalsList from "@/pages/EvalsList"
import Validate from "@/pages/Validate"
import Benchmark from "@/pages/Benchmark"

function NavItem({
  active,
  onClick,
  icon: Icon,
  children,
}: {
  active: boolean
  onClick: () => void
  icon: React.ComponentType<{ className?: string }>
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 font-mono text-xs transition-colors ${
        active ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
      }`}
    >
      <Icon className="size-3.5" />
      {children}
    </button>
  )
}

export default function App() {
  const route = useRoute()
  const isLive = route.name === "live"
  const isEvals = route.name === "evals"
  const isValidate = route.name === "validate"
  const isBenchmark = route.name === "benchmark"
  const isStress = !isLive && !isEvals && !isValidate && !isBenchmark

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-20 border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-md bg-foreground text-background">
              <Waves className="size-4" />
            </div>
            <div>
              <div className="font-mono text-sm font-semibold tracking-tight">
                TDvX · Research Console
              </div>
              <div className="font-mono text-[11px] text-muted-foreground">
                transcription · diarization · load testing
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1">
            <NavItem active={isLive} onClick={() => navigate("live")} icon={Mic}>
              Live
            </NavItem>
            <NavItem active={isBenchmark} onClick={() => navigate("benchmark")} icon={Trophy}>
              Benchmark
            </NavItem>
            <NavItem active={isEvals} onClick={() => navigate("evals")} icon={Gauge}>
              Evals
            </NavItem>
            <NavItem active={isValidate} onClick={() => navigate("validate")} icon={Gamepad2}>
              Validate
            </NavItem>
            <NavItem active={isStress} onClick={() => navigate("runs")} icon={FlaskConical}>
              Stress tests
            </NavItem>
          </nav>
        </div>
      </header>

      {route.name === "live" && <LiveConsole />}
      {route.name === "benchmark" && <Benchmark />}
      {route.name === "evals" && <EvalsList />}
      {route.name === "validate" && <Validate />}
      {route.name === "runs" && <RunsList />}
      {route.name === "run" && <RunDetail runId={route.id} />}
      {route.name === "compare" && <Compare ids={route.ids} />}
    </div>
  )
}
