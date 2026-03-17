import { Activity, Github } from "lucide-react";
import type { ReactNode } from "react";
import type { HealthResponse } from "../api/types";

interface Props {
  children: ReactNode;
  health: HealthResponse | null;
}

export default function Layout({ children, health }: Props) {
  const ok = health?.model_loaded && health?.rag_ready;

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-gray-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <Activity className="h-6 w-6 text-primary-600" />
            <span className="text-lg font-bold tracking-tight text-primary-800">
              RadiantCare
            </span>
          </div>

          <div className="flex items-center gap-4">
            {health && (
              <div className="flex items-center gap-1.5 text-xs font-medium">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${ok ? "bg-accent-500" : "bg-danger-500"}`}
                />
                <span className={ok ? "text-accent-600" : "text-danger-600"}>
                  {ok ? "API Ready" : "API Offline"}
                </span>
              </div>
            )}

            <a
              href="https://github.com/SulaimanFY/"
              target="_blank"
              rel="noreferrer"
              className="text-gray-400 transition hover:text-gray-700"
            >
              <Github className="h-5 w-5" />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-gray-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-center gap-1 px-4 py-4 text-xs text-gray-400">
          SulaimanFY © 2026
        </div>
      </footer>
    </div>
  );
}
