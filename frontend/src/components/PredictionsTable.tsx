import type { Prediction } from "../api/types";
import { AlertTriangle, CheckCircle } from "lucide-react";

interface Props {
  predictions: Prediction[];
}

function probColor(p: number, label: string): string {
  if (label === "No Finding") {
    if (p >= 0.7) return "bg-accent-500";
    if (p >= 0.4) return "bg-amber-400";
    return "bg-danger-500";
  }
  if (p >= 0.7) return "bg-danger-500";
  if (p >= 0.4) return "bg-amber-400";
  return "bg-accent-500";
}

function probTextColor(p: number, label: string): string {
  if (label === "No Finding") {
    if (p >= 0.7) return "text-accent-600";
    if (p >= 0.4) return "text-amber-600";
    return "text-danger-600";
  }
  if (p >= 0.7) return "text-danger-600";
  if (p >= 0.4) return "text-amber-600";
  return "text-accent-600";
}

export default function PredictionsTable({ predictions }: Props) {
  const sorted = [...predictions].sort(
    (a, b) => b.probability - a.probability,
  );

  return (
    <div className="rounded-xl border border-gray-200 bg-white">
      <div className="border-b border-gray-100 px-4 py-3">
        <h3 className="text-sm font-semibold text-gray-700">
          Model Predictions
        </h3>
      </div>

      <div className="divide-y divide-gray-50">
        {sorted.map((p) => (
          <div
            key={p.label}
            className="flex items-center gap-3 px-4 py-2.5 transition hover:bg-gray-50"
          >
            {p.label === "No Finding" ? (
              p.positive ? (
                <CheckCircle className="h-4 w-4 shrink-0 text-accent-500" />
              ) : (
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500" />
              )
            ) : p.positive ? (
              <AlertTriangle className="h-4 w-4 shrink-0 text-danger-500" />
            ) : (
              <CheckCircle className="h-4 w-4 shrink-0 text-accent-500" />
            )}

            <span className="min-w-[140px] text-sm font-medium text-gray-700">
              {p.label}
            </span>

            <div className="flex-1">
              <div className="h-2 overflow-hidden rounded-full bg-gray-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${probColor(p.probability, p.label)}`}
                  style={{ width: `${(p.probability * 100).toFixed(0)}%` }}
                />
              </div>
            </div>

            <span
              className={`min-w-[48px] text-right text-xs font-bold ${probTextColor(p.probability, p.label)}`}
            >
              {(p.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
